"""
MPI backend for ChemFit's graph scheduler.

Rank 0 owns graph traversal and structural-component lifecycles. Other ranks
run a persistent worker loop and evaluate only AtomicNode tasks. Tasks are
queued on rank 0 and dispatched to idle workers one at a time, allowing new
work to become available as earlier graph nodes finish.

Ordinary component exceptions are returned in NodeCompletion. MPI/protocol
failures are fatal to the prepared schedule and trigger worker shutdown.

All ranks must prepare the same component graph; only rank 0 evaluates it.
Nonzero ranks call ``schedule.worker_loop()`` and rank 0 calls ``schedule.close()``
after the final evaluation to release the workers.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

try:
    from mpi4py import MPI
except ImportError as e:
    if hasattr(e, "add_note"):
        e.add_note(
            "Install ChemFit's MPI dependencies with `pip install chemfit[mpi]`."
        )
    raise

from chemfit.abstract_objective_function import (
    Component,
    InputT_contra,
    OutputT_co,
)
from chemfit.callgraph import AtomicNode, CallGraph, objective_to_call_graph
from chemfit.debug_utils import log_all_methods
from chemfit.graph_schedule_base import (
    BatchState,
    GraphScheduleBase,
    NodeCompletion,
    NodeOutcome,
    NodeTask,
)
from chemfit.scheduling import Scheduler

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_COMMAND_TAG = 41001
_RESULT_TAG = 41002
_CONTROL_TAG = 41003


@dataclass(frozen=True, slots=True)
class _EvaluateRequest:
    task: NodeTask


@dataclass(frozen=True, slots=True)
class _Shutdown:
    pass


@dataclass(frozen=True, slots=True)
class _ShutdownAcknowledgement:
    """Confirm that a worker has stopped receiving and evaluating tasks."""


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    rank: int
    exception_type: str
    message: str


class MPIWorkerError(RuntimeError):
    """Fatal MPI worker or scheduling protocol failure."""


def _describe_worker_failure(rank: int, exception: BaseException) -> _WorkerFailure:
    try:
        message = str(exception)
    except BaseException:
        message = "<exception message could not be formatted>"

    return _WorkerFailure(
        rank=rank,
        exception_type=type(exception).__qualname__,
        message=message,
    )


class MPIGraphSchedule(GraphScheduleBase):
    """Prepared graph schedule using persistent MPI workers."""

    def __init__(
        self,
        graph: CallGraph,
        comm: Any,
        *,
        mpi_debug_log: bool = False,
    ) -> None:
        """Bind a call graph to an MPI communicator."""
        super().__init__(graph)
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self._workers_released = False

        if mpi_debug_log:
            self.comm = log_all_methods(
                self.comm,
                log_func=self._log_func,
                log_args=True,
                log_res=True,
            )

    def _log_func(self, message: str) -> None:
        logger.warning("[Rank %s] %s", self.rank, message)

    def _ensure_coordinator(self) -> None:
        if self.rank != 0:
            msg = "Graph evaluation is only supported on MPI rank 0"
            raise RuntimeError(msg)
        if self.closed:
            msg = "Prepared MPI graph schedule is closed"
            raise RuntimeError(msg)

    def _ensure_batch_state(self, batch_state: BatchState) -> None:
        if not hasattr(batch_state, "pending_tasks"):
            batch_state.pending_tasks = deque()
            batch_state.in_flight = {}
            # Rank availability changes only on dispatch and completion, so an
            # explicit queue avoids scanning every rank on every refill.
            batch_state.idle_workers = deque(range(1, self.size))

    def _evaluate_worker_task(
        self, task: NodeTask, *, capture_context_state: bool
    ) -> NodeCompletion:
        node = self.graph.nodes[task.node_id]
        if not isinstance(node, AtomicNode):
            msg = f"MPI worker received non-atomic node {task.node_id}"
            raise MPIWorkerError(msg)

        try:
            outcome: NodeOutcome = node.objective(task.input, task.ctx)
        except Exception as exception:
            outcome = exception

        context_state = task.ctx.to_result_state() if capture_context_state else None
        return NodeCompletion(
            run_id=task.run_id,
            node_id=task.node_id,
            outcome=outcome,
            ctx_result_state=context_state,
        )

    def _dispatch_available(self, batch_state: BatchState) -> None:
        """Assign at most one outstanding task to each idle worker."""
        queue: deque[NodeTask] = batch_state.pending_tasks
        in_flight: dict[int, NodeTask] = batch_state.in_flight
        idle_workers: deque[int] = batch_state.idle_workers

        while queue and idle_workers:
            rank = idle_workers.popleft()
            task = queue.popleft()
            self.comm.send(_EvaluateRequest(task), dest=rank, tag=_COMMAND_TAG)
            in_flight[rank] = task

    def submit_tasks(
        self,
        tasks: Sequence[NodeTask],
        batch_state: BatchState,
    ) -> None:
        """Queue tasks and immediately give available workers more work."""
        self._ensure_coordinator()
        self._ensure_batch_state(batch_state)
        batch_state.pending_tasks.extend(tasks)

        if self.size > 1:
            self._dispatch_available(batch_state)

    def _receive_worker_completion(self, batch_state: BatchState) -> NodeCompletion:
        """Receive and validate one worker result, then reuse or idle its rank."""
        in_flight: dict[int, NodeTask] = batch_state.in_flight
        status = MPI.Status()
        message = self.comm.recv(
            source=MPI.ANY_SOURCE,
            tag=_RESULT_TAG,
            status=status,
        )
        source = status.Get_source()

        if isinstance(message, _WorkerFailure):
            msg = (
                f"Worker rank {message.rank} failed with "
                f"{message.exception_type}: {message.message}"
            )
            raise MPIWorkerError(msg)

        expected_task = in_flight.pop(source, None)
        if expected_task is None:
            msg = f"Unexpected completion from idle worker {source}"
            raise MPIWorkerError(msg)
        if not isinstance(message, NodeCompletion):
            msg = f"Unexpected MPI worker result: {message!r}"
            raise MPIWorkerError(msg)
        if (message.run_id, message.node_id) != (
            expected_task.run_id,
            expected_task.node_id,
        ):
            msg = (
                f"Worker {source} returned completion for unexpected task "
                f"({message.run_id}, {message.node_id})"
            )
            raise MPIWorkerError(msg)

        queue: deque[NodeTask] = batch_state.pending_tasks
        if queue:
            # Refill the rank that just replied directly. Returning it to and
            # immediately removing it from idle_workers would add bookkeeping
            # without exposing any scheduling choice.
            task = queue.popleft()
            self.comm.send(_EvaluateRequest(task), dest=source, tag=_COMMAND_TAG)
            in_flight[source] = task
        else:
            batch_state.idle_workers.append(source)

        return message

    def wait_for_completions(
        self,
        batch_state: BatchState,
    ) -> Sequence[NodeCompletion]:
        """Block for one MPI result and drain other results already available."""
        self._ensure_coordinator()
        self._ensure_batch_state(batch_state)

        if self.size == 1:
            if not batch_state.pending_tasks:
                msg = "No queued MPI tasks to complete"
                raise MPIWorkerError(msg)
            task = batch_state.pending_tasks.popleft()
            return (self._evaluate_worker_task(task, capture_context_state=False),)

        in_flight: dict[int, NodeTask] = batch_state.in_flight
        if not in_flight:
            msg = "No MPI tasks are in flight"
            raise MPIWorkerError(msg)

        # Bound draining by the number in flight at entry. Very short newly
        # dispatched work must not keep the coordinator in an unbounded receive
        # loop and delay graph dependency release.
        drain_limit = len(in_flight)
        completions = [self._receive_worker_completion(batch_state)]
        while len(completions) < drain_limit and self.comm.Iprobe(
            source=MPI.ANY_SOURCE,
            tag=_RESULT_TAG,
        ):
            completions.append(self._receive_worker_completion(batch_state))

        return completions

    def next_completion(self, batch_state: BatchState) -> NodeCompletion:
        """Return one task completion through the compatibility API."""
        if not hasattr(batch_state, "completion_buffer"):
            batch_state.completion_buffer = deque()

        if not batch_state.completion_buffer:
            completions = self.wait_for_completions(batch_state)
            batch_state.completion_buffer.extend(completions)

        return batch_state.completion_buffer.popleft()

    def _send_worker_result(
        self,
        result: NodeCompletion | _WorkerFailure,
        prefetched: deque[_EvaluateRequest] | None = None,
    ) -> bool:
        """
        Send a result, returning false when shutdown was received.

        MPI result sends must always finish. Cancelling a send that may already
        be matched by rank 0 is fragile across MPI transports, and Open MPI's
        OFI transport can crash in that path. During shutdown rank 0 continues
        receiving result messages until every worker acknowledges the request.
        """
        request = self.comm.isend(result, dest=0, tag=_RESULT_TAG)
        shutdown_received = False
        protocol_error: MPIWorkerError | None = None
        while not request.Test():
            if self.comm.Iprobe(source=0, tag=_CONTROL_TAG):
                command = self.comm.recv(source=0, tag=_CONTROL_TAG)
                if not isinstance(command, _Shutdown):
                    msg = f"Worker {self.rank} received unknown control: {command!r}"
                    protocol_error = MPIWorkerError(msg)
                else:
                    shutdown_received = True

            if (
                not shutdown_received
                and protocol_error is None
                and prefetched is not None
                and not prefetched
                and self.comm.Iprobe(source=0, tag=_COMMAND_TAG)
            ):
                # Receiving while the result is still in flight prevents a
                # blocking coordinator send from extending the inter-task gap.
                # Queue depth two means there can be only one local prefetch.
                command = self.comm.recv(source=0, tag=_COMMAND_TAG)
                if not isinstance(command, _EvaluateRequest):
                    msg = (
                        f"Worker {self.rank} received unexpected task command: "
                        f"{command!r}"
                    )
                    protocol_error = MPIWorkerError(msg)
                else:
                    prefetched.append(command)

        if protocol_error is not None:
            raise protocol_error
        return not shutdown_received

    def _acknowledge_shutdown(self) -> None:
        """Tell rank 0 it may finish closing after all result sends complete."""
        self.comm.send(_ShutdownAcknowledgement(), dest=0, tag=_CONTROL_TAG)

    def _report_worker_failure(self, exception: BaseException) -> bool:
        """
        Report a fatal worker failure and keep serving shutdown control.

        This is intentionally used for every ``BaseException`` that escapes
        worker protocol progress, not only exceptions raised by user tasks.
        In particular, a signal delivered while a worker waits for a command
        must not make it disappear before rank 0 can complete shutdown.
        """
        return self._send_worker_result(
            _describe_worker_failure(self.rank, exception),
        )

    def _next_worker_command(self) -> object:
        """Receive either an evaluation request or shutdown control."""
        status = MPI.Status()
        command = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() not in {_COMMAND_TAG, _CONTROL_TAG}:
            msg = f"Worker {self.rank} received message with unknown tag"
            raise MPIWorkerError(msg)
        return command

    def worker_loop(self) -> None:
        """Process tasks on a nonzero MPI rank until shutdown."""
        if self.rank == 0:
            msg = "worker_loop() cannot be called on MPI rank 0"
            raise RuntimeError(msg)

        failed = False
        while True:
            try:
                command = self._next_worker_command()
            except BaseException as exception:
                if not self._report_worker_failure(exception):
                    self._acknowledge_shutdown()
                    return
                failed = True
                continue

            if isinstance(command, _Shutdown):
                self._acknowledge_shutdown()
                return
            if failed:
                # A fatal worker error poisons the prepared schedule.
                # Keep receiving so the coordinator can still shut us down.
                continue

            try:
                if not isinstance(command, _EvaluateRequest):
                    msg = f"Worker {self.rank} received unknown command: {command!r}"
                    raise MPIWorkerError(msg)
                result = self._evaluate_worker_task(
                    command.task,
                    capture_context_state=True,
                )
                sent = self._send_worker_result(result)
            except BaseException as exception:
                if not self._report_worker_failure(exception):
                    self._acknowledge_shutdown()
                    return
                failed = True
                continue

            if not sent:
                self._acknowledge_shutdown()
                return

    def release_workers(self) -> None:
        """Request shutdown and wait until every worker has stopped safely."""
        if self.rank != 0 or self.size <= 1 or self._workers_released:
            return

        requests = [
            self.comm.isend(_Shutdown(), dest=rank, tag=_CONTROL_TAG)
            for rank in range(1, self.size)
        ]

        # A worker may see shutdown while its previous result send is still
        # active. Keep draining results until it confirms shutdown so that send
        # can complete without MPI_Cancel. The acknowledgment also means close
        # does not return while a worker can still evaluate a prefetched task.
        waiting_for_ack = set(range(1, self.size))
        while waiting_for_ack:
            status = MPI.Status()
            message = self.comm.recv(
                source=MPI.ANY_SOURCE,
                tag=MPI.ANY_TAG,
                status=status,
            )
            source = status.Get_source()
            tag = status.Get_tag()
            if tag == _RESULT_TAG:
                continue
            if tag != _CONTROL_TAG or not isinstance(message, _ShutdownAcknowledgement):
                msg = f"Unexpected MPI message while shutting down: {message!r}"
                raise MPIWorkerError(msg)
            if source not in waiting_for_ack:
                msg = f"Unexpected shutdown acknowledgement from worker {source}"
                raise MPIWorkerError(msg)
            waiting_for_ack.remove(source)

        if requests:
            MPI.Request.Waitall(requests)
        self._workers_released = True

    def abort_batch(self, batch_state: BatchState) -> None:
        """Discard queued work and close the poisoned MPI schedule."""
        if hasattr(batch_state, "pending_tasks"):
            batch_state.pending_tasks.clear()
        self.close()

    def close(self) -> None:
        """Release workers and permanently close the prepared schedule."""
        if self.closed:
            return
        try:
            self.release_workers()
        finally:
            super().close()


class MPIGraphScheduler(Scheduler[MPIGraphSchedule]):
    """Prepare MPI-backed graph schedules."""

    def __init__(
        self,
        comm: Any | None = None,
        *,
        mpi_debug_log: bool = False,
    ) -> None:
        """Configure the communicator and optional protocol logging."""
        super().__init__()
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.mpi_debug_log = mpi_debug_log

    def prepare(
        self,
        objective: Component[InputT_contra, OutputT_co],
        /,
    ) -> MPIGraphSchedule:
        """Compile and bind the component graph to the MPI communicator."""
        return MPIGraphSchedule(
            graph=objective_to_call_graph(objective),
            comm=self.comm,
            mpi_debug_log=self.mpi_debug_log,
        )
