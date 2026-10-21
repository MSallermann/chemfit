"""
MPI backend for ChemFit's graph scheduler.

Rank 0 owns graph traversal and structural-component lifecycles. Other ranks
run a persistent worker loop and evaluate only AtomicNode tasks. Tasks are
queued on rank 0 and dispatched to idle workers in adaptive batches, allowing
new work to become available as earlier graph nodes finish without requiring a
message round trip for every task in broad graphs.

Ordinary component exceptions are returned in NodeCompletion. MPI/protocol
failures are fatal to the prepared schedule and trigger worker shutdown.

All ranks must prepare the same component graph; only rank 0 evaluates it.
Nonzero ranks call ``schedule.worker_loop()`` and rank 0 calls ``schedule.close()``
after the final evaluation to release the workers.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
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
class _EvaluateBatch:
    tasks: tuple[NodeTask, ...]


@dataclass(frozen=True, slots=True)
class _BatchCompletion:
    completions: tuple[NodeCompletion, ...]


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


def _validate_batching_options(
    *,
    max_tasks_per_message: int,
    target_batches_per_worker: int,
) -> None:
    """Reject batching settings that cannot produce a nonempty batch."""
    if type(max_tasks_per_message) is not int or max_tasks_per_message < 1:
        msg = "max_tasks_per_message must be a positive integer"
        raise ValueError(msg)
    if type(target_batches_per_worker) is not int or target_batches_per_worker < 1:
        msg = "target_batches_per_worker must be a positive integer"
        raise ValueError(msg)


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
        max_tasks_per_message: int = 8,
        target_batches_per_worker: int = 8,
        mpi_debug_log: bool = False,
    ) -> None:
        """Bind a call graph to an MPI communicator."""
        super().__init__(graph)
        _validate_batching_options(
            max_tasks_per_message=max_tasks_per_message,
            target_batches_per_worker=target_batches_per_worker,
        )
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.max_tasks_per_message = max_tasks_per_message
        self.target_batches_per_worker = target_batches_per_worker
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
            # A ready-work cohort keeps one bulk dispatch quantum, then chooses
            # one smaller quantum for its final worker wave. Recomputing after
            # every refill would create progressively smaller batches and
            # increase the result-message rate during a completion burst.
            batch_state.dispatch_batch_size = None
            batch_state.dispatch_tail_started = False
            batch_state.in_flight = {}
            # Keep nonblocking command sends alive until the corresponding
            # worker result proves that the command was received.  A blocking
            # send here can deadlock with a worker's rendezvous result send
            # when both task and result exceed the MPI eager limit.
            batch_state.command_sends = {}
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
        """Assign one adaptive batch to each idle worker while work remains."""
        queue: deque[NodeTask] = batch_state.pending_tasks
        in_flight: dict[int, tuple[NodeTask, ...]] = batch_state.in_flight
        command_sends: dict[int, Any] = batch_state.command_sends
        idle_workers: deque[int] = batch_state.idle_workers

        if not queue:
            batch_state.dispatch_batch_size = None
            batch_state.dispatch_tail_started = False
            return

        chunk_size: int | None = getattr(
            batch_state,
            "dispatch_batch_size",
            None,
        )
        if chunk_size is None:
            chunk_size = self._chunk_size(len(queue))
            batch_state.dispatch_batch_size = chunk_size

        # Keep the cohort quantum stable while there is more than one full
        # worker wave queued. For the tail, spread the remaining tasks across
        # the workers that are idle or still in flight instead of assigning
        # full batches to only a subset and leaving the rest unused.
        worker_slots = len(idle_workers) + len(in_flight)
        tail_started = getattr(batch_state, "dispatch_tail_started", False)
        if (
            not tail_started
            and worker_slots
            and len(queue) <= chunk_size * worker_slots
        ):
            chunk_size = (len(queue) + worker_slots - 1) // worker_slots
            batch_state.dispatch_batch_size = chunk_size
            batch_state.dispatch_tail_started = True

        while queue and idle_workers:
            rank = idle_workers.popleft()
            tasks = tuple(queue.popleft() for _ in range(min(chunk_size, len(queue))))
            command_sends[rank] = self.comm.isend(
                _EvaluateBatch(tasks), dest=rank, tag=_COMMAND_TAG
            )
            in_flight[rank] = tasks

        if not queue:
            batch_state.dispatch_batch_size = None
            batch_state.dispatch_tail_started = False

    def _chunk_size(self, queued_tasks: int) -> int:
        """Choose a message size that preserves several batches per worker."""
        worker_count = max(1, self.size - 1)
        target_batch_count = worker_count * self.target_batches_per_worker
        adaptive_size = (queued_tasks + target_batch_count - 1) // target_batch_count
        return min(self.max_tasks_per_message, max(1, adaptive_size))

    def submit_tasks(
        self,
        tasks: Sequence[NodeTask],
        batch_state: BatchState,
    ) -> None:
        """Queue tasks and immediately give available workers more work."""
        self._ensure_coordinator()
        self._ensure_batch_state(batch_state)
        queue_was_empty = not batch_state.pending_tasks
        batch_state.pending_tasks.extend(tasks)

        if queue_was_empty and batch_state.pending_tasks:
            batch_state.dispatch_batch_size = self._chunk_size(
                len(batch_state.pending_tasks)
            )
            batch_state.dispatch_tail_started = False

        if self.size > 1:
            self._dispatch_available(batch_state)

    def _receive_worker_completion(
        self, batch_state: BatchState
    ) -> tuple[NodeCompletion, ...]:
        """Receive and validate one worker batch, then reuse or idle its rank."""
        in_flight: dict[int, tuple[NodeTask, ...]] = batch_state.in_flight
        command_sends: dict[int, Any] = batch_state.command_sends
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

        expected_tasks = in_flight.get(source)
        if expected_tasks is None:
            msg = f"Unexpected completion from idle worker {source}"
            raise MPIWorkerError(msg)
        if not isinstance(message, _BatchCompletion):
            msg = f"Unexpected MPI worker result: {message!r}"
            raise MPIWorkerError(msg)
        if not all(
            isinstance(completion, NodeCompletion) for completion in message.completions
        ):
            msg = f"Worker {source} returned a malformed completion batch"
            raise MPIWorkerError(msg)

        expected = Counter((task.run_id, task.node_id) for task in expected_tasks)
        returned = Counter(
            (completion.run_id, completion.node_id)
            for completion in message.completions
        )
        duplicates = [identity for identity, count in returned.items() if count > 1]
        if duplicates:
            msg = (
                f"Worker {source} returned duplicate completion for task "
                f"{duplicates[0]}"
            )
            raise MPIWorkerError(msg)
        unexpected = list(returned.keys() - expected.keys())
        if unexpected:
            msg = (
                f"Worker {source} returned completion for unexpected task "
                f"{unexpected[0]}"
            )
            raise MPIWorkerError(msg)
        missing = list(expected.keys() - returned.keys())
        if missing:
            msg = f"Worker {source} omitted completion for task {missing[0]}"
            raise MPIWorkerError(msg)
        if returned != expected:
            msg = f"Worker {source} returned an invalid completion batch"
            raise MPIWorkerError(msg)

        # The matching result proves that the worker fully received this
        # command, so completing the retained send cannot wait on remote
        # progress.  Do this before replacing the per-rank request.
        command_send = command_sends.get(source)
        if command_send is None:
            msg = f"Missing command send for worker {source}"
            raise MPIWorkerError(msg)
        command_send.Wait()
        del command_sends[source]
        del in_flight[source]

        queue: deque[NodeTask] = batch_state.pending_tasks
        if queue:
            # Refill the completed worker before receiving another result. At
            # large rank counts, deferring all refills until the completion
            # burst has been drained leaves every worker in that burst idle
            # for the coordinator's aggregate receive/validation time.
            batch_state.idle_workers.appendleft(source)
            self._dispatch_available(batch_state)
        else:
            batch_state.idle_workers.append(source)

        return message.completions

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
            if not batch_state.pending_tasks:
                batch_state.dispatch_batch_size = None
                batch_state.dispatch_tail_started = False
            return (self._evaluate_worker_task(task, capture_context_state=False),)

        in_flight: dict[int, tuple[NodeTask, ...]] = batch_state.in_flight
        if not in_flight:
            msg = "No MPI tasks are in flight"
            raise MPIWorkerError(msg)

        # Bound draining by the number in flight at entry. Refilled work can
        # finish quickly, but consuming those new results in this call could
        # otherwise keep the coordinator in an unbounded receive loop and
        # delay graph dependency release.
        drain_limit = len(in_flight)
        completions = list(self._receive_worker_completion(batch_state))
        received_batches = 1
        while received_batches < drain_limit and self.comm.Iprobe(
            source=MPI.ANY_SOURCE,
            tag=_RESULT_TAG,
        ):
            completions.extend(self._receive_worker_completion(batch_state))
            received_batches += 1

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
        result: _BatchCompletion | _WorkerFailure,
        prefetched: deque[_EvaluateBatch] | None = None,
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
                if not isinstance(command, _EvaluateBatch):
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

        prefetched: deque[_EvaluateBatch] = deque()
        failed = False
        while True:
            try:
                command = (
                    prefetched.popleft() if prefetched else self._next_worker_command()
                )
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
                if not isinstance(command, _EvaluateBatch):
                    msg = f"Worker {self.rank} received unknown command: {command!r}"
                    raise MPIWorkerError(msg)
                result = _BatchCompletion(
                    tuple(
                        self._evaluate_worker_task(
                            task,
                            capture_context_state=True,
                        )
                        for task in command.tasks
                    )
                )
                sent = self._send_worker_result(result, prefetched)
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
        # Every dispatched worker receives at most one command at a time.  Its
        # command receive was posted before it could evaluate or report a
        # failure, so these sends can be completed safely before shutdown.
        command_sends = getattr(batch_state, "command_sends", {})
        if command_sends:
            MPI.Request.Waitall(list(command_sends.values()))
            command_sends.clear()
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
        max_tasks_per_message: int = 8,
        target_batches_per_worker: int = 8,
        mpi_debug_log: bool = False,
    ) -> None:
        """Configure the communicator, batching, and optional protocol logging."""
        super().__init__()
        _validate_batching_options(
            max_tasks_per_message=max_tasks_per_message,
            target_batches_per_worker=target_batches_per_worker,
        )
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.max_tasks_per_message = max_tasks_per_message
        self.target_batches_per_worker = target_batches_per_worker
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
            max_tasks_per_message=self.max_tasks_per_message,
            target_batches_per_worker=self.target_batches_per_worker,
            mpi_debug_log=self.mpi_debug_log,
        )
