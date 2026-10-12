"""MPI backend for ChemFit's graph scheduler.

Rank 0 owns graph traversal and structural-component lifecycles. Other ranks
run a persistent worker loop and evaluate only AtomicNode tasks. Tasks are
queued on rank 0 and dispatched to idle workers one at a time, allowing new
work to become available as earlier graph nodes finish.

Ordinary objective exceptions are returned in NodeCompletion. MPI/protocol
failures are fatal to the prepared schedule and trigger worker shutdown.

All ranks must prepare the same objective graph; only rank 0 evaluates it.
Nonzero ranks call ``schedule.worker_loop()`` and rank 0 calls ``schedule.close()``
after the final evaluation to release the workers.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from mpi4py import MPI

from chemfit.abstract_objective_function import (
    InputT_contra,
    ObjectiveComponent,
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
from chemfit.scheduling import EvaluationRequest, EvaluationResult, Scheduler

logger = logging.getLogger(__name__)

_COMMAND_TAG = 41001
_RESULT_TAG = 41002


@dataclass(frozen=True)
class _EvaluateRequest:
    task: NodeTask


@dataclass(frozen=True)
class _Shutdown:
    pass


@dataclass(frozen=True)
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
            raise RuntimeError("Graph evaluation is only supported on MPI rank 0")
        if self.closed:
            raise RuntimeError("Prepared MPI graph schedule is closed")

    @staticmethod
    def _ensure_batch_state(batch_state: BatchState) -> None:
        if not hasattr(batch_state, "pending_tasks"):
            batch_state.pending_tasks = deque()
            batch_state.in_flight = {}

    def _evaluate_worker_task(
        self, task: NodeTask, *, capture_context_state: bool
    ) -> NodeCompletion:
        node = self.graph.nodes[task.node_id]
        if not isinstance(node, AtomicNode):
            raise MPIWorkerError(f"MPI worker received non-atomic node {task.node_id}")

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

        for rank in range(1, self.size):
            if not queue:
                break
            if rank in in_flight:
                continue

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

    def next_completion(self, batch_state: BatchState) -> NodeCompletion:
        """Return one task completion, dispatching queued work as ranks free up."""
        self._ensure_coordinator()
        self._ensure_batch_state(batch_state)

        if self.size == 1:
            if not batch_state.pending_tasks:
                raise MPIWorkerError("No queued MPI tasks to complete")
            task = batch_state.pending_tasks.popleft()
            return self._evaluate_worker_task(task, capture_context_state=False)

        in_flight: dict[int, NodeTask] = batch_state.in_flight
        if not in_flight:
            raise MPIWorkerError("No MPI tasks are in flight")

        status = MPI.Status()
        message = self.comm.recv(
            source=MPI.ANY_SOURCE,
            tag=_RESULT_TAG,
            status=status,
        )
        source = status.Get_source()

        if isinstance(message, _WorkerFailure):
            raise MPIWorkerError(
                f"Worker rank {message.rank} failed with "
                f"{message.exception_type}: {message.message}"
            )

        expected_task = in_flight.pop(source, None)
        if expected_task is None:
            raise MPIWorkerError(f"Unexpected completion from idle worker {source}")
        if not isinstance(message, NodeCompletion):
            raise MPIWorkerError(f"Unexpected MPI worker result: {message!r}")
        if (message.run_id, message.node_id) != (
            expected_task.run_id,
            expected_task.node_id,
        ):
            raise MPIWorkerError(
                f"Worker {source} returned completion for unexpected task "
                f"({message.run_id}, {message.node_id})"
            )

        self._dispatch_available(batch_state)
        return message

    def _send_worker_result(self, result: NodeCompletion | _WorkerFailure) -> bool:
        """Send a result while allowing rank 0 to interrupt with shutdown."""
        request = self.comm.isend(result, dest=0, tag=_RESULT_TAG)
        while not request.Test():
            if not self.comm.Iprobe(source=0, tag=_COMMAND_TAG):
                continue

            command = self.comm.recv(source=0, tag=_COMMAND_TAG)
            if not isinstance(command, _Shutdown):
                request.Cancel()
                request.Wait()
                raise MPIWorkerError(
                    f"Worker {self.rank} received unexpected command while "
                    f"sending a result: {command!r}"
                )

            request.Cancel()
            request.Wait()
            return False

        return True

    def worker_loop(self) -> None:
        """Process tasks on a nonzero MPI rank until shutdown."""
        if self.rank == 0:
            raise RuntimeError("worker_loop() cannot be called on MPI rank 0")

        failed = False
        while True:
            command = self.comm.recv(source=0, tag=_COMMAND_TAG)
            if isinstance(command, _Shutdown):
                return
            if failed:
                # A fatal worker error poisons the prepared schedule.
                # Keep receiving so the coordinator can still shut us down.
                continue

            if not isinstance(command, _EvaluateRequest):
                exception = MPIWorkerError(
                    f"Worker {self.rank} received unknown command: {command!r}"
                )
                if not self._send_worker_result(
                    _describe_worker_failure(self.rank, exception)
                ):
                    return
                failed = True
                continue

            try:
                result = self._evaluate_worker_task(
                    command.task,
                    capture_context_state=True,
                )
                sent = self._send_worker_result(result)
            except BaseException as exception:
                failure = _describe_worker_failure(self.rank, exception)
                if not self._send_worker_result(failure):
                    return
                failed = True
                continue

            if not sent:
                return

    def release_workers(self) -> None:
        """Request shutdown from every persistent worker rank."""
        if self.rank != 0 or self.size <= 1 or self._workers_released:
            return

        requests = [
            self.comm.isend(_Shutdown(), dest=rank, tag=_COMMAND_TAG)
            for rank in range(1, self.size)
        ]
        if requests:
            MPI.Request.Waitall(requests)
        self._workers_released = True

    def abort_batch(self, batch_state: BatchState) -> None:
        """Discard queued work and close the poisoned MPI schedule."""
        if hasattr(batch_state, "pending_tasks"):
            batch_state.pending_tasks.clear()
        self.close()

    def evaluate_many(
        self,
        requests: Sequence[EvaluationRequest],
    ) -> Iterator[EvaluationResult]:
        """Evaluate on the coordinator; close workers on catastrophic failure."""
        self._ensure_coordinator()
        try:
            yield from super().evaluate_many(requests)
        except BaseException:
            try:
                self.close()
            except Exception:
                logger.exception("Failed to close MPI schedule after an error")
            raise

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
        super().__init__()
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.mpi_debug_log = mpi_debug_log

    def prepare(
        self,
        objective: ObjectiveComponent[InputT_contra, OutputT_co],
        /,
    ) -> MPIGraphSchedule:
        """Compile and bind the objective graph to the MPI communicator."""
        return MPIGraphSchedule(
            graph=objective_to_call_graph(objective),
            comm=self.comm,
            mpi_debug_log=self.mpi_debug_log,
        )
