"""
MPI execution for backend-neutral tree-schedule leaf tasks.

Rank 0 expands evaluation runs into :class:`~chemfit.tree_schedule.LeafTask`
values through ``TreeScheduleBase`` and distributes those tasks round-robin.
Worker ranks evaluate their assigned tasks and return backend-neutral
``LeafCompletion`` values. Tree propagation, reductions, exception handling,
context restoration, and objective lifecycles remain coordinator concerns.

Ordinary objective exceptions are carried in completions. A failure in MPI or
worker machinery is catastrophic: the coordinator closes the prepared
schedule, releases workers as cleanly as practical, and re-raises the failure.
The schedule is not reusable afterward.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from mpi4py import MPI

from chemfit.callgraph import CallTree, LeafNode, cob_to_call_tree
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.debug_utils import log_all_methods
from chemfit.scheduling import Scheduler
from chemfit.tree_schedule import (
    LeafCompletion,
    LeafTask,
    TreeScheduleBase,
    evaluate_leaf_task,
)

logger = logging.getLogger(__name__)

ParametersT = TypeVar("ParametersT", bound=Mapping[str, Any])

_COMMAND_TAG = 41001
_RESULT_TAG = 41002


@dataclass(frozen=True)
class _EvaluateRequest(Generic[ParametersT]):
    """Send one worker all leaf tasks assigned to it for a batch."""

    tasks: tuple[LeafTask[ParametersT], ...]


@dataclass(frozen=True)
class _Shutdown:
    """Request that a persistent worker loop terminate."""


class MPIWorkerError(RuntimeError):
    """Report catastrophic worker or MPI protocol failure to rank 0."""


@dataclass(frozen=True)
class _WorkerFailure:
    """Serializable description of a catastrophic worker failure."""

    rank: int
    exception_type: str
    message: str


def _describe_worker_failure(
    rank: int,
    exception: BaseException,
) -> _WorkerFailure:
    """Convert an arbitrary worker failure into a pickle-safe payload."""

    try:
        message = str(exception)
    except BaseException:
        message = "<exception message could not be formatted>"

    return _WorkerFailure(
        rank=rank,
        exception_type=type(exception).__qualname__,
        message=message,
    )


class MPITreeSchedule(TreeScheduleBase[ParametersT], Generic[ParametersT]):
    """
    Execute backend-neutral leaf tasks on persistent MPI worker ranks.

    Rank 0 coordinates and performs all tree semantics. Nonzero ranks only
    receive ``LeafTask`` values, evaluate their leaf objectives, and return
    ``LeafCompletion`` values.

    Args:
        tree: Compiled combined-objective call tree.
        comm: MPI communicator containing the coordinator and worker ranks.
        mpi_debug_log: Wrap the communicator with method-level debug logging.

    """

    def __init__(
        self,
        tree: CallTree,
        comm: Any,
        *,
        mpi_debug_log: bool = False,
    ) -> None:
        """Initialize the prepared MPI schedule."""

        super().__init__(tree)

        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self._workers_released = False

        if mpi_debug_log:
            self.comm = log_all_methods(
                self.comm,
                log_func=self._log_func,
                log_args=True,
                log_res=True,
            )

    def _log_func(self, msg: str) -> None:
        """Emit one communicator debug message with the local rank."""

        logger.warning("[Rank %s] %s", self.rank, msg)

    def _build_task_assignments(
        self,
        tasks: Sequence[LeafTask[ParametersT]],
    ) -> dict[int, list[LeafTask[ParametersT]]]:
        """Distribute a complete task batch round-robin across workers."""

        worker_ranks = tuple(range(1, self.size))
        assignments = {rank: [] for rank in worker_ranks}

        for task_idx, task in enumerate(tasks):
            rank = worker_ranks[task_idx % len(worker_ranks)]
            assignments[rank].append(task)

        return assignments

    def _evaluate_worker_task(
        self,
        task: LeafTask[ParametersT],
    ) -> LeafCompletion:
        """Evaluate one worker-local task and export its context state."""

        node = self.tree.nodes[task.node_id]
        assert isinstance(node, LeafNode)
        return evaluate_leaf_task(
            node.objective,
            task,
            capture_context_state=True,
        )

    def _send_worker_result(
        self,
        result: LeafCompletion | _WorkerFailure,
    ) -> bool:
        """Send a result while remaining responsive to coordinator shutdown."""

        request = self.comm.isend(result, dest=0, tag=_RESULT_TAG)
        while not request.Test():
            if not self.comm.Iprobe(source=0, tag=_COMMAND_TAG):
                continue

            command = self.comm.recv(source=0, tag=_COMMAND_TAG)
            if not isinstance(command, _Shutdown):
                request.Cancel()
                request.Free()
                msg = (
                    f"Worker rank {self.rank} received {command!r} while "
                    "sending a result"
                )
                raise MPIWorkerError(msg)

            # Do not wait for rank 0 to receive a large result after it has
            # requested shutdown. Completing the local cancellation keeps the
            # serialized send buffer alive for as long as MPI requires it,
            # without adding an acknowledgement or batch-recovery protocol.
            request.Cancel()
            request.Wait()
            return False

        return True

    def worker_loop(self) -> None:
        """Receive and execute task batches until rank 0 requests shutdown."""

        if self.rank == 0:
            msg = "worker_loop() cannot be used on rank 0"
            raise RuntimeError(msg)

        failed = False
        while True:
            command = self.comm.recv(source=0, tag=_COMMAND_TAG)

            if isinstance(command, _Shutdown):
                return

            if failed:
                # Catastrophic failure poisons the schedule. Rank 0 will close
                # it and send _Shutdown; no later work may be accepted.
                continue

            if not isinstance(command, _EvaluateRequest):
                msg = f"Worker rank {self.rank} received unknown command: {command!r}"
                failure = _WorkerFailure(
                    rank=self.rank,
                    exception_type=MPIWorkerError.__qualname__,
                    message=msg,
                )
                if not self._send_worker_result(failure):
                    return
                failed = True
                continue

            for task in command.tasks:
                try:
                    completion = self._evaluate_worker_task(task)
                except BaseException as exception:
                    failure = _describe_worker_failure(self.rank, exception)
                    if not self._send_worker_result(failure):
                        return
                    failed = True
                    break

                if not self._send_worker_result(completion):
                    return

    def execute_leaf_tasks(
        self,
        tasks: Sequence[LeafTask[ParametersT]],
    ) -> Iterator[LeafCompletion]:
        """Dispatch leaf tasks from rank 0 and yield remote completions."""

        if self.rank != 0:
            msg = "execute_leaf_tasks() can only be used on rank 0"
            raise RuntimeError(msg)

        if self.size == 1:
            for task in tasks:
                node = self.tree.nodes[task.node_id]
                assert isinstance(node, LeafNode)
                yield evaluate_leaf_task(
                    node.objective,
                    task,
                    capture_context_state=False,
                )
            return

        assignments = self._build_task_assignments(tasks)
        for rank, assigned_tasks in assignments.items():
            if assigned_tasks:
                self.comm.send(
                    _EvaluateRequest(tasks=tuple(assigned_tasks)),
                    dest=rank,
                    tag=_COMMAND_TAG,
                )

        for _ in range(len(tasks)):
            message = self.comm.recv(source=MPI.ANY_SOURCE, tag=_RESULT_TAG)

            if isinstance(message, _WorkerFailure):
                msg = (
                    f"Worker rank {message.rank} failed with "
                    f"{message.exception_type}: {message.message}"
                )
                raise MPIWorkerError(msg)
            if not isinstance(message, LeafCompletion):
                msg = f"Unknown MPI scheduler result: {message!r}"
                raise MPIWorkerError(msg)

            yield message

    def release_workers(self) -> None:
        """Best-effort request that every persistent worker exit."""

        if self.rank != 0 or self.size <= 1 or self._workers_released:
            return

        requests = [
            self.comm.isend(_Shutdown(), dest=rank, tag=_COMMAND_TAG)
            for rank in range(1, self.size)
        ]
        if requests:
            MPI.Request.Waitall(requests)

        self._workers_released = True

    def close(self) -> None:
        """Release workers and permanently close the prepared schedule."""

        if self.closed:
            return

        try:
            self.release_workers()
        finally:
            super().close()


class MPITreeScheduler(Scheduler[MPITreeSchedule[Any]]):
    """Prepare persistent MPI schedules for combined-objective trees."""

    def __init__(
        self,
        comm: Any | None = None,
        *,
        mpi_debug_log: bool = False,
    ) -> None:
        """Initialize reusable MPI scheduler configuration."""

        super().__init__()
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.mpi_debug_log = mpi_debug_log

    def prepare(
        self,
        objective: CombinedObjectiveFunction[ParametersT],
        /,
        *,
        profile: Mapping[tuple[int, ...], float] | None = None,  # noqa: ARG002
    ) -> MPITreeSchedule[ParametersT]:
        """Compile an objective tree and bind it to the MPI communicator."""

        return MPITreeSchedule(
            tree=cob_to_call_tree(objective),
            comm=self.comm,
            mpi_debug_log=self.mpi_debug_log,
        )
