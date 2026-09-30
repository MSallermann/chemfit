from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any, Generic, TypeVar, cast

from mpi4py import MPI

from chemfit.abstract_objective_function import EvaluateContext
from chemfit.callgraph import CallTree, CombineNode, LeafNode, NodeId, cob_to_call_tree
from chemfit.combined_objective_function import (
    CombinedObjectiveFunction,
    evaluate_weighted_term,
)
from chemfit.debug_utils import log_all_methods
from chemfit.executor_utils import AttachContextAsReturnValue
from chemfit.scheduling import Scheduler
from chemfit.tree_schedule import EvaluationState, TermResult, TreeScheduleBase

logger = logging.getLogger(__name__)

ParametersT = TypeVar("ParametersT", bound=Mapping[str, Any])

evaluate_weighted_term_with_ctx = AttachContextAsReturnValue(evaluate_weighted_term)

_COMMAND_TAG = 41001
_RESULT_TAG = 41002


def slice_up_range(n: int, n_ranks: int):
    """Split ``range(n)`` into contiguous rank-local chunks."""
    if n_ranks <= 0:
        msg = "n_ranks must be positive"
        raise ValueError(msg)

    chunk_size = math.ceil(n / n_ranks) if n else 0

    for rank in range(n_ranks):
        start = rank * chunk_size
        end = min(start + chunk_size, n)
        yield start, end


@dataclass(frozen=True)
class _LeafTask:
    node_id: NodeId
    ctx: EvaluateContext


@dataclass(frozen=True)
class _EvaluateRequest:
    eval_id: int
    parameters: Mapping[str, Any]
    tasks: tuple[_LeafTask, ...]


@dataclass(frozen=True)
class _CancelEvaluation:
    eval_id: int


@dataclass(frozen=True)
class _Shutdown:
    pass


@dataclass(frozen=True)
class _LeafCompleted:
    eval_id: int
    node_id: NodeId
    result: TermResult
    ctx_result_state: dict[str, Any]


@dataclass(frozen=True)
class _LeafFailed:
    eval_id: int
    node_id: NodeId
    exception: BaseException
    ctx_result_state: dict[str, Any]


@dataclass(frozen=True)
class _EvaluationDone:
    eval_id: int


@dataclass
class _MPIEvaluationRuntime:
    """MPI-specific state for one top-level evaluation."""

    eval_id: int
    started_workers: set[int]
    done_workers: set[int]


class MPIWorkerError(RuntimeError):
    """Report a worker failure that could not be serialized directly."""


class MPITreeSchedule(
    TreeScheduleBase[ParametersT],
    Generic[ParametersT],
):
    """
    Tree schedule whose leaves are evaluated by persistent MPI workers.

    Rank 0 is the coordinator. It owns the context tree, EvaluationState,
    nested COB lifecycles, nested reductions, and completion propagation.

    Nonzero ranks execute only leaves.

    When more than one rank is available, rank 0 deliberately does not execute
    leaves itself. This keeps the coordinator responsive to remote completion
    messages, allowing nested COBs to finish promptly when their own children
    are complete.
    """

    def __init__(
        self,
        tree: CallTree,
        comm: Any,
        *,
        mpi_debug_log: bool = False,
    ) -> None:
        """Initialize a prepared MPI schedule."""
        super().__init__(tree)

        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self._next_eval_id = 0
        self._workers_released = False
        self._evaluation_lock = Lock()

        if mpi_debug_log:
            self.comm = log_all_methods(
                self.comm,
                log_func=self._log_func,
                log_args=True,
                log_res=True,
            )

        self._leaf_assignments = self._build_leaf_assignments()

    def _log_func(self, msg: str) -> None:
        logger.warning("[Rank %s] %s", self.rank, msg)

    def _build_leaf_assignments(self) -> dict[int, tuple[NodeId, ...]]:
        """
        Statically assign leaves to nonzero worker ranks.

        The current baseline uses contiguous chunks. Profile-aware placement can
        replace this policy later without changing the tree runtime.
        """
        if self.size <= 1:
            return {}

        worker_ranks = tuple(range(1, self.size))
        slices = tuple(slice_up_range(len(self.leaf_ids), len(worker_ranks)))

        return {
            rank: tuple(self.leaf_ids[start:end])
            for rank, (start, end) in zip(worker_ranks, slices, strict=True)
        }

    def _new_eval_id(self) -> int:
        eval_id = self._next_eval_id
        self._next_eval_id += 1
        return eval_id

    def _set_mpi_runtime(
        self,
        eval_state: EvaluationState,
        runtime: _MPIEvaluationRuntime,
    ) -> None:
        root_ctx = eval_state.contexts[self.tree.root]
        assert root_ctx is not None
        root_ctx.temp.mpi_runtime = runtime

    def _get_mpi_runtime(
        self,
        eval_state: EvaluationState,
    ) -> _MPIEvaluationRuntime:
        root_ctx = eval_state.contexts[self.tree.root]
        assert root_ctx is not None
        return cast("_MPIEvaluationRuntime", root_ctx.temp.mpi_runtime)

    def _apply_result_state(
        self,
        node_id: NodeId,
        ctx_result_state: dict[str, Any],
        eval_state: EvaluationState,
    ) -> None:
        ctx = eval_state.contexts[node_id]
        assert ctx is not None
        ctx.apply_result_state(ctx_result_state)

    def _make_request(
        self,
        *,
        rank: int,
        eval_id: int,
        parameters: ParametersT,
        eval_state: EvaluationState,
    ) -> _EvaluateRequest:
        tasks: list[_LeafTask] = []

        for node_id in self._leaf_assignments[rank]:
            ctx = eval_state.contexts[node_id]
            assert ctx is not None
            tasks.append(_LeafTask(node_id=node_id, ctx=ctx))

        return _EvaluateRequest(
            eval_id=eval_id,
            parameters=parameters,
            tasks=tuple(tasks),
        )

    def _evaluate_worker_leaf(
        self,
        task: _LeafTask,
        parameters: ParametersT,
    ) -> _LeafCompleted | _LeafFailed:
        """Evaluate one leaf on a worker and return result plus context state."""
        node = self.tree.nodes[task.node_id]
        assert isinstance(node, LeafNode)
        assert node.parent_id is not None
        assert node.child_idx is not None

        parent = self.tree.nodes[node.parent_id]
        assert isinstance(parent, CombineNode)

        try:
            result, ctx_result_state = evaluate_weighted_term_with_ctx(
                node.objective,
                parent.objective.weights[node.child_idx],
                parent.objective.exception_handler,
                parameters,
                node.child_idx,
                task.ctx,
            )
        except BaseException as exc:
            return _LeafFailed(
                eval_id=-1,
                node_id=task.node_id,
                exception=exc,
                ctx_result_state=task.ctx.to_result_state(),
            )

        return _LeafCompleted(
            eval_id=-1,
            node_id=task.node_id,
            result=result,
            ctx_result_state=ctx_result_state,
        )

    def _poll_control(self, eval_id: int) -> tuple[bool, bool]:
        """
        Poll for control messages between leaf evaluations.

        Returns ``(cancel_current_evaluation, shutdown_worker)``.
        """
        cancel = False
        shutdown = False

        while self.comm.iprobe(source=0, tag=_COMMAND_TAG):
            msg = self.comm.recv(source=0, tag=_COMMAND_TAG)

            if isinstance(msg, _CancelEvaluation):
                if msg.eval_id == eval_id:
                    cancel = True
            elif isinstance(msg, _Shutdown):
                cancel = True
                shutdown = True
            elif isinstance(msg, _EvaluateRequest):
                msg = "MPI worker received a new evaluation request while another evaluation was still active."
                raise RuntimeError(msg)
            else:
                msg = f"Unknown MPI scheduler command: {msg!r}"
                raise RuntimeError(msg)

        return cancel, shutdown

    def _process_request(self, request: _EvaluateRequest) -> bool:
        """
        Process one evaluation request on a worker.

        Returns True if the worker should exit after the request.
        """
        shutdown = False

        try:
            parameters = cast("ParametersT", request.parameters)

            for task in request.tasks:
                cancel, shutdown_now = self._poll_control(request.eval_id)
                shutdown = shutdown or shutdown_now

                if cancel:
                    break

                outcome = self._evaluate_worker_leaf(task, parameters)

                if isinstance(outcome, _LeafCompleted):
                    msg: _LeafCompleted | _LeafFailed = _LeafCompleted(
                        eval_id=request.eval_id,
                        node_id=outcome.node_id,
                        result=outcome.result,
                        ctx_result_state=outcome.ctx_result_state,
                    )
                else:
                    msg = _LeafFailed(
                        eval_id=request.eval_id,
                        node_id=outcome.node_id,
                        exception=outcome.exception,
                        ctx_result_state=outcome.ctx_result_state,
                    )

                self.comm.send(msg, dest=0, tag=_RESULT_TAG)

                if isinstance(msg, _LeafFailed):
                    break

        finally:
            self.comm.send(
                _EvaluationDone(request.eval_id),
                dest=0,
                tag=_RESULT_TAG,
            )

        return shutdown

    def worker_loop(self) -> None:
        """Run the persistent worker loop on a nonzero rank."""
        if self.rank == 0:
            msg = "worker_loop() cannot be used on rank 0"
            raise RuntimeError(msg)

        while True:
            msg = self.comm.recv(source=0, tag=_COMMAND_TAG)

            if isinstance(msg, _Shutdown):
                return

            if isinstance(msg, _CancelEvaluation):
                # Stale cancellation received while idle.
                continue

            if not isinstance(msg, _EvaluateRequest):
                err_msg = f"Unknown MPI scheduler command: {msg!r}"
                raise RuntimeError(err_msg)

            if self._process_request(msg):
                return

    def evaluate_leaves(
        self,
        parameters: ParametersT,
        eval_state: EvaluationState,
    ):
        """Dispatch leaf work and yield parent-facing completion events."""
        if self.rank != 0:
            msg = "evaluate_leaves() can only be used on rank 0"
            raise RuntimeError(msg)

        if self.size == 1:
            for node_id in self.leaf_ids:
                yield node_id, self.evaluate_leaf(node_id, parameters, eval_state)
            return

        eval_id = self._new_eval_id()
        runtime = _MPIEvaluationRuntime(
            eval_id=eval_id,
            started_workers=set(),
            done_workers=set(),
        )
        self._set_mpi_runtime(eval_state, runtime)

        for rank in range(1, self.size):
            request = self._make_request(
                rank=rank,
                eval_id=eval_id,
                parameters=parameters,
                eval_state=eval_state,
            )
            self.comm.send(request, dest=rank, tag=_COMMAND_TAG)
            runtime.started_workers.add(rank)

        status = MPI.Status()

        while runtime.done_workers != runtime.started_workers:
            msg = self.comm.recv(
                source=MPI.ANY_SOURCE,
                tag=_RESULT_TAG,
                status=status,
            )
            source = status.Get_source()

            if getattr(msg, "eval_id", None) != eval_id:
                msg = (
                    f"Received MPI result for evaluation "
                    f"{getattr(msg, 'eval_id', None)}, expected {eval_id}."
                )
                raise RuntimeError(msg)

            if isinstance(msg, _EvaluationDone):
                runtime.done_workers.add(source)
                continue

            if isinstance(msg, _LeafCompleted):
                self._apply_result_state(
                    msg.node_id,
                    msg.ctx_result_state,
                    eval_state,
                )
                yield msg.node_id, msg.result
                continue

            if isinstance(msg, _LeafFailed):
                self._apply_result_state(
                    msg.node_id,
                    msg.ctx_result_state,
                    eval_state,
                )
                raise msg.exception

            msg = f"Unknown MPI scheduler result: {msg!r}"
            raise RuntimeError(msg)

    def cancel_pending_and_wait(
        self,
        eval_state: EvaluationState,
    ) -> None:
        """
        Quiesce all MPI leaf work belonging to this evaluation.

        Running leaves are allowed to finish. Workers observe cancellation
        between leaves and do not start further pending leaves. Late leaf
        context states are restored on rank 0 but their numerical completions
        are not propagated after catastrophic abort has begun.
        """
        if self.rank != 0:
            msg = "cancel_pending_and_wait() can only be used on rank 0"
            raise RuntimeError(msg)

        if self.size == 1:
            return

        runtime = self._get_mpi_runtime(eval_state)
        outstanding = runtime.started_workers - runtime.done_workers

        cancel_requests = [
            self.comm.isend(
                _CancelEvaluation(runtime.eval_id),
                dest=rank,
                tag=_COMMAND_TAG,
            )
            for rank in outstanding
        ]

        status = MPI.Status()

        while runtime.done_workers != runtime.started_workers:
            msg = self.comm.recv(
                source=MPI.ANY_SOURCE,
                tag=_RESULT_TAG,
                status=status,
            )
            source = status.Get_source()

            if getattr(msg, "eval_id", None) != runtime.eval_id:
                msg = (
                    "Received MPI result for evaluation "
                    f"{getattr(msg, 'eval_id', None)}, "
                    f"expected {runtime.eval_id} while aborting."
                )
                raise RuntimeError(msg)

            if isinstance(msg, _EvaluationDone):
                runtime.done_workers.add(source)
                continue

            if isinstance(msg, (_LeafCompleted, _LeafFailed)):
                self._apply_result_state(
                    msg.node_id,
                    msg.ctx_result_state,
                    eval_state,
                )
                continue

            msg = f"Unknown MPI scheduler result: {msg!r}"
            raise RuntimeError(msg)

        if cancel_requests:
            MPI.Request.Waitall(cancel_requests)

    def release_workers(self) -> None:
        """Tell persistent worker loops to exit."""
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
        self.release_workers()
        super().close()


class MPITreeScheduler(
    Scheduler[ParametersT],
    Generic[ParametersT],
):
    """
    Scheduler creating an MPI-backed tree schedule.

    All ranks must prepare the same objective tree. Then rank 0 evaluates the
    objective normally and nonzero ranks enter ``prepared.worker_loop()``.

    This baseline intentionally supports one active evaluation at a time per
    communicator.
    """

    def __init__(
        self,
        comm: Any | None = None,
        *,
        mpi_debug_log: bool = False,
    ) -> None:
        """Initialize the mpi tree scheduler."""
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
        return MPITreeSchedule(
            tree=cob_to_call_tree(objective),
            comm=self.comm,
            mpi_debug_log=self.mpi_debug_log,
        )
