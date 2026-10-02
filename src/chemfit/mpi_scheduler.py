"""
MPI-backed tree scheduling with persistent worker ranks.

Rank 0 coordinates each evaluation batch, owns the authoritative context and
tree state, and performs nested reductions. Nonzero ranks keep persistent
worker loops that receive statically assigned leaf tasks, evaluate objectives,
and return raw outcomes with transferable context state.

Ordinary objective Exceptions are returned to the coordinator for parent-level
handling. MPI protocol or worker-runtime failures instead abort the batch and
trigger coordinated cancellation before open evaluation lifecycles are closed.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from queue import Empty, Queue
from threading import Lock
from typing import Any, Generic, TypeVar, cast

from mpi4py import MPI

from chemfit.abstract_objective_function import EvaluateContext
from chemfit.callgraph import CallTree, NodeId, cob_to_call_tree
from chemfit.combined_objective_function import (
    CombinedObjectiveFunction,
    evaluate_weighted_term,
)
from chemfit.debug_utils import log_all_methods
from chemfit.executor_utils import AttachContextAsReturnValue
from chemfit.scheduling import Scheduler
from chemfit.tree_schedule import (
    EvaluationRun,
    EvaluationState,
    NodeOutcome,
    SerialTreeSchedule,
    TreeScheduleBase,
)

logger = logging.getLogger(__name__)

ParametersT = TypeVar("ParametersT", bound=Mapping[str, Any])

evaluate_weighted_term_with_ctx = AttachContextAsReturnValue(evaluate_weighted_term)

_COMMAND_TAG = 41001
_RESULT_TAG = 41002


def slice_up_range(n: int, n_ranks: int):
    """
    Split a range into contiguous rank-local chunks.

    Args:
        n: Number of positions to distribute.
        n_ranks: Number of chunks to produce.

    Yields:
        Start and exclusive end indices for each rank, including empty chunks
        when there are more ranks than positions.

    Raises:
        ValueError: If n_ranks is not positive.

    """

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
    """
    Describe one leaf evaluation sent to a worker rank.

    Args:
        node_id: Identifier of the leaf in the compiled call tree.
        ctx: Worker-side context for this leaf evaluation.

    """

    node_id: NodeId
    ctx: EvaluateContext


@dataclass(frozen=True)
class _EvaluateRequest:
    """
    Group the leaf tasks assigned to one worker for one batch run.

    Args:
        eval_id: Worker-local identifier for the request.
        run_id: Position of the evaluation run in the coordinator's batch.
        parameters: Parameter mapping shared by the request's leaf tasks.
        tasks: Ordered leaf tasks assigned to the receiving rank.

    """

    eval_id: int
    run_id: int
    parameters: Mapping[str, Any]
    tasks: tuple[_LeafTask, ...]


@dataclass(frozen=True)
class _CancelPending:
    """Request that a worker cancel queued work and become quiescent."""


@dataclass(frozen=True)
class _Shutdown:
    """Request that a persistent worker loop terminate."""


@dataclass(frozen=True)
class _LeafCompleted:
    """
    Represent a completed leaf response.

    Args:
        eval_id: Worker-local identifier of the originating request.
        run_id: Position of the evaluation run in the coordinator's batch.
        node_id: Identifier of the completed leaf node.
        result: Raw numerical value or Exception produced by the leaf.
        ctx_result_state: Transferable result state of the worker context.

    """

    eval_id: int
    run_id: int
    node_id: NodeId
    result: NodeOutcome
    ctx_result_state: dict[str, Any]


@dataclass(frozen=True)
class _EvaluationDone:
    """Acknowledge that a worker has become quiescent after cancellation."""


@dataclass
class _MPIEvaluationRuntime:
    """
    Track MPI request progress for one top-level evaluation.

    Args:
        eval_id: Identifier to assign to the next worker request.
        started_workers: Ranks that received work for the evaluation.
        done_workers: Ranks that acknowledged completion or cancellation.

    """

    eval_id: int
    started_workers: set[int]
    done_workers: set[int]


class MPIWorkerError(RuntimeError):
    """Report a failure in worker machinery rather than objective evaluation."""


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

    Args:
        tree: Compiled combined-objective call tree.
        comm: MPI communicator containing the coordinator and worker ranks.
        mpi_debug_log: Wrap the communicator with method-level debug logging.

    Attributes:
        comm: MPI communicator, optionally wrapped for debug logging.
        rank: Rank of the current process in comm.
        size: Number of ranks in comm.
        work_requests: Worker-local queue of evaluation requests.

    """

    def __init__(
        self,
        tree: CallTree,
        comm: Any,
        *,
        mpi_debug_log: bool = False,
    ) -> None:
        """
        Initialize the prepared schedule and static leaf assignments.

        Args:
            tree: Compiled combined-objective call tree.
            comm: MPI communicator containing all participating ranks.
            mpi_debug_log: Log communicator method calls and their results.

        """

        super().__init__(tree)

        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.work_requests: Queue[_EvaluateRequest] = Queue()
        self.event_cancel_work = threading.Event()
        self.event_shutdown_worker = threading.Event()

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
        """
        Emit one communicator debug message with the local rank.

        Args:
            msg: Message produced by the communicator logging wrapper.

        """

        logger.warning("[Rank %s] %s", self.rank, msg)

    def _build_leaf_assignments(self) -> dict[int, tuple[NodeId, ...]]:
        """
        Statically assign leaves to nonzero worker ranks.

        The current baseline uses contiguous chunks. Profile-aware placement can
        replace this policy later without changing the tree runtime.

        Returns:
            Mapping from each nonzero rank to its assigned leaf identifiers.
            An empty mapping is returned for a single-rank communicator.

        """

        if self.size <= 1:
            return {}

        worker_ranks = tuple(range(1, self.size))
        slices = tuple(slice_up_range(len(self.leaf_ids), len(worker_ranks)))

        return {
            rank: tuple(self.leaf_ids[start:end])
            for rank, (start, end) in zip(worker_ranks, slices, strict=True)
        }

    def set_mpi_runtime(
        self,
        eval_state: EvaluationState,
        runtime: _MPIEvaluationRuntime,
    ) -> None:
        """
        Attach MPI runtime bookkeeping to an evaluation's root context.

        Args:
            eval_state: Evaluation state whose runtime should be recorded.
            runtime: MPI-specific request progress for the evaluation.

        """

        root_ctx = eval_state.contexts[self.tree.root]
        assert root_ctx is not None
        root_ctx.temp.mpi_runtime = runtime

    def get_mpi_runtime(
        self,
        eval_state: EvaluationState,
    ) -> _MPIEvaluationRuntime:
        """
        Return MPI runtime bookkeeping from an evaluation's root context.

        Args:
            eval_state: Evaluation state whose runtime should be retrieved.

        Returns:
            MPI-specific request progress for the evaluation.

        """

        root_ctx = eval_state.contexts[self.tree.root]
        assert root_ctx is not None
        return cast("_MPIEvaluationRuntime", root_ctx.temp.mpi_runtime)

    def evaluate_leaf_worker(
        self,
        node_id: NodeId,
        parameters: ParametersT,
        ctx: EvaluateContext,
    ) -> tuple[NodeOutcome, dict[str, Any]]:
        """
        Evaluate one leaf and export its worker-side context state.

        Args:
            node_id: Identifier of the leaf node to evaluate.
            parameters: Parameter mapping for the evaluation.
            ctx: Worker-side evaluation context for the leaf.

        Returns:
            A pair containing the raw objective value or Exception and the
            transferable result state of the worker-side context.

        Notes:
            Ordinary Exceptions are returned for parent-level handling.
            BaseException subclasses and context-export failures escape to the
            worker request processor.

        """

        objective = self.tree.nodes[node_id].objective

        try:
            result = objective(parameters, ctx)
            return result, ctx.to_result_state()
        except Exception as e:
            return e, ctx.to_result_state()

    def process_requests(self) -> None:
        """
        Process queued evaluation requests in the worker thread.

        The loop evaluates tasks sequentially on this rank and sends each raw
        outcome to rank 0. Cancellation prevents further queued tasks from
        starting, while shutdown terminates the thread.

        Notes:
            Failures in worker machinery are sent to rank 0 as MPIWorkerError
            and put this worker into cancellation mode. Ordinary objective
            Exceptions are returned by evaluate_leaf_worker instead.

        """

        while True:
            if self.event_shutdown_worker.is_set():
                return

            if self.event_cancel_work.is_set():
                continue

            try:
                request = self.work_requests.get_nowait()
            except Empty:
                continue

            try:
                parameters = cast("ParametersT", request.parameters)

                for task in request.tasks:
                    if self.event_cancel_work.is_set():
                        break

                    outcome, ctx_result_state = self.evaluate_leaf_worker(
                        task.node_id,
                        parameters=parameters,
                        ctx=task.ctx,
                    )

                    self.comm.send(
                        (
                            request.run_id,
                            task.node_id,
                            outcome,
                            ctx_result_state,
                        ),
                        dest=0,
                        tag=_RESULT_TAG,
                    )

            except BaseException as e:
                # This is not an objective failure. Something went wrong in the
                # MPI worker machinery itself, so the whole evaluation must abort.
                # Stop this worker from starting any further work from the current batch.
                self.event_cancel_work.set()

                self.comm.send(
                    MPIWorkerError(
                        f"Worker rank {self.rank} failed while processing "
                        f"run {request.run_id}: {e!r}"
                    ),
                    dest=0,
                    tag=_RESULT_TAG,
                )

            finally:
                self.work_requests.task_done()

    def worker_loop(self) -> None:
        """
        Receive coordinator commands on a nonzero rank until shutdown.

        Evaluation requests are placed on a local queue consumed by a worker
        thread. The main thread remains available for cancellation and shutdown
        commands.

        Raises:
            RuntimeError: If called on rank 0 or if an unknown command is
                received.

        """

        if self.rank == 0:
            msg = "worker_loop() cannot be used on rank 0"
            raise RuntimeError(msg)

        worker_thread = threading.Thread(target=self.process_requests)
        worker_thread.start()

        while True:
            msg = self.comm.recv(source=0, tag=_COMMAND_TAG)

            if isinstance(msg, _Shutdown):
                self.event_shutdown_worker.set()
                worker_thread.join()
                return

            if isinstance(msg, _CancelPending):
                # cancel all pending work by draining the queue
                self.event_cancel_work.set()

                # Remove work that hasn't started.
                while True:
                    try:
                        self.work_requests.get_nowait()
                    except Empty:  # noqa: PERF203
                        break
                    else:
                        self.work_requests.task_done()

                # Wait for the request currently being processed, if any.
                self.work_requests.join()

                # Tell rank 0 this worker is now quiescent.
                self.comm.send(
                    _EvaluationDone(),
                    dest=0,
                    tag=_RESULT_TAG,
                )

                self.event_cancel_work.clear()
                continue

            if not isinstance(msg, _EvaluateRequest):
                self.event_shutdown_worker.set()
                worker_thread.join()
                err_msg = f"Unknown MPI scheduler command: {msg!r}"
                raise RuntimeError(err_msg)

            # append work requests
            self.work_requests.put(msg)

    def evaluate_leaves(
        self, runs: Sequence[EvaluationRun[ParametersT]]
    ) -> Iterator[tuple[int, NodeId, float | Exception]]:
        """
        Dispatch pending leaves and yield remote completions on rank 0.

        Args:
            runs: Successfully prepared evaluation runs.

        Yields:
            Tuples containing the position in runs, completed leaf identifier,
            and raw node outcome. Remote completions are yielded in arrival
            order.

        Raises:
            RuntimeError: If called on a nonzero rank.
            MPIWorkerError: If worker machinery fails while processing a
                request.

        Notes:
            With a single-rank communicator, leaves are evaluated through
            SerialTreeSchedule. With multiple ranks, rank 0 coordinates only
            and does not execute leaves.

        """

        if self.rank != 0:
            msg = "evaluate_leaves() can only be used on rank 0"
            raise RuntimeError(msg)

        # Serial base-case
        if self.size == 1:
            yield from SerialTreeSchedule(self.tree).evaluate_leaves(runs)
            return

        for run_idx, run in enumerate(runs):
            eval_state = run.state
            parameters = run.parameters

            runtime = _MPIEvaluationRuntime(
                eval_id=0,
                started_workers=set(),
                done_workers=set(),
            )
            self.set_mpi_runtime(eval_state, runtime)

            for rank in range(1, self.size):
                tasks: list[_LeafTask] = []

                # build the list of tasks for each rank by checking
                # that leaf ids are (i) pending and (ii) in the leaf assignment of that rank
                for node_id in self.leaf_ids:
                    if (
                        eval_state.is_pending(node_id)
                        and node_id in self._leaf_assignments[rank]
                    ):
                        ctx = eval_state.contexts[node_id]
                        assert ctx is not None
                        tasks.append(_LeafTask(node_id=node_id, ctx=ctx))

                # once the tasks have been built, the request can be sent
                request = _EvaluateRequest(
                    run_id=run_idx,
                    parameters=parameters,
                    eval_id=runtime.eval_id,
                    tasks=tuple(tasks),
                )
                runtime.eval_id += 1

                # then we send the request to the rank
                self.comm.send(request, dest=rank, tag=_COMMAND_TAG)
                runtime.started_workers.add(rank)

        status = MPI.Status()

        # as long as there are pending leaf nodes we listen for results
        while any(
            run.state.is_pending(node_id)
            for run, node_id in product(runs, self.leaf_ids)
        ):
            msg = self.comm.recv(
                source=MPI.ANY_SOURCE,
                tag=_RESULT_TAG,
                status=status,
            )

            if isinstance(msg, MPIWorkerError):
                raise msg

            run_id, node_id, outcome, ctx_result_state = msg
            assert isinstance(outcome, NodeOutcome)

            ctx = runs[run_id].state.contexts[node_id]
            assert ctx is not None
            ctx.apply_result_state(ctx_result_state)

            yield run_id, node_id, outcome

    def cancel_pending_and_wait(
        self,
        runs: Sequence[EvaluationRun],
    ) -> None:
        """
        Quiesce MPI leaf work for a batch before aborting its lifecycles.

        Args:
            runs: Evaluation runs whose remote work must be stopped.

        Raises:
            RuntimeError: If called on a nonzero rank.

        Notes:
            Running leaves are allowed to finish. Workers observe cancellation
            between leaves and do not start further queued leaves. Late context
            states are restored on rank 0, but their numerical outcomes are
            not propagated after batch abort has begun.

        """

        if self.rank != 0:
            msg = "cancel_pending_and_wait() can only be used on rank 0"
            raise RuntimeError(msg)

        if self.size == 1:
            return

        # send _CancelPending requests to all ranks
        requests = [
            self.comm.isend(_CancelPending(), dest=rank, tag=_COMMAND_TAG)
            for rank in range(1, self.size)
        ]

        if requests:
            MPI.Request.Waitall(requests)

        done_workers: set[int] = set()
        status = MPI.Status()

        while len(done_workers) < self.size - 1:
            msg = self.comm.recv(
                source=MPI.ANY_SOURCE,
                tag=_RESULT_TAG,
                status=status,
            )

            source = status.Get_source()

            if isinstance(msg, _EvaluationDone):
                done_workers.add(source)
                continue

            if isinstance(msg, MPIWorkerError):
                # We are already aborting because of another failure.
                # Keep quiescing the workers.
                continue

            # A leaf that was already running when cancellation occurred
            # is allowed to finish. Restore its context state, but do not
            # propagate its numerical result after abort has begun.
            run_id, node_id, outcome, ctx_result_state = msg

            ctx = runs[run_id].state.contexts[node_id]
            assert ctx is not None
            ctx.apply_result_state(ctx_result_state)

    def release_workers(self) -> None:
        """
        Tell every persistent worker loop to exit.

        The operation is idempotent. It is a no-op on nonzero ranks, with a
        single-rank communicator, or after workers have already been released.

        """

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
        """Release persistent workers and mark the prepared schedule closed."""

        self.release_workers()
        super().close()


class MPITreeScheduler(Scheduler[MPITreeSchedule[Any]]):
    """
    Prepare tree schedules for a persistent MPI coordinator-worker backend.

    Every participating rank must prepare the same objective tree. Rank 0 then
    evaluates requests through the prepared schedule, while nonzero ranks enter
    MPITreeSchedule.worker_loop and remain there until rank 0 closes the
    schedule.

    Args:
        comm: MPI communicator containing rank 0 and all worker ranks. Defaults
            to MPI.COMM_WORLD.
        mpi_debug_log: Enable method-level communicator logging.

    """

    def __init__(
        self,
        comm: Any | None = None,
        *,
        mpi_debug_log: bool = False,
    ) -> None:
        """
        Initialize the reusable MPI scheduler configuration.

        Args:
            comm: MPI communicator containing all participating ranks. None
                selects MPI.COMM_WORLD.
            mpi_debug_log: Enable method-level communicator logging.

        """

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
        """
        Compile a combined objective into an MPI tree schedule.

        Args:
            objective: Root combined objective whose complete nested call tree
                should be distributed.
            profile: Optional cost profile. The current contiguous placement
                policy ignores measured costs.

        Returns:
            Prepared MPI schedule using this scheduler's communicator.

        Notes:
            All ranks must call prepare with structurally equivalent objective
            trees before rank 0 begins evaluation and worker ranks enter their
            persistent loops.

        """

        return MPITreeSchedule(
            tree=cob_to_call_tree(objective),
            comm=self.comm,
            mpi_debug_log=self.mpi_debug_log,
        )
