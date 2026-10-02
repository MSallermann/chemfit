"""
Executor-backed tree scheduling for combined-objective evaluations.

ExecutorTreeScheduler prepares a combined-objective call tree and binds it to
either a caller-provided executor or an executor created by a factory. Leaf
objectives are submitted as independent futures, while TreeScheduleBase
retains responsibility for nested reduction, exception handling, evaluation
lifecycles, and result ordering.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Executor, Future, as_completed, wait
from typing import Any, Generic, TypeVar, cast

from chemfit.abstract_objective_function import EvaluateContext, ObjectiveFunctor
from chemfit.callgraph import CallTree, NodeId, cob_to_call_tree
from chemfit.combined_objective_function import (
    CombinedObjectiveFunction,
)
from chemfit.scheduling import Scheduler
from chemfit.tree_schedule import (
    EvaluationRun,
    EvaluationState,
    NodeOutcome,
    TreeScheduleBase,
)

ParametersT_contra = TypeVar(
    "ParametersT_contra", contravariant=True, bound=Mapping[str, Any]
)

# evaluate_weighted_term_with_ctx = AttachContextAsReturnValue(evaluate_weighted_term)


def evaluate_leaf_worker(
    objective: ObjectiveFunctor, parameters: Mapping[str, Any], ctx: EvaluateContext
) -> tuple[NodeOutcome, dict[str, Any]]:
    """
    Evaluate one leaf and return its outcome with transferable context state.

    Args:
        objective: Leaf objective to evaluate.
        parameters: Parameter mapping for the evaluation.
        ctx: Worker-side evaluation context for the leaf.

    Returns:
        A pair containing the raw objective value or Exception and the
        serializable result state of the worker-side context.

    Notes:
        Ordinary Exceptions are returned as node outcomes so the parent
        combined objective can apply its exception handler. BaseException
        subclasses and failures while exporting context state escape to the
        future.

    """

    try:
        result = objective(
            parameters,
            ctx,
        )
        return result, ctx.to_result_state()
    except Exception as e:
        return e, ctx.to_result_state()


class ExecutorTreeSchedule(
    TreeScheduleBase[ParametersT_contra], Generic[ParametersT_contra]
):
    """
    Evaluate tree leaves concurrently through an Executor.

    Args:
        tree: Compiled combined-objective call tree.
        executor: Executor used to submit leaf evaluations.
        owns_executor: Whether closing this schedule should shut down the
            executor.

    Attributes:
        executor: Executor used by this prepared schedule.
        owns_executor: Whether the executor was created for this schedule.

    """

    def __init__(self, tree: CallTree, executor: Executor, owns_executor: bool) -> None:
        """Initialize the executor schedule."""

        super().__init__(tree)
        self.executor = executor
        self.owns_executor = owns_executor

    def close(self):
        """
        Close the schedule and shut down an owned executor.

        A caller-provided executor is left open. Repeated calls are safe and do
        not attempt to shut down the executor again.

        """

        if not self.closed and self.owns_executor:
            self.executor.shutdown()
        super().close()

    def evaluate_leaf(
        self,
        node_id: NodeId,  # noqa: ARG002
        parameters: ParametersT_contra,  # noqa: ARG002
        eval_state: EvaluationState,  # noqa: ARG002
    ) -> NodeOutcome:
        """
        Reject direct leaf evaluation on the coordinator.

        Args:
            node_id: Identifier of the leaf that would be evaluated.
            parameters: Parameter mapping for the evaluation.
            eval_state: State containing the leaf context.

        Raises:
            RuntimeError: Always. Executor schedules must use
                evaluate_leaf_worker through evaluate_leaves so worker context
                state can be restored on the coordinator.

        """

        msg = "Use evaluate_leaf_worker instead."
        raise RuntimeError(msg)

    def set_leaf_futures(self, eval_state: EvaluationState, fs: list[Future]):
        """
        Attach submitted leaf futures to an evaluation's root context.

        Args:
            eval_state: Evaluation state whose futures should be recorded.
            fs: Mutable list of futures submitted for the evaluation.

        """

        root_ctx: EvaluateContext = cast(
            "EvaluateContext", eval_state.contexts[self.tree.root]
        )
        root_ctx.temp.leaf_futures = fs

    def get_leaf_futures(self, eval_state: EvaluationState) -> list[Future]:
        """
        Return the leaf futures recorded for an evaluation.

        Args:
            eval_state: Evaluation state whose futures should be retrieved.

        Returns:
            Mutable list of futures submitted for the evaluation.

        """

        root_ctx: EvaluateContext = cast(
            "EvaluateContext", eval_state.contexts[self.tree.root]
        )
        return root_ctx.temp.leaf_futures

    def evaluate_leaves(
        self,
        runs: Sequence[EvaluationRun[ParametersT_contra]],
    ) -> Iterator[tuple[int, NodeId, NodeOutcome]]:
        """
        Submit every pending leaf and yield outcomes in completion order.

        Args:
            runs: Successfully prepared evaluation runs.

        Yields:
            Tuples containing the position in runs, completed leaf identifier,
            and raw node outcome.

        Notes:
            Worker context result state is applied to the corresponding
            coordinator-side leaf context before its completion is yielded.
            Future bookkeeping is initialized before submission so partial
            submission failures can be cancelled safely.

        """

        future_to_leaf: dict[
            Future[tuple[NodeOutcome, dict[str, Any]]],
            tuple[int, NodeId],
        ] = {}

        # Initialize the future bookkeeping for every run before submitting
        # anything. If submission fails partway through, cancellation can then
        # safely inspect every run.
        for run in runs:
            self.set_leaf_futures(run.state, [])

        # 1. Submit all pending leaves across all evaluations.
        for run_idx, run in enumerate(runs):
            eval_state = run.state
            parameters = run.parameters
            leaf_futures = self.get_leaf_futures(eval_state)

            for node_id in self.leaf_ids:
                # Only leaves that were reached successfully during the top-down
                # pass should be submitted. Leaves below a failed setup remain
                # INACTIVE.
                if not eval_state.is_pending(node_id):
                    continue

                leaf_node = self.tree.nodes[node_id]

                ctx = eval_state.contexts[node_id]
                assert ctx is not None

                future = self.executor.submit(
                    evaluate_leaf_worker,
                    objective=leaf_node.objective,
                    parameters=parameters,
                    ctx=ctx,
                )

                leaf_futures.append(future)
                future_to_leaf[future] = (run_idx, node_id)

        # 2. Yield future results in completion order.
        for future in as_completed(future_to_leaf):
            run_idx, node_id = future_to_leaf[future]
            eval_state = runs[run_idx].state

            result, ctx_result_state = future.result()

            ctx = eval_state.contexts[node_id]
            assert ctx is not None
            ctx.apply_result_state(ctx_result_state)

            yield run_idx, node_id, result

    def cancel_pending_and_wait(self, runs: Sequence[EvaluationRun]) -> None:
        """
        Cancel queued leaf futures and wait for running futures to finish.

        Args:
            runs: Evaluation runs whose submitted work must be quiesced.

        Notes:
            Future cancellation is best-effort. Work that has already started
            is allowed to finish before this method returns.

        """

        futures = []
        for run in runs:
            futures.extend(self.get_leaf_futures(run.state))

        # first try to cancel all leaf futures
        for fs in futures:
            assert fs is not None
            fs.cancel()

        wait(futures)


class ExecutorTreeScheduler(Scheduler[ExecutorTreeSchedule[Any]]):
    """
    Prepare tree schedules backed by a reusable or per-schedule executor.

    Exactly one executor source must be provided. A supplied executor remains
    owned by the caller and can be shared across schedules. An executor factory
    creates a new executor for each prepared schedule, which then owns and
    shuts down that executor.

    Args:
        executor_factory: Zero-argument callable that creates an executor for
            each prepared schedule.
        executor: Existing executor to reuse without taking ownership.

    Raises:
        ValueError: If both executor sources or neither source are provided.

    """

    def __init__(
        self,
        /,
        executor_factory: Callable[[], Executor] | None = None,
        executor: Executor | None = None,
    ) -> None:
        """
        Initialize the executor-backed scheduler configuration.

        Args:
            executor_factory: Zero-argument callable that creates an executor
                for each prepared schedule.
            executor: Existing executor to reuse without taking ownership.

        Raises:
            ValueError: If both arguments or neither argument are provided.

        """

        if (executor is None) == (executor_factory is None):
            msg = "Specify exactly one of executor or executor_factory."
            raise ValueError(msg)

        self.executor = executor
        self.executor_factory = executor_factory

        super().__init__()
        self.executor_factory = executor_factory

    def prepare(
        self,
        objective: CombinedObjectiveFunction[ParametersT_contra],
        /,
        *,
        profile: Mapping[tuple[int, ...], float] | None = None,  # noqa: ARG002
    ) -> ExecutorTreeSchedule[ParametersT_contra]:
        """
        Compile a combined objective and bind it to an executor.

        Args:
            objective: Root combined objective whose complete nested call tree
                should be scheduled.
            profile: Optional cost profile. The executor backend currently
                ignores static placement costs.

        Returns:
            Prepared executor-backed tree schedule. It owns the executor only
            when that executor was created by executor_factory.

        """

        if self.executor is not None:
            executor = self.executor
            owns_executor = False
        else:
            assert self.executor_factory is not None
            executor = self.executor_factory()
            owns_executor = True

        return ExecutorTreeSchedule(
            tree=cob_to_call_tree(objective),
            executor=executor,
            owns_executor=owns_executor,
        )
