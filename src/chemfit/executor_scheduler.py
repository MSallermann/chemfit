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
    Evaluate one leaf on a worker.

    The worker returns both the parent-facing term result and the resulting
    context state. The coordinator applies the context state to its own copy
    of the context.
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
    def __init__(self, tree: CallTree, executor: Executor, owns_executor: bool) -> None:
        """Initialize the executor schedule."""
        super().__init__(tree)
        self.executor = executor
        self.owns_executor = owns_executor

    def close(self):
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
        Evaluate one leaf and return the term value it contributes to its parent.

        We have to override this from the base class to implement the attached context result mechanism.
        """
        msg = "Use evaluate_leaf_worker instead."
        raise RuntimeError(msg)

    def set_leaf_futures(self, eval_state: EvaluationState, fs: list[Future]):
        root_ctx: EvaluateContext = cast(
            "EvaluateContext", eval_state.contexts[self.tree.root]
        )
        root_ctx.temp.leaf_futures = fs

    def get_leaf_futures(self, eval_state: EvaluationState) -> list[Future]:
        root_ctx: EvaluateContext = cast(
            "EvaluateContext", eval_state.contexts[self.tree.root]
        )
        return root_ctx.temp.leaf_futures

    def evaluate_leaves(
        self,
        runs: Sequence[EvaluationRun[ParametersT_contra]],
    ) -> Iterator[tuple[int, NodeId, NodeOutcome]]:
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
        """Cancel pending leaves and wait for other leaves to complete."""

        futures = []
        for run in runs:
            futures.extend(self.get_leaf_futures(run.state))

        # first try to cancel all leaf futures
        for fs in futures:
            assert fs is not None
            fs.cancel()

        wait(futures)


class ExecutorTreeScheduler(Scheduler[ExecutorTreeSchedule[Any]]):
    def __init__(
        self,
        /,
        executor_factory: Callable[[], Executor] | None = None,
        executor: Executor | None = None,
    ) -> None:
        """Initialize executor schedule."""

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
