from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Executor, Future, as_completed, wait
from typing import Any, Generic, TypeVar, cast

from chemfit.abstract_objective_function import EvaluateContext, ObjectiveFunctor
from chemfit.callgraph import CallTree, CombineNode, NodeId, cob_to_call_tree
from chemfit.combined_objective_function import (
    CombinedObjectiveFunction,
    ExceptionHandler,
    evaluate_weighted_term,
)
from chemfit.executor_utils import AttachContextAsReturnValue
from chemfit.scheduling import Scheduler
from chemfit.tree_schedule import EvaluationState, TermResult, TreeScheduleBase

ParametersT_contra = TypeVar(
    "ParametersT_contra", contravariant=True, bound=Mapping[str, Any]
)

evaluate_weighted_term_with_ctx = AttachContextAsReturnValue(evaluate_weighted_term)


def evaluate_leaf_worker(
    node_id: NodeId,
    objective: ObjectiveFunctor[ParametersT_contra],
    weight: float,
    exception_handler: ExceptionHandler,
    parameters: ParametersT_contra,
    child_idx: int,
    ctx: EvaluateContext,
) -> tuple[NodeId, float | None, dict[str, Any]]:
    """
    Evaluate one leaf on a worker.

    The worker returns both the parent-facing term result and the resulting
    context state. The coordinator applies the context state to its own copy
    of the context.
    """

    result, ctx_result_state = evaluate_weighted_term_with_ctx(
        objective,
        weight,
        exception_handler,
        parameters,
        child_idx,
        ctx,
    )

    return node_id, result, ctx_result_state


class ExecutorTreeSchedule(
    TreeScheduleBase[ParametersT_contra], Generic[ParametersT_contra]
):
    def __init__(self, tree: CallTree, executor: Executor, owns_executor: bool) -> None:
        """Initialize the executor schedule."""
        super().__init__(tree)
        self.executor = executor
        self.owns_executor = owns_executor

    def close(self):
        super().close()
        if self.owns_executor:
            self.executor.shutdown()

    def evaluate_leaf(
        self,
        node_id: int,  # noqa: ARG002
        parameters: ParametersT_contra,  # noqa: ARG002
        eval_state: EvaluationState,  # noqa: ARG002
    ) -> TermResult:
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
        parameters: ParametersT_contra,
        eval_state: EvaluationState,
    ) -> Iterator[tuple[NodeId, TermResult]]:
        """Evaluate leaves and yield completion events."""

        # 1. submit each leaf as a future and save the futures in the eval state
        leaf_futures = []
        self.set_leaf_futures(eval_state, leaf_futures)

        for node_id in self.leaf_ids:
            leaf_node = self.tree.nodes[node_id]
            assert leaf_node.parent_id is not None
            parent_node = self.tree.nodes[leaf_node.parent_id]
            assert isinstance(parent_node, CombineNode)
            assert leaf_node.child_idx is not None
            weight = parent_node.objective.weights[leaf_node.child_idx]
            exception_handler = parent_node.objective.exception_handler
            ctx = eval_state.contexts[node_id]
            assert ctx is not None

            leaf_futures.append(
                self.executor.submit(
                    evaluate_leaf_worker,
                    node_id,
                    objective=leaf_node.objective,
                    weight=weight,
                    exception_handler=exception_handler,
                    parameters=parameters,
                    child_idx=leaf_node.child_idx,
                    ctx=ctx,
                )
            )

        # 2. yield future results in completion order
        for fs in as_completed(self.get_leaf_futures(eval_state=eval_state)):
            node_id, result, ctx_result_state = fs.result()
            eval_state.contexts[node_id].apply_result_state(ctx_result_state)
            yield node_id, result

    def cancel_pending_and_wait(self, eval_state: EvaluationState) -> None:
        """Cancel pending leaves and wait for other leaves to complete."""

        futures = self.get_leaf_futures(eval_state)

        # first try to cancel all leaf futures
        for fs in futures:
            assert fs is not None
            fs.cancel()

        wait(futures)


class ExecutorTreeScheduler(Scheduler[ExecutorTreeSchedule[Any]]):
    def __init__(
        self,
        executor_factory: Callable[[], Executor] | None,
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
