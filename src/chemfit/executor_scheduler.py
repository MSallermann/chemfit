from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Executor, Future, as_completed, wait
from typing import Any, Generic, TypeVar, cast

from chemfit.abstract_objective_function import (
    EvaluateContext,
)
from chemfit.callgraph import CallTree, CombineNode, LeafNode, NodeId, cob_to_call_tree
from chemfit.combined_objective_function import (
    CombinedObjectiveFunction,
    evaluate_weighted_term,
)
from chemfit.executor_utils import AttachContextAsReturnValue
from chemfit.scheduling import Scheduler
from chemfit.tree_schedule import EvaluationState, TermResult, TreeScheduleBase

ParametersT_contra = TypeVar(
    "ParametersT_contra", contravariant=True, bound=Mapping[str, Any]
)

evaluate_weighted_term_with_ctx = AttachContextAsReturnValue(evaluate_weighted_term)


class ExecutorTreeSchedule(
    TreeScheduleBase[ParametersT_contra], Generic[ParametersT_contra]
):
    def __init__(self, tree: CallTree, executor: Executor) -> None:
        """Initialize the executor schedule."""
        super().__init__(tree)
        self.executor = executor

    def close(self):
        super().close()
        self.executor.shutdown()

    def evaluate_leaf(
        self,
        node_id: int,
        parameters: ParametersT_contra,
        eval_state: EvaluationState,
    ) -> tuple[NodeId, TermResult]:
        """
        Evaluate one leaf and return the term value it contributes to its parent.

        We have to override this from the base class to implement the attached context result mechanism.
        """

        node = self.tree.nodes[node_id]
        assert isinstance(node, LeafNode)

        assert node.parent_id is not None
        assert node.child_idx is not None

        parent = self.tree.nodes[node.parent_id]
        assert isinstance(parent, CombineNode)

        ctx = eval_state.contexts[node_id]
        assert ctx is not None

        res, ctx_res_state = evaluate_weighted_term_with_ctx(
            node.objective,
            parent.objective.weights[node.child_idx],
            parent.objective.exception_handler,
            parameters,
            node.child_idx,
            ctx,
        )

        ctx.apply_result_state(ctx_res_state)
        return (node_id, res)

    def set_leaf_futures(self, eval_state: EvaluationState, fs: Iterable[Future]):
        root_ctx: EvaluateContext = cast(
            "EvaluateContext", eval_state.contexts[self.tree.root]
        )
        root_ctx.temp.leaf_futures = list(fs)

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

        fs = []

        # 1. submit each leaf as a future and save the futures
        self.set_leaf_futures(
            eval_state,
            (
                self.executor.submit(
                    self.evaluate_leaf, node_id, parameters, eval_state
                )
                for node_id in self.leaf_ids
            ),
        )

        # 2. yield future results in completion order
        for fs in as_completed(self.get_leaf_futures(eval_state=eval_state)):
            yield fs.result()

    def cancel_pending_and_wait(self, eval_state: EvaluationState) -> None:
        """Cancel pending leaves and wait for other leaves to complete."""

        futures = self.get_leaf_futures(eval_state)

        # first try to cancel all leaf futures
        for fs in futures:
            assert fs is not None
            fs.cancel()

        wait(futures)


class ExecutorTreeScheduler(Scheduler[ParametersT_contra], Generic[ParametersT_contra]):
    def __init__(self, executor_factory: Callable[[], Executor]) -> None:
        """Initialize executor schedule."""

        super().__init__()
        self.executor_factory = executor_factory

    def prepare(
        self,
        objective: CombinedObjectiveFunction[ParametersT_contra],
        /,
        *,
        profile: Mapping[tuple[int, ...], float] | None = None,  # noqa: ARG002
    ) -> TreeScheduleBase[ParametersT_contra]:
        return ExecutorTreeSchedule(
            tree=cob_to_call_tree(objective), executor=self.executor_factory()
        )
