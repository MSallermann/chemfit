from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from chemfit.abstract_objective_function import EvaluateContext
from chemfit.callgraph import (
    CallTree,
    CombineNode,
    LeafNode,
    NodeId,
    cob_to_call_tree,
)
from chemfit.combined_objective_function import (
    CombinedObjectiveFunction,
    evaluate_weighted_term,
)
from chemfit.scheduling import (
    EvaluationRequest,
    EvaluationResult,
    PreparedScheduleBase,
    Scheduler,
)

ParametersT_contra = TypeVar(
    "ParametersT_contra", contravariant=True, bound=Mapping[str, Any]
)
ParametersT = TypeVar("ParametersT", bound=Mapping[str, Any])
ParametersT_co = TypeVar("ParametersT_co", covariant=True, bound=Mapping[str, Any])


class _Pending:
    """
    Sentinel for pending evaluations.

    Note:
    We cannot use None since that has a special meaning in ChemFit already.

    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<PENDING>"


PENDING = _Pending()

TermResult = float | None
TermSlot = TermResult | _Pending


@dataclass
class EvaluationState:
    contexts: list[EvaluateContext | None]
    term_results: list[TermSlot]
    remaining_children: list[int]
    open_nodes: set[NodeId]

    def __init__(self, tree: CallTree, root_ctx: EvaluateContext):
        """Initialize the eval state."""
        self.contexts = [None] * len(tree.nodes)
        self.contexts[tree.root] = root_ctx
        self.term_results = [PENDING] * len(tree.nodes)
        self.remaining_children = [
            len(n.children) if isinstance(n, CombineNode) else 0 for n in tree.nodes
        ]
        self.open_nodes = set()


@dataclass
class EvaluationRun(Generic[ParametersT_co]):
    index: int
    parameters: ParametersT_co
    state: EvaluationState


class TreeScheduleBase(
    PreparedScheduleBase[ParametersT_contra], Generic[ParametersT_contra]
):
    def __init__(self, tree: CallTree) -> None:
        """Initialize the tree schedule from a call tree."""
        super().__init__()
        self.tree = tree
        self.leaf_ids = tuple(
            node.id for node in self.tree.nodes if isinstance(node, LeafNode)
        )

    def top_down_pass(
        self,
        parameters: ParametersT_contra,
        root_ctx: EvaluateContext,
        eval_state: EvaluationState,
    ) -> None:
        """Construct the context tree and begin nested COB life cycles."""

        def visit(
            combine_ctx: EvaluateContext,
            node_id: int,
        ) -> None:
            node = self.tree.nodes[node_id]
            assert isinstance(node, CombineNode)

            cob = node.objective

            cob._begin_evaluation(  # noqa: SLF001
                parameters=parameters,
                ctx=combine_ctx,
            )
            eval_state.open_nodes.add(node_id)

            child_contexts = combine_ctx.spawn_children(
                cob.n_terms(),
                cob.child_context_configurator,
            )

            for child_id, child_ctx in zip(
                node.children,
                child_contexts,
                strict=True,
            ):
                eval_state.contexts[child_id] = child_ctx

                child_node = self.tree.nodes[child_id]

                if isinstance(child_node, CombineNode):
                    visit(combine_ctx=child_ctx, node_id=child_id)

        try:
            visit(combine_ctx=root_ctx, node_id=self.tree.root)
        except BaseException as e:
            self.abort_evaluation(exception=e, eval_state=eval_state)
            raise

    def abort_evaluation(
        self,
        exception: BaseException,
        eval_state: EvaluationState,
    ) -> None:
        """
        Abort an evaluation and close all still-open nested COB lifecycles.

        Nested combine nodes are closed bottom-up so that descendants are ended
        before their ancestors. The root COB is not closed here; its lifecycle is
        owned by ObjectiveFunctor.__call__().

        This method assumes that no leaf evaluations are still running.

        Any exception raised while closing a COB is attached as a note to the
        original exception. Cleanup continues so that every open lifecycle gets
        exactly one attempt to close.
        """

        def visit(node_id: NodeId) -> None:
            node = self.tree.nodes[node_id]

            if not isinstance(node, CombineNode):
                return

            # Descendants must be closed before this node.
            for child_id in node.children:
                visit(child_id)

            if node_id not in eval_state.open_nodes:
                return

            ctx = eval_state.contexts[node_id]
            assert ctx is not None

            try:
                node.objective._end_evaluation(  # noqa: SLF001
                    ctx,
                    exception,
                )
            except BaseException as cleanup_error:
                exception.add_note(
                    f"Exception while aborting combine node {node_id}: "
                    f"{cleanup_error!r}"
                )
            finally:
                # Calling _end_evaluation consumes this lifecycle even if a
                # post-evaluation hook itself raises. Retrying it would run hooks
                # twice.
                eval_state.open_nodes.discard(node_id)

        visit(self.tree.root)

    def finish_combine_node(self, node_id: int, eval_state: EvaluationState):
        """
        Finish a combine node, after all its terms have been computed.

        This includes the collection of the child meta-data, the reduction of the individual terms and the end of the evaluation lifecyle
        """

        node = self.tree.nodes[node_id]
        assert isinstance(node, CombineNode)
        cob = node.objective

        terms = cast(
            "list[float|None]",
            [eval_state.term_results[child_id] for child_id in node.children],
        )

        for t in terms:
            assert t is not PENDING

        ctx = eval_state.contexts[node_id]
        assert ctx is not None

        ctx.collect_child_meta_data(recursive=False)

        try:
            value = cob._reduce_terms(terms, ctx)  # noqa: SLF001
            ctx.loss = value
        except BaseException as e:
            cob._end_evaluation(ctx, e)  # noqa: SLF001
            raise
        else:
            cob._end_evaluation(ctx, None)  # noqa: SLF001
        finally:
            eval_state.open_nodes.discard(node_id)

        return value

    def evaluate_leaf(
        self,
        node_id: int,
        parameters: ParametersT_contra,
        eval_state: EvaluationState,
    ) -> TermResult:
        """Evaluate one leaf and return the term value it contributes to its parent."""

        node = self.tree.nodes[node_id]
        assert isinstance(node, LeafNode)

        assert node.parent_id is not None
        assert node.child_idx is not None

        parent = self.tree.nodes[node.parent_id]
        assert isinstance(parent, CombineNode)

        ctx = eval_state.contexts[node_id]
        assert ctx is not None

        return evaluate_weighted_term(
            objective=node.objective,
            weight=parent.objective.weights[node.child_idx],
            exception_handler=parent.objective.exception_handler,
            parameters=parameters,
            idx=node.child_idx,
            ctx=ctx,
        )

    def propagate_completion(
        self,
        node_id: NodeId,
        result: TermResult,
        eval_state: EvaluationState,
    ) -> None | float:
        """
        Record a completed term and propagate completion towards the root.

        ``result`` is the value contributed by ``node_id`` to its parent, i.e.
        after the parent's weight and exception handling have been applied.

        Whenever this completes the last outstanding child of a nested combine
        node, that combine node is reduced immediately and its resulting term is
        propagated further upwards.

        The root combine node is not reduced here. Once all of its immediate
        children have completed, propagation stops and the root reduction remains
        the responsibility of ``CombinedObjectiveFunction._evaluate()``.
        """

        value_to_propagate: TermResult = result
        current_node_id: NodeId = node_id

        while True:
            node = self.tree.nodes[current_node_id]

            # This function should never be invoked on nodes that are not PENDING
            # so we check that
            if eval_state.term_results[current_node_id] is not PENDING:
                msg = f"Node {current_node_id} has already completed."
                raise RuntimeError(msg)

            # Record the result in the eval_state.
            # This is already the value this node contributes to its parent with weights applied.
            eval_state.term_results[current_node_id] = value_to_propagate

            # Now check if this nodes parent can be completed
            parent_id = node.parent_id

            # Make sure the parent exists (only the root has no parent)
            # If there is no parent this means we have invoked propagate_completion on the root,
            # which is technically not sound
            assert parent_id is not None

            # decrease the number of pending children by one
            eval_state.remaining_children[parent_id] -= 1

            # if the number of pending children is less than zero something went wrong
            if eval_state.remaining_children[parent_id] < 0:
                msg = f"Combine node {parent_id} received too many completions."
                raise RuntimeError(msg)

            # If the parent is still waiting for another child, there is nothing to dos
            if eval_state.remaining_children[parent_id] > 0:
                return None

            # If this was the last missing child, we can finish the combine node
            # (aka the parent of the current node).
            #
            # The parent can either:
            # (i) be the root, in which case finishing it completes this run, or
            # (ii) itself be a child of another combine node, in which case its
            #      result must be converted into the term it contributes to its parent
            #      by applying that parent's weight / exception handler.

            parent = self.tree.nodes[parent_id]
            assert isinstance(parent, CombineNode)

            grandparent_id = parent.parent_id

            # Case (i): the parent is the root, so its reduced value is the final result.
            if grandparent_id is None:
                return self.finish_combine_node(
                    parent_id,
                    eval_state,
                )

            # Case (ii): the parent is nested inside another combine node.
            # Finishing it gives its own raw loss; we then convert that into the
            # parent-facing term that should continue propagating upward.

            parent_ctx = eval_state.contexts[parent_id]
            assert parent_ctx is not None
            child_idx = parent.child_idx
            assert child_idx is not None

            grandparent = self.tree.nodes[grandparent_id]
            assert isinstance(grandparent, CombineNode)

            try:
                raw_result = self.finish_combine_node(
                    parent_id,
                    eval_state,
                )

                value_to_propagate = (
                    raw_result * grandparent.objective.weights[child_idx]
                )

            except Exception as e:
                value_to_propagate = grandparent.objective.exception_handler(
                    e,
                    parent_ctx,
                    child_idx,
                )

            # The completed parent now behaves exactly like another completed
            # child term, so continue propagating upward.
            current_node_id = parent_id

    def evaluate_leaves(
        self,
        runs: Sequence[EvaluationRun[ParametersT_contra]],
    ) -> Iterator[tuple[int, NodeId, TermResult]]:
        """Evaluate leaves and yield completion events. The meaning of the tuple is [run_id, node_id, result]."""
        raise NotImplementedError

    def cancel_pending_and_wait(self, eval_state: EvaluationState) -> None:
        """Cancel pending leaves and wait for other leaves to complete."""
        raise NotImplementedError

    def evaluate_many(
        self, requests: Sequence[EvaluationRequest[ParametersT_contra]]
    ) -> Iterator[EvaluationResult]:
        runs: list[EvaluationRun[ParametersT_contra]] = []

        try:
            # 1. Perform the top-down pass for all requests
            for idx, req in enumerate(requests):
                eval_state = EvaluationState(
                    self.tree,
                    root_ctx=req.ctx,
                )

                run = EvaluationRun(
                    index=idx,
                    parameters=req.parameters,
                    state=eval_state,
                )
                runs.append(run)

                self.top_down_pass(
                    req.parameters,
                    root_ctx=req.ctx,
                    eval_state=eval_state,
                )

        except BaseException as e:
            # No leaves have been submitted yet, so there is nothing to cancel.
            for run in runs:
                self.abort_evaluation(e, run.state)
            raise

        # 2. perform leaf evaluation
        try:
            for run_id, node_id, result in self.evaluate_leaves(runs):
                eval_state = runs[run_id].state
                # after a leaf result is available we propagate the result upwards
                propagate_result = self.propagate_completion(
                    node_id, result, eval_state
                )
                if propagate_result is not None:
                    yield EvaluationResult(run_id, propagate_result)
        except BaseException as e:
            for run in runs:
                self.cancel_pending_and_wait(eval_state=run.state)
                self.abort_evaluation(e, run.state)
            raise


class SerialTreeSchedule(TreeScheduleBase[ParametersT], Generic[ParametersT]):
    def evaluate_leaves(
        self, runs: Sequence[EvaluationRun[ParametersT_contra]]
    ) -> Iterator[tuple[int, NodeId, float | None]]:
        for run in runs:
            for node_id in self.leaf_ids:
                yield (
                    run.index,
                    node_id,
                    self.evaluate_leaf(node_id, run.parameters, run.state),
                )

    # def evaluate_leaves(
    #     self,
    #     parameters: ParametersT_contra,
    #     eval_state: EvaluationState,
    # ) -> Iterator[tuple[NodeId, TermResult]]:
    #     """Evaluate leaves serially and yield completion events."""

    #     for node_id in self.leaf_ids:
    #         yield (
    #             node_id,
    #             self.evaluate_leaf(
    #                 node_id,
    #                 parameters,
    #                 eval_state,
    #             ),
    #         )

    def cancel_pending_and_wait(self, eval_state: EvaluationState) -> None: ...


class SerialTreeScheduler(Scheduler[SerialTreeSchedule[Any]]):
    def prepare(
        self,
        objective: CombinedObjectiveFunction[ParametersT_contra],
        /,
        *,
        profile: Mapping[tuple[int, ...], float] | None = None,  # noqa: ARG002
    ) -> SerialTreeSchedule[ParametersT_contra]:
        return SerialTreeSchedule(tree=cob_to_call_tree(objective))
