from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
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
from chemfit.scheduling import PreparedSchedule, Scheduler

ParametersT_contra = TypeVar(
    "ParametersT_contra", contravariant=True, bound=Mapping[str, Any]
)


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

    extra_state: SimpleNamespace

    def __init__(self, tree: CallTree, root_ctx: EvaluateContext):
        """Initialize the eval state."""
        self.contexts = [None] * len(tree.nodes)
        self.contexts[tree.root] = root_ctx
        self.term_results = [PENDING] * len(tree.nodes)
        self.remaining_children = [
            len(n.children) if isinstance(n, CombineNode) else 0 for n in tree.nodes
        ]
        self.open_nodes = set()


class TreeScheduleBase(
    PreparedSchedule[ParametersT_contra], Generic[ParametersT_contra]
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
        """
        Construct the context tree and begin nested COB life cycles.

        The root lifecycle is already managed by ObjectiveFunctor.__call__().
        """

        def visit(
            combine_ctx: EvaluateContext,
            node_id: int,
            begin_lifecycle: bool,
        ) -> None:
            node = self.tree.nodes[node_id]
            assert isinstance(node, CombineNode)

            cob = node.objective

            if begin_lifecycle:
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
                    visit(
                        combine_ctx=child_ctx,
                        node_id=child_id,
                        begin_lifecycle=True,
                    )

        try:
            visit(
                combine_ctx=root_ctx,
                node_id=self.tree.root,
                begin_lifecycle=False,
            )
        # we need to close life cycles of open combine nodes in *reverse* order
        # so that exceptions are propagated correctly from inner to outer scopes
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

            # The root should never be managed by this set.
            assert node_id != self.tree.root

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
            ctx.loss = cob._reduce_terms(terms, ctx)  # noqa: SLF001
        except BaseException as e:
            cob._end_evaluation(ctx, e)  # noqa: SLF001
            raise
        else:
            cob._end_evaluation(ctx, None)  # noqa: SLF001
        finally:
            eval_state.open_nodes.discard(node_id)

        return cast("float", ctx.loss)

    def evaluate_leaf(
        self,
        node_id: int,
        parameters: ParametersT_contra,
        eval_state: EvaluationState,
    ) -> tuple[NodeId, TermResult]:
        """Evaluate one leaf and return the term value it contributes to its parent."""

        node = self.tree.nodes[node_id]
        assert isinstance(node, LeafNode)

        assert node.parent_id is not None
        assert node.child_idx is not None

        parent = self.tree.nodes[node.parent_id]
        assert isinstance(parent, CombineNode)

        ctx = eval_state.contexts[node_id]
        assert ctx is not None

        return node_id, evaluate_weighted_term(
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
    ) -> None:
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

        while True:
            node = self.tree.nodes[node_id]

            if eval_state.term_results[node_id] is not PENDING:
                msg = f"Node {node_id} has already completed."
                raise RuntimeError(msg)

            # This is already the value this node contributes to its parent.
            eval_state.term_results[node_id] = result

            parent_id = node.parent_id

            # Only the root has no parent.
            if parent_id is None:
                return

            eval_state.remaining_children[parent_id] -= 1

            if eval_state.remaining_children[parent_id] < 0:
                msg = f"Combine node {parent_id} received too many completions."
                raise RuntimeError(msg)

            # Parent is still waiting for another child.
            if eval_state.remaining_children[parent_id] > 0:
                return

            # The root is deliberately not finished here.
            if parent_id == self.tree.root:
                return

            parent = self.tree.nodes[parent_id]
            assert isinstance(parent, CombineNode)
            assert parent.parent_id is not None
            assert parent.child_idx is not None

            parent_ctx = eval_state.contexts[parent_id]
            assert parent_ctx is not None

            grandparent = self.tree.nodes[parent.parent_id]
            assert isinstance(grandparent, CombineNode)

            try:
                raw_result = self.finish_combine_node(
                    parent_id,
                    eval_state,
                )

                # Convert the completed COB's own loss into the term value that
                # it contributes to its parent.
                result = raw_result * grandparent.objective.weights[parent.child_idx]

            except Exception as e:
                result = grandparent.objective.exception_handler(
                    e,
                    parent_ctx,
                    parent.child_idx,
                )

            # The completed parent now behaves exactly like another completed
            # child term, so continue upwards.
            node_id = parent_id

    def evaluate_leaves(
        self,
        parameters: ParametersT_contra,
        eval_state: EvaluationState,
    ) -> Iterator[tuple[NodeId, TermResult]]:
        """Evaluate leaves and yield completion events."""
        raise NotImplementedError

    def cancel_pending_and_wait(self, eval_state: EvaluationState) -> None:
        """Cancel pending leaves and wait for other leaves to complete."""
        raise NotImplementedError

    def evaluate_terms(
        self,
        parameters: ParametersT_contra,
        ctx: EvaluateContext,
        /,
    ) -> list[float | None]:
        """Evaluate all terms in index order."""

        # 1. create the tree eval state
        eval_state = EvaluationState(self.tree, root_ctx=ctx)

        # 2. perform the top down pass
        self.top_down_pass(parameters, root_ctx=ctx, eval_state=eval_state)

        # 3. perform leaf evaluation
        try:
            for node_id, result in self.evaluate_leaves(
                parameters=parameters, eval_state=eval_state
            ):
                self.propagate_completion(
                    node_id,
                    result,
                    eval_state,
                )
        except BaseException as e:
            self.cancel_pending_and_wait(eval_state=eval_state)
            self.abort_evaluation(e, eval_state)
            raise

        # 4. collect root node metadata
        root_node = cast("CombineNode", self.tree.nodes[self.tree.root])
        ctx.collect_child_meta_data(recursive=False)

        # 5. return only the terms of the outermost objective function
        terms = cast(
            "list[float|None]",
            [eval_state.term_results[child_id] for child_id in root_node.children],
        )

        for t in terms:
            assert t is not PENDING

        return terms

    def close(self): ...


class SerialTreeSchedule(
    TreeScheduleBase[ParametersT_contra], Generic[ParametersT_contra]
):
    def evaluate_leaves(
        self,
        parameters: ParametersT_contra,
        eval_state: EvaluationState,
    ) -> Iterator[tuple[NodeId, TermResult]]:
        """Evaluate leaves serially and yield completion events."""

        for node_id in self.leaf_ids:
            yield self.evaluate_leaf(
                node_id,
                parameters,
                eval_state,
            )

    def cancel_pending_and_wait(self, eval_state: EvaluationState) -> None: ...


class SerialTreeScheduler(Scheduler[ParametersT_contra], Generic[ParametersT_contra]):
    def prepare(
        self,
        objective: CombinedObjectiveFunction[ParametersT_contra],
        /,
        *,
        profile: Mapping[tuple[int, ...], float] | None = None,  # noqa: ARG002
    ) -> TreeScheduleBase[ParametersT_contra]:
        return SerialTreeSchedule(tree=cob_to_call_tree(objective))
