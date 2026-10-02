from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from chemfit.abstract_objective_function import EvaluateContext
from chemfit.callgraph import (
    CallTree,
    CombineNode,
    LeafNode,
    NodeId,
    cob_to_call_tree,
)
from chemfit.combined_objective_function import CombinedObjectiveFunction
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


class _Inactive:
    """
    Sentinel for inactive evaluations.

    This is for leaf nodes that can never be reached due to an exception in a parent setup phase.

    Note:
    We cannot use None since that has a special meaning in ChemFit already.

    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<INACTIVE>"


INACTIVE = _Inactive()

TermResult = float | None
TermOutcome = TermResult | Exception
TermSlot = TermOutcome | _Pending | _Inactive
SetupOutcome = float | Exception | _Pending


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
        self.term_results = [INACTIVE] * len(
            tree.nodes
        )  # all nodes start out as INACTIVE
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
    ) -> SetupOutcome:
        """Construct the context tree, begin nested COB lifecycles and change leaf term_slots to PENDING."""

        def propagate_setup_failure(
            node_id: NodeId,
            exception: Exception,
        ) -> SetupOutcome:
            node = self.tree.nodes[node_id]

            # A failure of the root is already the final outcome of this run.
            if node.parent_id is None:
                eval_state.term_results[node_id] = exception
                return exception

            root_outcome = self.propagate_completion(
                node_id,
                exception,
                eval_state,
            )

            if root_outcome is None:
                return PENDING

            return root_outcome

        def visit(
            combine_ctx: EvaluateContext,
            node_id: NodeId,
        ) -> SetupOutcome:
            node = self.tree.nodes[node_id]
            assert isinstance(node, CombineNode)
            cob = node.objective

            # Start this COBs lifecycle
            # ... once evaluation begins, _end_evaluation() must be called exactly once.
            eval_state.open_nodes.add(node_id)
            # mark combine node pending
            eval_state.term_results[node_id] = PENDING

            try:
                cob._begin_evaluation(  # noqa: SLF001
                    parameters=parameters,
                    ctx=combine_ctx,
                )

                child_contexts = combine_ctx.spawn_children(
                    cob.n_terms(),
                    cob.child_context_configurator,
                )
            except Exception as e:
                # an exception in `_begin_evaluation` or `spawn_children`,
                # ends the current evaluation, and the child_nodes stay marked
                # as INACTIVE
                try:
                    cob._end_evaluation(combine_ctx, e)  # noqa: SLF001
                finally:
                    eval_state.open_nodes.discard(node_id)

                # setup failure has to be propagated up the tree
                return propagate_setup_failure(
                    node_id,
                    e,
                )

            # BaseException deliberately escapes with this node still in open_nodes.
            # evaluate_many() will then abort the incomplete evaluation.
            for child_id, child_ctx in zip(
                node.children,
                child_contexts,
                strict=True,
            ):
                eval_state.contexts[child_id] = child_ctx

                child_node = self.tree.nodes[child_id]

                # now we iterate over the children
                # ... if a child is a CombineNode, we recurse
                if isinstance(child_node, CombineNode):
                    root_outcome = visit(
                        combine_ctx=child_ctx,
                        node_id=child_id,
                    )

                    if root_outcome is not PENDING:
                        return root_outcome
                else:
                    # ... for LeafNodes we simply change the state to PENDING
                    eval_state.term_results[child_id] = PENDING

            return PENDING

        return visit(
            combine_ctx=root_ctx,
            node_id=self.tree.root,
        )

    def abort_evaluation(
        self,
        exception: BaseException,
        eval_state: EvaluationState,
    ) -> None:
        """
        Abort an incomplete evaluation.

        All combine nodes whose evaluation lifecycle has begun but has not yet
        finished are closed bottom-up. This is cleanup for evaluations that cannot
        complete through normal outcome propagation.

        This method assumes that no leaf evaluation belonging to this run is still
        executing.
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

    def finish_combine_node(
        self,
        node_id: int,
        eval_state: EvaluationState,
    ) -> float:
        """
        Finish a combine node, after all its terms have been computed.

        This includes the collection of the child meta-data,
        the reduction of the individual terms and the end of the evaluation lifecycle.
        """

        # NOTE: this overall function needs to mimic the semantics of the later part of
        # ObjectiveFunctor.__call__ (including the end of life cycle invocations)
        # the first part (the beginning of the life cycle) is already handled in the top_down pass

        # tree state bookkeeping
        node = self.tree.nodes[node_id]
        ctx = eval_state.contexts[node_id]

        assert ctx is not None
        assert isinstance(node, CombineNode)

        cob = node.objective

        try:
            # this sub-stage is similar to CombinedObjectiveFunction._evaluate
            # ... it iterates over the child outcomes
            # ... successful outcomes are converted into weighted terms
            # ... exceptions are passed to this COB's exception handler
            # ... since the ctx.child_context context manager is mimicked here
            # ... the finally block needs to call `collect_child_meta_data`

            terms: list[TermResult] = []

            try:
                for idx, child_id in enumerate(node.children):
                    outcome = eval_state.term_results[child_id]

                    # None of the child outcomes may still be pending
                    assert outcome is not PENDING

                    child_ctx = eval_state.contexts[child_id]
                    assert child_ctx is not None

                    if isinstance(outcome, Exception):
                        # If the child evaluation failed, this COB's exception handler
                        # gets a chance to convert the exception into a valid term.
                        term = cob.exception_handler(
                            outcome,
                            child_ctx,
                            idx,
                        )
                    else:
                        try:
                            # Successful child results are converted into the weighted
                            # term contributed to this COB.
                            term = outcome * cob.weights[idx]
                        except Exception as e:
                            # Applying the weight is part of evaluating the term too,
                            # so failures here follow the same exception-handler semantics.
                            term = cob.exception_handler(
                                e,
                                child_ctx,
                                idx,
                            )

                    terms.append(term)

            finally:
                # Mimic cleanup performed by the child-context context manager.
                ctx.collect_child_meta_data(recursive=False)

            # Once all child outcomes have been converted into terms,
            # the COB can perform its reduction.
            value = cob._reduce_terms(terms, ctx)  # noqa: SLF001
            ctx.loss = value

        except BaseException as e:
            # Any exception that escapes the term handling or reduction means
            # this combine node itself failed.
            cob._end_evaluation(ctx, e)  # noqa: SLF001
            raise

        else:
            # Successful reduction completes this COB's evaluation lifecycle.
            cob._end_evaluation(ctx, None)  # noqa: SLF001

        finally:
            # The lifecycle has been consumed whether it ended successfully or failed.
            eval_state.open_nodes.discard(node_id)

        return value

    def evaluate_leaf(
        self,
        node_id: NodeId,
        parameters: ParametersT_contra,
        eval_state: EvaluationState,
    ) -> TermOutcome:
        """
        Evaluate one leaf and return its raw outcome.

        A successful leaf returns its objective value. If the leaf evaluation raises
        an Exception, the exception itself is returned so that the parent combine
        node can apply its own exception-handling semantics.
        """

        # We should only invoke this on pending leaves
        assert eval_state.term_results[node_id] is PENDING

        # tree state bookkeeping
        node = self.tree.nodes[node_id]
        ctx = eval_state.contexts[node_id]

        assert isinstance(node, LeafNode)
        assert ctx is not None

        # A leaf is evaluated normally, including its own ObjectiveFunctor lifecycle.
        # We deliberately do not apply a weight or invoke the parent's exception
        # handler here: interpreting the outcome of a child belongs to its parent COB.
        try:
            return node.objective(
                parameters,
                ctx,
            )
        except Exception as e:
            # Store evaluation failures as outcomes rather than allowing them to
            # escape the backend. The parent combine node will decide how to handle
            # the exception when all of its child outcomes are available.
            return e

    def propagate_completion(
        self,
        node_id: NodeId,
        outcome: TermOutcome,
        eval_state: EvaluationState,
    ) -> float | Exception | None:
        """
        Record a completed node outcome and propagate completion towards the root.

        Each node produces a raw outcome: either its objective value or an exception.
        When all children of a combine node have completed, that combine node is
        finished and its own raw outcome is propagated further upwards.

        Returns the root outcome when this completion finishes the entire run.
        Otherwise returns None.
        """

        outcome_to_propagate: TermOutcome = outcome
        current_node_id: NodeId = node_id

        while True:
            node = self.tree.nodes[current_node_id]

            # A node may only complete once.
            if eval_state.term_results[current_node_id] is not PENDING:
                msg = f"Node {current_node_id} has already completed."
                raise RuntimeError(msg)

            # Store the raw outcome of this node. The parent combine node will
            # interpret it later by applying its weight / exception handler.
            eval_state.term_results[current_node_id] = outcome_to_propagate

            # propagate_completion() is only called for nodes that have a parent.
            # The root itself is finished internally below and is never propagated.
            parent_id = node.parent_id
            assert parent_id is not None

            # One more child of the parent has completed.
            eval_state.remaining_children[parent_id] -= 1

            if eval_state.remaining_children[parent_id] < 0:
                msg = f"Combine node {parent_id} received too many completions."
                raise RuntimeError(msg)

            # The parent cannot be finished until all of its children have completed.
            if eval_state.remaining_children[parent_id] > 0:
                return None

            parent = self.tree.nodes[parent_id]
            assert isinstance(parent, CombineNode)

            # All children of the parent are now complete. Finishing the combine
            # node interprets those child outcomes, reduces its terms, and completes
            # its ObjectiveFunctor lifecycle.
            try:
                outcome_to_propagate = self.finish_combine_node(
                    parent_id,
                    eval_state,
                )
            except Exception as e:
                # A combine node can itself fail, for example because its exception
                # handler, reducer, or post-evaluation hook raises. That failure is
                # simply the raw outcome propagated to its own parent.
                outcome_to_propagate = e

            # The root has no parent, so its raw outcome is the final outcome
            # of this evaluation run.
            if parent.parent_id is None:
                eval_state.term_results[parent_id] = outcome_to_propagate
                return outcome_to_propagate

            # Otherwise the completed combine node behaves exactly like any other
            # completed child. Store its outcome on the next iteration and continue
            # propagating towards the root.
            current_node_id = parent_id

    def evaluate_leaves(
        self,
        runs: Sequence[EvaluationRun[ParametersT_contra]],
    ) -> Iterator[tuple[int, NodeId, TermOutcome]]:
        """Evaluate leaves and yield completion events. The meaning of the tuple is [run_id, node_id, result]."""
        raise NotImplementedError

    def cancel_pending_and_wait(self, runs: Sequence[EvaluationRun]) -> None:
        """Cancel pending leaves and wait for other leaves to complete."""
        raise NotImplementedError

    def evaluate_many(
        self,
        requests: Sequence[EvaluationRequest[ParametersT_contra]],
    ) -> Iterator[EvaluationResult]:
        # 1. Perform the top-down pass for every request.
        #
        # Ordinary setup failures belong only to the corresponding evaluation.
        # No leaf work has been submitted at this point, so failed setup only
        # requires closing the COB lifecycles opened for that evaluation.

        runs: list[EvaluationRun[ParametersT_contra]] = []
        setup_results: list[EvaluationResult] = []
        setup_states: list[EvaluationState] = []

        try:
            for idx, req in enumerate(requests):
                eval_state = EvaluationState(
                    self.tree,
                    root_ctx=req.ctx,
                )
                setup_states.append(eval_state)

                setup_outcome = self.top_down_pass(
                    req.parameters,
                    root_ctx=req.ctx,
                    eval_state=eval_state,
                )

                if isinstance(setup_outcome, _Pending):
                    runs.append(
                        EvaluationRun(
                            index=idx,
                            parameters=req.parameters,
                            state=eval_state,
                        )
                    )
                else:
                    setup_results.append(
                        EvaluationResult(
                            index=idx,
                            value=setup_outcome,
                        )
                    )

        except BaseException as e:
            # No leaves have been submitted yet.
            for eval_state in setup_states:
                self.abort_evaluation(e, eval_state)

            raise

        # 2. Yield setup failures and evaluate all successfully prepared runs.
        #
        # From here on, a BaseException means evaluation cannot continue through
        # the normal outcome-propagation machinery. Any outstanding leaf work must
        # therefore be stopped before the remaining open lifecycles are aborted.
        try:
            yield from setup_results

            for run_id, node_id, outcome in self.evaluate_leaves(runs):
                run = runs[run_id]

                # Propagate this completed node through its evaluation tree.
                # A returned outcome means that the root has completed.
                root_outcome = self.propagate_completion(
                    node_id,
                    outcome,
                    run.state,
                )

                if root_outcome is not None:
                    yield EvaluationResult(
                        index=run.index,
                        value=root_outcome,
                    )

        except BaseException as e:
            self.cancel_pending_and_wait(runs)

            for run in runs:
                self.abort_evaluation(e, run.state)

            raise


class SerialTreeSchedule(TreeScheduleBase[ParametersT], Generic[ParametersT]):
    def evaluate_leaves(
        self, runs: Sequence[EvaluationRun[ParametersT]]
    ) -> Iterator[tuple[int, NodeId, TermOutcome]]:
        for run_idx, run in enumerate(runs):
            for node_id in self.leaf_ids:
                # only submit pending leaves
                if run.state.term_results[node_id] is PENDING:
                    yield (
                        run_idx,
                        node_id,
                        self.evaluate_leaf(node_id, run.parameters, run.state),
                    )

    def cancel_pending_and_wait(self, runs: Sequence[EvaluationRun]) -> None: ...


class SerialTreeScheduler(Scheduler[SerialTreeSchedule[Any]]):
    def prepare(
        self,
        objective: CombinedObjectiveFunction[ParametersT_contra],
        /,
        *,
        profile: Mapping[tuple[int, ...], float] | None = None,  # noqa: ARG002
    ) -> SerialTreeSchedule[ParametersT_contra]:
        return SerialTreeSchedule(tree=cob_to_call_tree(objective))
