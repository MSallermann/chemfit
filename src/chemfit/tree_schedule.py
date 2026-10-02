"""
Tree-based prepared schedules for nested combined objectives.

The module compiles a combined-objective hierarchy into a CallTree and tracks
each evaluation in a separate EvaluationState. A top-down pass creates
contexts and begins objective lifecycles, backend-specific code evaluates the
reachable leaves, and completion events propagate bottom-up until the root
produces an EvaluationResult.

TreeScheduleBase implements the backend-independent lifecycle and propagation
logic. Concrete schedules only need to execute leaves and quiesce outstanding
work when a batch aborts.
"""

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
        None cannot represent this state because it denotes an omitted term in
        ChemFit.

    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return the sentinel's debug representation."""

        return "<PENDING>"


PENDING = _Pending()


class _Inactive:
    """
    Sentinel for inactive evaluations.

    This is for nodes that can never be reached due to an exception in a parent
    setup phase.

    Note:
        None cannot represent this state because it denotes an omitted term in
        ChemFit.

    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return the sentinel's debug representation."""

        return "<INACTIVE>"


INACTIVE = _Inactive()


# A TermResult is the value consumed by CombinedObjectiveFunction._reduce_terms.
# None is valid here because an exception handler may choose to omit a term.
TermResult = float | None

# A node itself always produces either a numerical value or an Exception.
# None is therefore deliberately not part of NodeOutcome.
NodeOutcome = float | Exception

# Per-node execution state. Nodes begin INACTIVE, become PENDING when they are
# reachable for this run, and are finally replaced by their raw NodeOutcome.
NodeSlot = NodeOutcome | _Pending | _Inactive

# Propagation either reaches the root and produces its outcome, or the run is
# still waiting for other nodes to complete.
PropagationResult = NodeOutcome | _Pending
SetupOutcome = PropagationResult


@dataclass
class EvaluationState:
    """
    Mutable tree state for one objective evaluation.

    Args:
        tree: Compiled objective tree evaluated by the schedule.
        root_ctx: Context belonging to the root combined objective.

    Attributes:
        contexts: Context associated with each tree node. Contexts for
            unreachable nodes remain None.
        node_slots: Current execution state or completed raw outcome for every
            node.
        remaining_children: Number of incomplete children for every combine
            node.
        open_nodes: Combine nodes whose evaluation lifecycle has begun but has
            not ended.

    """

    contexts: list[EvaluateContext | None]
    node_slots: list[NodeSlot]
    remaining_children: list[int]
    open_nodes: set[NodeId]

    def __init__(self, tree: CallTree, root_ctx: EvaluateContext):
        """Initialize all nodes as inactive and register the root context."""

        self.contexts = [None] * len(tree.nodes)
        self.contexts[tree.root] = root_ctx
        self.node_slots = [INACTIVE] * len(
            tree.nodes
        )  # all nodes start out as INACTIVE
        self.remaining_children = [
            len(node.children) if isinstance(node, CombineNode) else 0
            for node in tree.nodes
        ]
        self.open_nodes = set()

    def activate(self, node_id: NodeId) -> None:
        """
        Mark a node as participating in this evaluation.

        Args:
            node_id: Identifier of the node to mark pending.

        Raises:
            RuntimeError: If the node is already active or complete.

        """

        if self.node_slots[node_id] is not INACTIVE:
            msg = f"Node {node_id} is already active."
            raise RuntimeError(msg)

        self.node_slots[node_id] = PENDING

    def complete(self, node_id: NodeId, outcome: NodeOutcome) -> None:
        """
        Store the raw outcome of a pending node.

        Args:
            node_id: Identifier of the node that completed.
            outcome: Numerical value or exception produced by the node.

        Raises:
            RuntimeError: If the node is not pending.

        """

        if self.node_slots[node_id] is not PENDING:
            msg = f"Node {node_id} is not pending."
            raise RuntimeError(msg)

        self.node_slots[node_id] = outcome

    def outcome(self, node_id: NodeId) -> NodeOutcome:
        """
        Return the raw outcome of a completed node.

        Args:
            node_id: Identifier of the completed node.

        Returns:
            Numerical value or exception produced by the node.

        Raises:
            RuntimeError: If the node is inactive or still pending.

        """

        slot = self.node_slots[node_id]

        if isinstance(slot, (_Pending, _Inactive)):
            msg = f"Node {node_id} has no completed outcome."
            raise RuntimeError(msg)

        return slot

    def child_completed(self, node_id: NodeId) -> bool:
        """
        Record one child completion for a combine node.

        Args:
            node_id: Identifier of the parent combine node.

        Returns:
            True when every child of the node has completed.

        Raises:
            RuntimeError: If more completions are recorded than the node has
                children.

        """

        self.remaining_children[node_id] -= 1

        if self.remaining_children[node_id] < 0:
            msg = f"Combine node {node_id} received too many completions."
            raise RuntimeError(msg)

        return self.remaining_children[node_id] == 0

    def is_pending(self, node_id: NodeId) -> bool:
        """
        Return whether a node still needs to produce an outcome.

        Args:
            node_id: Identifier of the node to inspect.

        Returns:
            True if the node is active and incomplete.

        """

        return self.node_slots[node_id] is PENDING


@dataclass
class EvaluationRun(Generic[ParametersT_co]):
    """
    Associate one batch request with its parameters and tree state.

    Args:
        index: Position of the request in the original input batch.
        parameters: Parameter mapping for the evaluation.
        state: Mutable state of this evaluation's tree traversal.

    """

    index: int
    parameters: ParametersT_co
    state: EvaluationState


class TreeScheduleBase(
    PreparedScheduleBase[ParametersT_contra],
    Generic[ParametersT_contra],
):
    """
    Backend-independent schedule for evaluating a compiled objective tree.

    The base class owns context construction, objective lifecycles, result
    propagation, and batch cleanup. Subclasses provide the mechanism for
    evaluating leaf nodes and cancelling outstanding backend work.

    Args:
        tree: Compiled call tree rooted at the combined objective represented
            by this schedule.

    """

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
        """
        Prepare one evaluation by traversing its combine nodes top-down.

        The traversal creates child contexts, begins each reachable combined
        objective lifecycle, and marks reachable leaves as pending.

        Args:
            parameters: Parameter mapping for the evaluation.
            root_ctx: Context belonging to the root combined objective.
            eval_state: Mutable tree state initialized for this evaluation.

        Returns:
            PENDING when leaf evaluation is required, or the root outcome when
            a setup failure completes the run early.

        Notes:
            Ordinary Exceptions raised during setup are treated as node
            outcomes and propagated through parent exception handlers.
            BaseException subclasses escape so evaluate_many can abort every
            lifecycle opened for the batch.

        """

        def propagate_setup_failure(
            node_id: NodeId,
            exception: Exception,
        ) -> SetupOutcome:
            node = self.tree.nodes[node_id]

            # A failure of the root is already the final outcome of this run.
            if node.parent_id is None:
                eval_state.complete(node_id, exception)
                return exception

            # Nested setup failures are raw outcomes of the failed combine node.
            # The immediate parent COB owns interpretation of that exception,
            # exactly as it would for an exception raised during normal evaluation.
            return self.propagate_completion(
                node_id,
                exception,
                eval_state,
            )

        def visit(
            combine_ctx: EvaluateContext,
            node_id: NodeId,
        ) -> SetupOutcome:
            node = self.tree.nodes[node_id]
            assert isinstance(node, CombineNode)
            cob = node.objective

            # Start this COBs lifecycle
            # ... once evaluation begins, _end_evaluation() must be called exactly once.
            eval_state.activate(node_id)
            eval_state.open_nodes.add(node_id)

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

                    if not isinstance(root_outcome, _Pending):
                        return root_outcome

                else:
                    # ... for LeafNodes we simply change the state to PENDING
                    eval_state.activate(child_id)

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
        finished are closed bottom-up. Cleanup failures are attached as notes
        to the original exception so they do not replace the primary failure.

        Args:
            exception: Failure that caused the evaluation to abort.
            eval_state: State of the incomplete evaluation.

        Notes:
            No leaf evaluation belonging to this run may still be executing
            when this method is called.

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
        node_id: NodeId,
        eval_state: EvaluationState,
    ) -> float:
        """
        Finish a combine node after all of its children complete.

        Child outcomes are converted into weighted terms, exceptions are
        offered to the combined objective's handler, metadata is collected,
        and the terms are reduced. The node's evaluation lifecycle is ended
        whether reduction succeeds or fails.

        Args:
            node_id: Identifier of the combine node to finish.
            eval_state: Evaluation state containing all child outcomes.

        Returns:
            Reduced numerical value produced by the combine node.

        Raises:
            BaseException: If term handling, reduction, metadata collection,
                or lifecycle finalization fails.

        """

        # NOTE: this overall function needs to mimic the semantics of the later part of
        # ObjectiveFunctor.__call__ (including the end of life cycle invocations)
        # the first part (the beginning of the life cycle) is already handled in the
        # top_down pass

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
                    # All children must have completed before this combine node is
                    # finished. EvaluationState owns the node-state checks here so
                    # this function only deals with actual raw outcomes.
                    outcome = eval_state.outcome(child_id)

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
    ) -> NodeOutcome:
        """
        Evaluate one leaf and return its raw outcome.

        Args:
            node_id: Identifier of the pending leaf node.
            parameters: Parameter mapping for the evaluation.
            eval_state: Evaluation state containing the leaf context.

        Returns:
            The objective value on success, or the raised Exception as a raw
            outcome so the parent combine node can apply its exception handler.

        Notes:
            This method deliberately applies neither the parent weight nor the
            parent exception handler.

        """

        # We should only invoke this on pending leaves
        assert eval_state.is_pending(node_id)

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
        outcome: NodeOutcome,
        eval_state: EvaluationState,
    ) -> PropagationResult:
        """
        Record a completed node outcome and propagate completion towards the root.

        Each node produces a raw outcome: either its objective value or an exception.
        When all children of a combine node have completed, that combine node is
        finished and its own raw outcome is propagated further upwards.

        Args:
            node_id: Identifier of the node that completed.
            outcome: Raw numerical value or exception produced by the node.
            eval_state: Mutable state of the corresponding evaluation.

        Returns:
            The root outcome when this completion finishes the run, otherwise
            PENDING while another child remains incomplete.

        Raises:
            RuntimeError: If completion violates the evaluation state's node
                or child-count invariants.

        """

        outcome_to_propagate: NodeOutcome = outcome
        current_node_id: NodeId = node_id

        while True:
            node = self.tree.nodes[current_node_id]

            # A node may only complete once.
            # Store the raw outcome of this node. The parent combine node will
            # interpret it later by applying its weight / exception handler.
            eval_state.complete(
                current_node_id,
                outcome_to_propagate,
            )

            # propagate_completion() is only called for nodes that have a parent.
            # The root itself is finished internally below and is never propagated.
            parent_id = node.parent_id
            assert parent_id is not None

            # One more child of the parent has completed.
            # The parent cannot be finished until all of its children have completed.
            if not eval_state.child_completed(parent_id):
                return PENDING

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
                eval_state.complete(
                    parent_id,
                    outcome_to_propagate,
                )
                return outcome_to_propagate

            # Otherwise the completed combine node behaves exactly like any other
            # completed child. Store its outcome on the next iteration and continue
            # propagating towards the root.
            current_node_id = parent_id

    def evaluate_leaves(
        self,
        runs: Sequence[EvaluationRun[ParametersT_contra]],
    ) -> Iterator[tuple[int, NodeId, NodeOutcome]]:
        """
        Evaluate pending leaves for a batch of prepared runs.

        Args:
            runs: Successfully prepared evaluation runs.

        Yields:
            Tuples containing the position in runs, completed node identifier,
            and raw node outcome. Events may be yielded in completion order.

        Notes:
            Subclasses must implement this method using their execution
            backend.

        """
        raise NotImplementedError

    def cancel_pending_and_wait(
        self,
        runs: Sequence[EvaluationRun[ParametersT_contra]],
    ) -> None:
        """
        Cancel pending leaf work and wait until the backend is quiescent.

        Args:
            runs: Evaluation runs whose outstanding work must be stopped.

        Notes:
            This method is called before open objective lifecycles are aborted,
            so it must not return while a leaf can still mutate result state.

        """

        raise NotImplementedError

    def evaluate_many(
        self,
        requests: Sequence[EvaluationRequest[ParametersT_contra]],
    ) -> Iterator[EvaluationResult]:
        """
        Evaluate a batch through top-down setup and bottom-up propagation.

        Args:
            requests: Evaluation requests to prepare and execute.

        Yields:
            One indexed result for every request. Setup failures are yielded
            before leaf completions; successful runs may otherwise complete in
            backend-defined order.

        Raises:
            BaseException: If setup or backend execution cannot continue for
                the batch. Outstanding leaf work is quiesced and open
                lifecycles are aborted before the exception is re-raised.

        """

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

                if not isinstance(root_outcome, _Pending):
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
    """
    Tree schedule that evaluates every active leaf synchronously.

    Args:
        tree: Compiled call tree to evaluate.

    """

    def evaluate_leaves(
        self,
        runs: Sequence[EvaluationRun[ParametersT]],
    ) -> Iterator[tuple[int, NodeId, NodeOutcome]]:
        """
        Evaluate pending leaves serially in run and tree order.

        Args:
            runs: Successfully prepared evaluation runs.

        Yields:
            Tuples containing the position in runs, completed leaf identifier,
            and raw leaf outcome.

        """

        for run_idx, run in enumerate(runs):
            for node_id in self.leaf_ids:
                # only submit pending leaves
                if run.state.is_pending(node_id):
                    yield (
                        run_idx,
                        node_id,
                        self.evaluate_leaf(
                            node_id,
                            run.parameters,
                            run.state,
                        ),
                    )

    def cancel_pending_and_wait(
        self,
        runs: Sequence[EvaluationRun[ParametersT]],
    ) -> None:
        """
        Complete the no-op cancellation step for serial execution.

        Args:
            runs: Evaluation runs being aborted. Serial execution has no
                outstanding work by the time this method is called.

        """


class SerialTreeScheduler(Scheduler[SerialTreeSchedule[Any]]):
    """Prepare synchronous tree schedules for combined objectives."""

    def prepare(
        self,
        objective: CombinedObjectiveFunction[ParametersT_contra],
        /,
        *,
        profile: Mapping[tuple[int, ...], float] | None = None,  # noqa: ARG002
    ) -> SerialTreeSchedule[ParametersT_contra]:
        """
        Compile a combined objective into a serial tree schedule.

        Args:
            objective: Root combined objective whose complete nested call tree
                should be scheduled.
            profile: Optional cost profile. Serial scheduling does not use
                placement costs, so this argument is ignored.

        Returns:
            Prepared serial schedule for the objective tree.

        """

        return SerialTreeSchedule(tree=cob_to_call_tree(objective))
