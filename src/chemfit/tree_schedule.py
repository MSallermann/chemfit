"""
Tree-based prepared schedules for objective functors.

The module compiles an ordinary or combined objective into a CallTree and
tracks each evaluation in a separate EvaluationState. A top-down pass creates
contexts and begins combined-objective lifecycles, backend-specific code
evaluates the reachable leaves, and completion events propagate bottom-up
until the root produces an EvaluationResult.

TreeScheduleBase implements the backend-independent lifecycle and propagation
logic. Concrete schedules only need to execute backend-neutral LeafTask values,
return LeafCompletion values, and clean up their resources when closed.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from chemfit.abstract_objective_function import EvaluateContext, ObjectiveFunctor
from chemfit.callgraph import (
    CallTree,
    CombineNode,
    LeafNode,
    NodeId,
    objective_to_call_tree,
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
        root_ctx: Context belonging directly to the root objective.

    Attributes:
        contexts: Context associated with each tree node. Contexts for
            unreachable nodes remain None.
        node_slots: Current execution state or completed raw outcome for every
            node.
        remaining_children: Number of incomplete children for every combine
            node.

    """

    contexts: list[EvaluateContext | None]
    node_slots: list[NodeSlot]
    remaining_children: list[int]

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


@dataclass(frozen=True)
class LeafTask(Generic[ParametersT_co]):
    """
    Describe one pending leaf evaluation independently of its backend.

    Args:
        run_id: Position of the associated run in the active run sequence.
        node_id: Identifier of the leaf in the compiled call tree.
        parameters: Parameter mapping for the associated evaluation run.
        ctx: Context belonging to this leaf evaluation.

    """

    run_id: int
    node_id: NodeId
    parameters: ParametersT_co
    ctx: EvaluateContext


@dataclass(frozen=True)
class LeafCompletion:
    """
    Report one completed leaf evaluation independently of its backend.

    Args:
        run_id: Position of the associated run in the active run sequence.
        node_id: Identifier of the completed leaf in the call tree.
        outcome: Numerical result or ordinary objective exception.
        ctx_result_state: Optional context state produced outside the
            coordinator process and requiring restoration before propagation.

    """

    run_id: int
    node_id: NodeId
    outcome: NodeOutcome
    ctx_result_state: dict[str, Any] | None = None


def evaluate_leaf_task(
    objective: ObjectiveFunctor[ParametersT],
    task: LeafTask[ParametersT],
    *,
    capture_context_state: bool,
) -> LeafCompletion:
    """Evaluate one task and convert ordinary exceptions into node outcomes."""

    try:
        outcome: NodeOutcome = objective(task.parameters, task.ctx)
    except Exception as exception:
        outcome = exception

    ctx_result_state = task.ctx.to_result_state() if capture_context_state else None
    return LeafCompletion(
        run_id=task.run_id,
        node_id=task.node_id,
        outcome=outcome,
        ctx_result_state=ctx_result_state,
    )


class TreeScheduleBase(
    PreparedScheduleBase[ParametersT_contra],
    Generic[ParametersT_contra],
):
    """
    Backend-independent schedule for evaluating a compiled objective tree.

    The base class owns context construction, objective lifecycles, result
    propagation, and catastrophic-failure handling. Subclasses only provide
    the mechanism for executing backend-neutral leaf tasks.

    Args:
        tree: Compiled ordinary or combined objective call tree represented by
            this schedule.

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
        Prepare an evaluation by activating its root and traversing top-down.

        An ordinary root leaf is marked pending with ``root_ctx`` directly.
        For a combined root, the traversal creates child contexts, begins each
        reachable combined-objective lifecycle, and marks its reachable leaves
        as pending.

        Args:
            parameters: Parameter mapping for the evaluation.
            root_ctx: Context belonging directly to the root objective.
            eval_state: Mutable tree state initialized for this evaluation.

        Returns:
            PENDING when leaf evaluation is required, or the root outcome when
            a setup failure completes the run early.

        Notes:
            Ordinary Exceptions raised during setup are treated as node
            outcomes and propagated through parent exception handlers.
            BaseException subclasses escape so evaluate_many can poison and
            close the schedule. Partially completed lifecycle state is then
            undefined.

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
                cob._end_evaluation(combine_ctx, e)  # noqa: SLF001

                # spawn_children() installs the complete child batch before it
                # invokes configurators. A configurator may therefore fail
                # after creating partially configured, inactive contexts.
                # Serial evaluation exposes those contexts when the parent
                # recursively collects metadata. Tree schedules collect
                # completed nodes non-recursively, so materialize this failed
                # nested node's inactive children here to preserve the serial
                # semantics. A failed root has no parent collection step and
                # deliberately keeps the same unmaterialized state as serial.
                if node.parent_id is not None:
                    combine_ctx.collect_child_meta_data(recursive=True)

                # setup failure has to be propagated up the tree
                return propagate_setup_failure(
                    node_id,
                    e,
                )

            # BaseException deliberately escapes. The prepared schedule is then
            # poisoned and closed; partial lifecycle state is undefined.
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

        root = self.tree.nodes[self.tree.root]
        if isinstance(root, LeafNode):
            eval_state.activate(root.id)
            return PENDING

        return visit(combine_ctx=root_ctx, node_id=root.id)

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

        return value

    def expand_leaf_tasks(
        self,
        runs: Sequence[EvaluationRun[ParametersT_contra]],
    ) -> list[LeafTask[ParametersT_contra]]:
        """Expand every pending leaf in a run sequence into a backend task."""

        tasks: list[LeafTask[ParametersT_contra]] = []
        for run_id, run in enumerate(runs):
            for node_id in self.leaf_ids:
                if not run.state.is_pending(node_id):
                    continue

                ctx = run.state.contexts[node_id]
                assert ctx is not None
                tasks.append(
                    LeafTask(
                        run_id=run_id,
                        node_id=node_id,
                        parameters=run.parameters,
                        ctx=ctx,
                    )
                )

        return tasks

    def restore_leaf_completion(
        self,
        completion: LeafCompletion,
        runs: Sequence[EvaluationRun[ParametersT_contra]],
    ) -> tuple[int, NodeId, NodeOutcome]:
        """Restore worker context state and return a propagation event."""

        run = runs[completion.run_id]
        ctx = run.state.contexts[completion.node_id]
        assert ctx is not None

        if completion.ctx_result_state is not None:
            ctx.apply_result_state(completion.ctx_result_state)

        return completion.run_id, completion.node_id, completion.outcome

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

            parent_id = node.parent_id
            if parent_id is None:
                return outcome_to_propagate

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

    def execute_leaf_tasks(
        self,
        tasks: Sequence[LeafTask[ParametersT_contra]],
    ) -> Iterator[LeafCompletion]:
        """
        Execute backend-neutral leaf tasks.

        Args:
            tasks: Pending leaf evaluations prepared by TreeScheduleBase.

        Yields:
            Backend-neutral completions in any order.

        Notes:
            This is the complete backend-specific execution contract. Backends
            do not perform tree discovery, context restoration, propagation,
            reduction, exception handling, or lifecycle transitions.

        """
        raise NotImplementedError

    def evaluate_leaves(
        self,
        runs: Sequence[EvaluationRun[ParametersT_contra]],
    ) -> Iterator[tuple[int, NodeId, NodeOutcome]]:
        """
        Execute all pending leaves and restore their returned context state.

        Args:
            runs: Successfully prepared evaluation runs.

        Yields:
            Existing tree-propagation events containing run ID, node ID, and
            raw node outcome.

        """

        tasks = self.expand_leaf_tasks(runs)
        for completion in self.execute_leaf_tasks(tasks):
            yield self.restore_leaf_completion(completion, runs)

    def _close_after_catastrophic_failure(self, exception: BaseException) -> None:
        """Poison this schedule without replacing the original failure."""

        try:
            self.close()
        except BaseException as cleanup_exception:
            exception.add_note(f"Schedule cleanup failed: {cleanup_exception!r}")
        finally:
            # A backend close implementation may itself fail before delegating
            # to PreparedScheduleBase.close(). Catastrophic failure still makes
            # this prepared schedule permanently unusable.
            if not self.closed:
                super().close()

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
            BaseException: If setup or backend execution fails
                catastrophically. The prepared schedule is closed and cannot
                be reused; partial evaluation state is undefined.

        """
        try:
            if self.closed:
                msg = "Prepared schedule is closed."
                raise RuntimeError(msg)

            runs: list[EvaluationRun[ParametersT_contra]] = []
            setup_results: list[EvaluationResult] = []

            for idx, req in enumerate(requests):
                eval_state = EvaluationState(
                    self.tree,
                    root_ctx=req.ctx,
                )

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

        except BaseException as exception:
            self._close_after_catastrophic_failure(exception)
            raise


class SerialTreeSchedule(TreeScheduleBase[ParametersT], Generic[ParametersT]):
    """
    Tree schedule that evaluates every active leaf synchronously.

    Args:
        tree: Compiled call tree to evaluate.

    """

    def execute_leaf_tasks(
        self,
        tasks: Sequence[LeafTask[ParametersT]],
    ) -> Iterator[LeafCompletion]:
        """
        Evaluate leaf tasks serially in input order.

        Args:
            tasks: Backend-neutral pending leaf tasks.

        Yields:
            One completion per task in input order.

        """

        for task in tasks:
            node = self.tree.nodes[task.node_id]
            assert isinstance(node, LeafNode)
            yield evaluate_leaf_task(
                node.objective,
                task,
                capture_context_state=False,
            )


class SerialTreeScheduler(Scheduler[SerialTreeSchedule[Any]]):
    """Prepare synchronous tree schedules for objective functors."""

    def prepare(
        self,
        objective: ObjectiveFunctor[ParametersT_contra],
        /,
        *,
        profile: Mapping[tuple[int, ...], float] | None = None,  # noqa: ARG002
    ) -> SerialTreeSchedule[ParametersT_contra]:
        """
        Compile an objective functor into a serial tree schedule.

        Args:
            objective: Root ordinary or combined objective to schedule.
            profile: Optional cost profile. Serial scheduling does not use
                placement costs, so this argument is ignored.

        Returns:
            Prepared serial schedule for the objective tree.

        """

        return SerialTreeSchedule(tree=objective_to_call_tree(objective))
