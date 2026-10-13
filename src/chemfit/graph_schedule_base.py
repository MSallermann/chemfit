import json
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generic, TypeAlias, TypeVar, cast

from chemfit.abstract_objective_function import (
    SKIPPED,
    EvaluateContext,
    ObjectiveComponent,
    Skipped,
    StructuralComponent,
)
from chemfit.callgraph import (
    ROOT,
    AtomicNode,
    CallGraph,
    FinishNode,
    StartNode,
    objective_to_call_graph,
)
from chemfit.scheduling import (
    EvaluationRequest,
    EvaluationResult,
    OutputT_co,
    PreparedScheduleBase,
    Scheduler,
)

InputT = TypeVar("InputT")


class BatchState(SimpleNamespace):
    """
    Hold backend-owned state for one graph-evaluation batch.

    The graph driver deliberately treats this object as opaque. Concrete
    schedules may attach queues, futures, request handles, or other temporary
    resources to it in :meth:`GraphScheduleBase.submit_tasks` and consume them
    in :meth:`GraphScheduleBase.next_completion`.
    """


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


class _Running:
    """
    Mark an atomic node that has been submitted but has not completed.

    Structural nodes execute synchronously on the coordinator and therefore
    transition directly from :data:`PENDING` to a terminal outcome.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return the sentinel's debug representation."""

        return "<Running>"


RUNNING = _Running()


NodeOutcome: TypeAlias = OutputT_co | Exception
TerminalSlot: TypeAlias = NodeOutcome | Skipped
NodeSlot: TypeAlias = TerminalSlot | _Pending | _Running


class GraphScheduleError(RuntimeError):
    """
    Report an invalid or stalled internal graph-scheduling state.

    Ordinary component exceptions are graph outcomes and do not use this
    exception type. ``GraphScheduleError`` instead signals a violated driver
    invariant, such as completing a node twice or consuming unfinished input.
    """


@dataclass
class EvaluationState:
    """
    Track coordinator-side state for one traversal of a compiled call graph.

    Every list is indexed by call-graph node ID. Keeping the state in parallel
    arrays makes readiness checks cheap and allows one prepared graph to serve
    many independent requests in a batch. Structural lifecycle methods run on
    the coordinator and mutate this state directly; backend workers only
    return atomic outcomes and serialized context result state.

    Args:
        graph: Compiled objective graph evaluated by the schedule.
        root_ctx: Context belonging directly to the root objective.

    """

    root_ctx: EvaluateContext
    contexts: list[EvaluateContext | None]
    node_slots: list[NodeSlot]
    remaining_dependencies: list[int]

    def __init__(self, graph: CallGraph, root_ctx: EvaluateContext):
        """
        Initialize every graph node as pending and attach the root context.

        Child contexts are intentionally left unallocated. Their containing
        structural start nodes create them only after their own pre-evaluation
        lifecycle succeeds, matching direct component evaluation.

        Args:
            graph: Compiled call graph shared by all evaluation runs.
            root_ctx: Caller-owned context for the root component.

        """

        self.contexts = [None] * len(graph.nodes)

        self.root_ctx = root_ctx
        self.contexts[graph.root_id] = root_ctx

        self.node_slots = [PENDING] * len(graph.nodes)
        self.graph = graph
        self.remaining_dependencies = [len(node.incoming) for node in graph.nodes]

    def assemble_node_inputs(self, node_id: int) -> tuple[Any, ...]:
        """
        Resolve a node's declared data sources to a tuple of raw values.

        Root references resolve to the request input, literal
        :class:`~chemfit.abstract_objective_function.Skipped` sources remain
        literal, and node references resolve to completed slots. No policy is
        applied here: exceptions and skipped values remain data for the
        containing structural component to inspect.

        Args:
            node_id: Node whose input sources should be resolved.

        Returns:
            Source values in the order declared by the call graph.

        Raises:
            GraphScheduleError: If a referenced node has not completed.

        """
        node = self.graph.nodes[node_id]
        inputs = []

        for source in node.inputs:
            if source is ROOT:
                inputs.append(self.root_ctx.input)
            elif isinstance(source, Skipped):
                inputs.append(source)
            else:
                assert isinstance(source, int)
                slot = self.node_slots[source]

                if slot is PENDING or slot is RUNNING:
                    msg = f"Input source {source} has not completed."
                    raise GraphScheduleError(msg)

                inputs.append(slot)

        return tuple(inputs)

    def assemble_node_input(self, node_id: int) -> Any:
        """
        Assemble input using ChemFit's single-versus-multiple value convention.

        Atomic and start nodes receive their sole source directly and multiple
        sources as a tuple. Finish nodes always receive a tuple because their
        structural reducer consumes a sequence even when only one source was
        selected.

        Args:
            node_id: Node whose input should be assembled.

        Returns:
            A single source value or a tuple of source values.

        """
        inputs = self.assemble_node_inputs(node_id)

        if isinstance(self.graph.nodes[node_id], FinishNode):
            return inputs

        return inputs[0] if len(inputs) == 1 else inputs

    def mark_node_running(self, node_id: int) -> None:
        """
        Mark a pending atomic node as submitted to a backend.

        This transition occurs before backend submission so repeated readiness
        scans cannot enqueue the same node twice.

        Args:
            node_id: Identifier of the atomic node being submitted.

        """
        self.node_slots[node_id] = RUNNING

    def complete(self, node_id: int, outcome: TerminalSlot) -> None:
        """
        Store a terminal node outcome and unlock its outgoing dependencies.

        Outcomes are deliberately not interpreted here. Values, ordinary
        exceptions, and skips all satisfy execution dependencies; the parent
        structural component decides how downstream children or its reducer
        should handle them.

        Args:
            node_id: Identifier of the node that completed.
            outcome: Value, exception, or skip produced for the node.

        Raises:
            GraphScheduleError: If the node already has a terminal outcome.

        """

        node = self.graph.nodes[node_id]

        slot = self.node_slots[node_id]
        if slot is not RUNNING and slot is not PENDING:
            msg = f"Node {node_id} is not running or pending."
            raise GraphScheduleError(msg)

        self.node_slots[node_id] = outcome

        for nid in node.outgoing:
            self.dependency_completed(node_id, nid)

    def skip(self, node_id: int, skipped: Skipped = SKIPPED) -> None:
        """
        Complete one node as skipped without executing its lifecycle.

        Args:
            node_id: Identifier of the node to skip.
            skipped: Skip marker, optionally carrying the causal exception.

        """
        self.complete(node_id, skipped)

    def skip_subtree(
        self,
        node_id: int,
        skipped: Skipped = SKIPPED,
    ) -> None:
        """
        Skip a component and every nested node without invoking lifecycle code.

        Structural scopes contain separate start and finish nodes. The start is
        completed first, descendants are recursively marked skipped, and the
        paired finish is completed last. This preserves dependency accounting
        while ensuring neither hooks nor child objective bodies run.

        Args:
            node_id: Atomic node or structural start node at the subtree root.
            skipped: Skip marker propagated to every node in the subtree.

        """
        self.skip(node_id, skipped)
        node = self.graph.nodes[node_id]

        if isinstance(node, StartNode):
            for child_id in node.contained_component_start_ids:
                self.skip_subtree(child_id, skipped)

            self.skip(node.paired_end_node_id, skipped)

    def complete_scope_early(self, node_id: int, outcome: NodeOutcome) -> None:
        """
        Finalize a structural scope short-circuited by its begin lifecycle.

        A pre-evaluation hook may return a value or the begin lifecycle may
        fail. In either case no child may run, but the structural component's
        post-evaluation lifecycle must still run exactly once, just as it does
        during a direct call. A begin exception becomes the reason attached to
        skipped descendants.

        Args:
            node_id: Structural start node whose begin lifecycle has ended.
            outcome: Short-circuit value or exception from that lifecycle.

        """

        node = self.graph.nodes[node_id]
        assert isinstance(node, StartNode)

        self.complete(node_id, outcome)

        reason = outcome if isinstance(outcome, Exception) else None
        skipped = Skipped(reason=reason)

        for nid in node.contained_component_start_ids:
            self.skip_subtree(nid, skipped)

        ctx = self.contexts[node_id]
        assert ctx is not None

        # A short circuit bypasses only the structural body. Post hooks still
        # see and may transform the pre-hook value or exception.
        try:
            final_outcome = node.objective._end_evaluation(outcome, ctx)  # noqa: SLF001
        except Exception as e:
            final_outcome = e

        # The finish node, rather than the start node, is the externally visible
        # result of a structural scope.
        self.complete(node.paired_end_node_id, final_outcome)

    def outcome(self, node_id: int) -> TerminalSlot:
        """
        Return the raw outcome of a completed node.

        Args:
            node_id: Identifier of the completed node.

        Returns:
            Value, exception, or skip produced by the node.

        Raises:
            GraphScheduleError: If the node is pending or still running.

        """

        slot = self.node_slots[node_id]

        if isinstance(slot, (_Pending, _Running)):
            msg = f"Node {node_id} has no completed outcome."
            raise GraphScheduleError(msg)

        return slot

    def dependency_completed(self, source: int, target: int) -> None:
        """
        Decrement a target's open-dependency count exactly once.

        Args:
            source: Node whose completion is being propagated.
            target: Downstream node waiting for that completion.

        Raises:
            GraphScheduleError: If completion is propagated more often than
                represented by the graph's incoming edges.

        """
        self.remaining_dependencies[target] -= 1

        if self.remaining_dependencies[target] < 0:
            msg = f"Node {target} has received a dependency completion from {source}, but it has zero open dependencies."
            raise GraphScheduleError(msg)

    def is_pending(self, node_id: int) -> bool:
        """
        Return whether a node still needs to produce an outcome.

        Args:
            node_id: Identifier of the node to inspect.

        Returns:
            True if the node is active and incomplete.

        """

        return self.node_slots[node_id] is PENDING

    def is_running(self, node_id: int) -> bool:
        """
        Return whether an atomic node is currently executing in a backend.

        Args:
            node_id: Identifier of the node to inspect.

        Returns:
            True if the node was submitted and has not completed.

        """

        return self.node_slots[node_id] is RUNNING

    def runnable_nodes(self) -> Sequence[int]:
        """
        Return pending nodes for which every execution dependency is complete.

        Readiness is based on execution edges, not on whether dependency
        outcomes were successful. Failure and skip policy is applied later by
        :meth:`~chemfit.abstract_objective_function.StructuralComponent.resolve_child_input`.

        Returns:
            Node IDs in stable call-graph order.

        """
        return [
            node_id
            for node_id, dep in enumerate(self.remaining_dependencies)
            if dep == 0 and self.is_pending(node_id)
        ]

    def write_dot_snapshot(
        self,
        directory: str | Path = "graph_states",
        prefix: str = "state",
    ) -> Path:
        """
        Write a collision-free DOT snapshot of the current evaluation state.

        Nodes are labelled with their lifecycle state, abbreviated outcome,
        and number of open dependencies. The method is intended for scheduler
        debugging and is not called during ordinary evaluation.

        Args:
            directory: Destination directory, created when necessary.
            prefix: Filename prefix used before the numeric sequence.

        Returns:
            Path of the newly created ``.dot`` file.

        """

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        def describe_slot(slot: NodeSlot) -> tuple[str, str, str]:
            if slot is PENDING:
                return "PENDING", "-", "#eeeeee"
            if slot is RUNNING:
                return "RUNNING", "-", "#fff2b3"
            if isinstance(slot, Skipped):
                reason = "-" if slot.reason is None else repr(slot.reason)
                return "SKIPPED", reason, "#cccccc"
            if isinstance(slot, Exception):
                return "FAILED", repr(slot), "#ffcccc"
            return "COMPLETED", repr(slot), "#ccffcc"

        lines = [
            "digraph EvaluationState {",
            "  rankdir=TB;",
            '  node [shape=box, style="rounded,filled", fontname="monospace"];',
            '  edge [color="#777777"];',
        ]

        for node_id, node in enumerate(self.graph.nodes):
            slot = self.node_slots[node_id]
            status, outcome, color = describe_slot(slot)

            objective = node.objective
            objective_name = getattr(
                getattr(objective, "func", None),
                "__name__",
                type(objective).__name__,
            )

            if status == "PENDING" and self.remaining_dependencies[node_id] == 0:
                status = "READY"

            # Avoid enormous labels for large arrays or dictionaries.
            if len(outcome) > 120:
                outcome = outcome[:117] + "..."

            label = (
                f"{node_id}: {type(node).__name__}\n"
                f"{objective_name}\n"
                f"slot: {status}\n"
                f"outcome: {outcome}\n"
                f"remaining deps: {self.remaining_dependencies[node_id]}"
            )

            lines.append(
                f'  n{node_id} [label={json.dumps(label)}, fillcolor="{color}"];'
            )

            lines.extend(f"  n{node_id} -> n{target};" for target in node.outgoing)

        lines.append("}")
        dot = "\n".join(lines) + "\n"

        # Exclusive creation below handles concurrent writers; scanning merely
        # chooses a useful starting number and is not relied on for correctness.
        existing = [
            int(suffix)
            for path in directory.glob(f"{prefix}_*.dot")
            if (suffix := path.stem.removeprefix(f"{prefix}_")).isdigit()
        ]
        index = max(existing, default=-1) + 1

        while True:
            path = directory / f"{prefix}_{index:04d}.dot"
            try:
                with path.open("x", encoding="utf-8") as file:
                    file.write(dot)
                return path
            except FileExistsError:
                index += 1


@dataclass
class EvaluationRun(Generic[InputT]):
    """
    Associate one batch request with its independent graph traversal.

    Runs share the immutable compiled graph held by the schedule, but never
    contexts, node slots, or dependency counters. ``yielded`` prevents a
    completed request from being returned more than once while other requests
    in the same batch continue executing.

    Args:
        index: Position of the request in the original input batch.
        state: Mutable state of this evaluation's graph traversal.

    """

    index: int
    state: EvaluationState
    yielded: bool = False


@dataclass(frozen=True)
class NodeTask(Generic[InputT]):
    """
    Describe one atomic evaluation ready for backend submission.

    Structural nodes never become ``NodeTask`` instances; their lifecycle is
    coordinator-owned. Keeping this record backend-neutral lets serial,
    executor, and MPI schedules share the graph traversal implementation.

    Args:
        run_id: Position of the associated run in the active run sequence.
        node_id: Identifier of the atomic node in the compiled call graph.
        input: Input assembled for the node.
        ctx: Context belonging to this leaf evaluation.

    """

    run_id: int
    node_id: int
    input: InputT
    ctx: EvaluateContext


@dataclass(frozen=True)
class NodeCompletion:
    """
    Report one completed atomic evaluation independently of its backend.

    A backend may execute in another thread or process. In the latter case the
    context object itself is not shared, so ``ctx_result_state`` carries the
    worker-visible mutations that the coordinator must restore before making
    downstream nodes runnable.

    Args:
        run_id: Position of the associated run in the active run sequence.
        node_id: Identifier of the completed node in the call graph.
        outcome: Numerical result or ordinary objective exception.
        ctx_result_state: Optional context state produced outside the
            coordinator process and requiring restoration before propagation.

    """

    run_id: int
    node_id: int
    outcome: NodeOutcome
    ctx_result_state: dict[str, Any] | None = None


class GraphScheduleBase(PreparedScheduleBase[InputT, OutputT_co]):
    """
    Drive a compiled objective graph while delegating atomic work to a backend.

    The base class owns all backend-independent semantics:

    - structural begin, reduce, and end lifecycles run on the coordinator;
    - child inputs are resolved by their containing structural component;
    - skipped nodes never execute hooks or objective bodies;
    - ordinary component exceptions remain node outcomes so independent work
      can continue and structural parents can recover from them; and
    - batch results are yielded as soon as each root graph completes.

    Subclasses implement only task submission, completion retrieval, and
    optional cancellation. A prepared schedule is reusable across calls, while
    each :meth:`evaluate_many` invocation receives a fresh :class:`BatchState`
    and one :class:`EvaluationState` per request.

    Args:
        graph: Immutable call graph compiled for the root objective.

    """

    def __init__(self, graph: CallGraph):
        """
        Initialize a reusable schedule for one compiled objective graph.

        Args:
            graph: Call graph whose node IDs and edges remain stable for the
                lifetime of the prepared schedule.

        """
        super().__init__()
        self.graph = graph

    def resolve_node_input(
        self,
        node_id: int,
        run: EvaluationRun,
    ) -> Any | Skipped:
        """
        Resolve the effective input for a start or atomic node.

        The graph root receives the caller's input directly. Every child node
        delegates to its structural parent, which may combine source values,
        recover from failed dependencies, or return
        :class:`~chemfit.abstract_objective_function.Skipped`. An exception
        raised by that policy is converted to a skip reason because input
        resolution happens before the child's lifecycle begins.

        Args:
            node_id: Start or atomic node whose input should be resolved.
            run: Evaluation run containing source outcomes and contexts.

        Returns:
            Input passed to the node, or a skip marker preventing execution.

        """

        node = self.graph.nodes[node_id]
        inputs = run.state.assemble_node_inputs(node_id)

        if node.parent is None:
            return inputs[0] if len(inputs) == 1 else inputs

        parent = self.graph.nodes[node.parent]
        assert isinstance(parent, StartNode)
        assert isinstance(parent.objective, StructuralComponent)
        assert node.index is not None

        parent_ctx = run.state.contexts[node.parent]
        assert parent_ctx is not None

        try:
            return parent.objective.resolve_child_input(
                node.index,
                inputs,
                parent_ctx,
            )
        except Exception as exception:
            return Skipped(reason=exception)

    def execute_start_node(self, node_id: int, run: EvaluationRun) -> None:
        """
        Execute one structural component's begin lifecycle on the coordinator.

        Successful entry creates and links one context per immediate child,
        then completes the start node with the resolved parent input. A skipped
        input bypasses the entire scope. A pre-hook result or begin exception
        short-circuits child execution while still running the component's end
        lifecycle.

        Args:
            node_id: Runnable :class:`~chemfit.callgraph.StartNode` identifier.
            run: Evaluation run being advanced.

        """
        node = cast("StartNode", self.graph.nodes[node_id])
        component = node.objective
        assert isinstance(component, StructuralComponent)

        component_input = self.resolve_node_input(node_id, run)
        if isinstance(component_input, Skipped):
            run.state.skip_subtree(node_id, component_input)
            return
        component_ctx = run.state.contexts[node_id]

        # Start and finish are two graph nodes for one logical component call;
        # sharing a context makes its pre-hooks, reducer, and post-hooks observe
        # one continuous lifecycle.
        run.state.contexts[node.paired_end_node_id] = component_ctx

        assert component_ctx is not None

        try:
            pre_eval_result = component._begin_evaluation(  # noqa: SLF001
                component_input, component_ctx
            )
        except Exception as e:
            pre_eval_result = e

        if pre_eval_result is not None:
            run.state.complete_scope_early(node_id, pre_eval_result)
            return

        # Allocate children only after begin succeeds. Consequently, an early
        # structural result has no child contexts, matching direct evaluation.
        for child_node_id in node.contained_component_start_ids:
            child_ctx = component_ctx.spawn_ctx()
            component_ctx.add_incoming(child_ctx)
            run.state.contexts[child_node_id] = child_ctx

        run.state.complete(node_id, component_input)

    def execute_finish_node(self, node_id: int, run: EvaluationRun) -> None:
        """
        Reduce a structural scope and execute its end lifecycle.

        Finish nodes become runnable only after the start and every immediate
        child finish. ``combine_inputs()`` may select fewer outcomes, but all
        children must still finish so their side effects and contexts are
        complete before the parent post-hooks run. Reducer and post-hook
        failures are stored as ordinary graph outcomes.

        Args:
            node_id: Runnable :class:`~chemfit.callgraph.FinishNode` identifier.
            run: Evaluation run being advanced.

        """
        node = cast("FinishNode", self.graph.nodes[node_id])
        component = node.objective
        assert isinstance(component, StructuralComponent)
        component_input = run.state.assemble_node_input(node_id)
        assert isinstance(component_input, tuple)
        component_ctx = run.state.contexts[node_id]
        assert component_ctx is not None

        try:
            outcome = component.combine_outcomes(component_input, component_ctx)
        except Exception as e:
            outcome = e

        # _end_evaluation() always writes the authoritative result to the
        # context before it either returns or raises. Suppression is therefore
        # safe: a post-hook failure remains available in component_ctx.output.
        with suppress(Exception):
            component._end_evaluation(outcome, component_ctx)  # noqa: SLF001

        run.state.complete(node_id, component_ctx.output)

    def advance_host_nodes(self, run: EvaluationRun) -> None:
        """
        Execute runnable structural nodes until coordinator progress stops.

        A start or finish completion can synchronously unlock another host
        node, so one readiness pass is insufficient for nested structures. The
        fixed-point loop stops only when remaining work is atomic, running, or
        blocked on a dependency.

        Args:
            run: Evaluation run whose structural nodes should advance.

        """

        while True:
            ready = [
                node_id
                for node_id in run.state.runnable_nodes()
                if isinstance(
                    self.graph.nodes[node_id],
                    (StartNode, FinishNode),
                )
            ]

            if not ready:
                break

            for node_id in ready:
                if not run.state.is_pending(node_id):
                    continue

                node = self.graph.nodes[node_id]

                if isinstance(node, StartNode):
                    self.execute_start_node(node_id, run)
                else:
                    self.execute_finish_node(node_id, run)

    def collect_ready_tasks(
        self, runs: Sequence[EvaluationRun[InputT]]
    ) -> list[NodeTask]:
        """
        Advance host work and collect every newly runnable atomic task.

        Literal skips and parent input-policy decisions are resolved on the
        coordinator. Because skipping an atomic node can immediately unlock
        more graph work, each run is drained to a fixed point before the method
        returns tasks to the backend.

        Args:
            runs: Active evaluation runs in the current batch.

        Returns:
            Atomic tasks ready to submit, ordered first by run and then by
            stable call-graph order.

        """
        tasks: list[NodeTask] = []

        for run in runs:
            while True:
                # A skip can immediately unlock a structural finish node or
                # another atomic child. Drain that synchronous progress before
                # waiting for a backend completion.
                self.advance_host_nodes(run)
                skipped_any = False

                for node_id in run.state.runnable_nodes():
                    node = self.graph.nodes[node_id]
                    if not isinstance(node, AtomicNode):
                        continue

                    task_input = self.resolve_node_input(node_id, run)
                    if isinstance(task_input, Skipped):
                        run.state.skip(node_id, task_input)
                        skipped_any = True
                        continue

                    task_ctx = run.state.contexts[node_id]
                    assert task_ctx is not None
                    # Reserve the node before handing the task to a backend;
                    # the next scan must not submit it again.
                    run.state.mark_node_running(node_id)
                    tasks.append(
                        NodeTask(
                            run_id=run.index,
                            node_id=node_id,
                            input=task_input,
                            ctx=task_ctx,
                        )
                    )

                if not skipped_any:
                    break
        return tasks

    def result_if_complete(self, run: EvaluationRun) -> EvaluationResult | None:
        """
        Return a root result once, or ``None`` while the run is incomplete.

        Args:
            run: Evaluation run to inspect.

        Returns:
            Indexed result when the graph's result node has a terminal slot;
            otherwise ``None``. Subsequent calls after emission also return
            ``None``.

        """
        if run.yielded:
            return None

        state = run.state
        result_id = self.graph.result_id

        if state.is_pending(result_id) or state.is_running(result_id):
            return None

        run.yielded = True
        return EvaluationResult(
            index=run.index,
            value=state.outcome(result_id),
        )

    def submit_tasks(
        self,
        tasks: Sequence[NodeTask],
        batch_state: BatchState,
    ) -> None:
        """
        Submit atomic tasks without waiting for them to complete.

        Backend implementations may attach any required handles to
        ``batch_state``. Every submitted task must eventually be returned by
        :meth:`next_completion` unless the batch is aborted.

        Args:
            tasks: Newly runnable tasks, already marked as running.
            batch_state: Mutable backend state scoped to this batch.

        """
        raise NotImplementedError

    def next_completion(
        self,
        batch_state: BatchState,
    ) -> NodeCompletion:
        """
        Wait for and return one previously submitted atomic completion.

        Args:
            batch_state: Mutable backend state scoped to this batch.

        Returns:
            Completion identifying both the request and call-graph node.

        """
        raise NotImplementedError

    def abort_batch(self, batch_state: BatchState) -> None:
        """
        Attempt to cancel or discard outstanding backend work after failure.

        The default implementation is a no-op for synchronous backends.
        Implementations should treat cleanup as best effort: the original
        catastrophic exception remains primary if cleanup also fails.

        Args:
            batch_state: State belonging to the failed batch.

        """

    def evaluate_many(  # noqa: PLR0912, PLR0915
        self, requests: Sequence[EvaluationRequest[InputT]]
    ) -> Iterator[EvaluationResult]:
        """
        Evaluate a batch by interleaving graph progress across all requests.

        Results are yielded in completion order and retain their original
        request index. Ordinary objective exceptions are yielded as result
        values for the public scheduling layer to handle. Failures of the
        scheduler or backend itself abort outstanding work, close the prepared
        schedule when required, and are re-raised.

        Args:
            requests: Inputs and caller-owned root contexts to evaluate.

        Yields:
            Indexed results as soon as their root result nodes complete.

        Raises:
            RuntimeError: If the prepared schedule has already been closed.
            GraphScheduleError: If unfinished graphs have neither runnable nor
                in-flight work.

        """
        if self.closed:
            msg = "Prepared schedule is closed."
            raise RuntimeError(msg)
        batch_state = BatchState()
        runs: list[EvaluationRun[InputT]] = []

        try:
            # Each request needs independent slots and dependency counts even
            # though all runs share the schedule's immutable compiled graph.
            for idx, req in enumerate(requests):
                req.ctx.input = req.parameters
                eval_state = EvaluationState(
                    self.graph,
                    root_ctx=req.ctx,
                )
                runs.append(EvaluationRun(index=idx, state=eval_state))

            # Enter root structural scopes before submitting atomic work. A
            # pre-hook may complete an entire request without using a backend.
            for run in runs:
                self.advance_host_nodes(run)

                result = self.result_if_complete(run)
                if result is not None:
                    yield result

            tasks_in_flight = 0
            while not all(run.yielded for run in runs):
                # Batch all work exposed by the latest host-side progress. The
                # backend remains free to complete it in any order.
                tasks = self.collect_ready_tasks(runs)

                if tasks:
                    self.submit_tasks(tasks, batch_state)
                    tasks_in_flight += len(tasks)

                for run in runs:
                    result = self.result_if_complete(run)
                    if result is not None:
                        yield result

                if all(run.yielded for run in runs):
                    break

                # Consume one completion at a time so it can immediately expose
                # downstream tasks instead of waiting for an entire wave.
                if tasks_in_flight:
                    completion = self.next_completion(batch_state)
                    tasks_in_flight -= 1

                    run_id = completion.run_id
                    run = runs[run_id]
                    node_id = completion.node_id

                    ctx = runs[run_id].state.contexts[node_id]
                    assert ctx is not None

                    if completion.ctx_result_state is not None:
                        ctx_result_state = completion.ctx_result_state
                        ctx.apply_result_state(ctx_result_state)

                    # Restore worker-side context mutations before structural
                    # parents inspect child contexts or run post-hooks.
                    run.state.complete(completion.node_id, completion.outcome)

                    # One atomic result may close several nested scopes.
                    self.advance_host_nodes(run)

                    result = self.result_if_complete(run)
                    if result is not None:
                        yield result
                else:
                    # A final host-only sweep distinguishes synchronous progress
                    # from a malformed or cyclic graph with no possible work.
                    completed = 0
                    for run in runs:
                        self.advance_host_nodes(run)

                        result = self.result_if_complete(run)
                        if result is not None:
                            completed += 1
                            yield result
                    if completed == 0 and not all(run.yielded for run in runs):
                        msg = "Graph evaluation stalled without runnable nodes."
                        raise GraphScheduleError(msg)
        except BaseException as exception:
            try:
                self.abort_batch(batch_state)
            except BaseException as cleanup_exception:
                if hasattr(exception, "add_note"):
                    exception.add_note(
                        f"Batch cancellation failed: {cleanup_exception!r}"
                    )
            self._close_after_catastrophic_failure(exception)
            raise


class SerialGraphSchedule(GraphScheduleBase):
    """
    Execute a compiled call graph synchronously in the calling process.

    Tasks still pass through the same submission/completion interface used by
    concurrent backends. This keeps serial execution a reference implementation
    of graph semantics rather than a separate recursive execution path.
    """

    def __init__(self, graph: CallGraph):
        """
        Initialize a serial schedule for a compiled graph.

        Args:
            graph: Call graph to execute synchronously.

        """
        super().__init__(graph)

    def submit_tasks(
        self,
        tasks: Sequence[NodeTask],
        batch_state: BatchState,
    ) -> None:
        """
        Append ready tasks to the current batch's in-process FIFO queue.

        Actual component execution is deferred to :meth:`next_completion`,
        preserving the same driver ordering used by asynchronous schedules.

        Args:
            tasks: Atomic tasks exposed by graph traversal.
            batch_state: Batch-local state holding the FIFO queue.

        """
        if not hasattr(batch_state, "tasks"):
            batch_state.tasks = []

        batch_state.tasks.extend(tasks)

    def next_completion(
        self,
        batch_state: BatchState,
    ) -> NodeCompletion:
        """
        Execute and return the oldest queued task in the calling process.

        Ordinary component exceptions are captured as outcomes so structural
        parents and independent batch runs can continue. No serialized context
        state is needed because execution mutates the coordinator's context
        object directly.

        Args:
            batch_state: Batch-local state containing submitted tasks.

        Returns:
            Completion for the task at the front of the FIFO queue.

        """

        task = batch_state.tasks.pop(0)
        ctx = task.ctx
        node = self.graph.nodes[task.node_id]

        try:
            outcome = node.objective(task.input, ctx)
        except Exception as e:
            outcome = e

        return NodeCompletion(
            run_id=task.run_id,
            node_id=task.node_id,
            outcome=outcome,
            ctx_result_state=None,
        )


class SerialGraphScheduler(Scheduler[SerialGraphSchedule]):
    """Prepare synchronous graph schedules for objective components."""

    def prepare(
        self,
        objective: ObjectiveComponent[InputT, OutputT_co],
        /,
    ) -> SerialGraphSchedule:
        """
        Compile an objective component into a serial graph schedule.

        Args:
            objective: Root ordinary or composite objective to schedule.

        Returns:
            Prepared serial schedule for the objective graph.

        """

        return SerialGraphSchedule(graph=objective_to_call_graph(objective))
