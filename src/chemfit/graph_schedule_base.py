from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Generic, TypeVar, cast

from chemfit.abstract_objective_function import (
    EvaluateContext,
    ObjectiveComponent,
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
    """Backend scratch state for one evaluate_many() call."""


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
    """Sentinel for atomic evaluations submitted to a backend."""

    __slots__ = ()

    def __repr__(self) -> str:
        """Return the sentinel's debug representation."""

        return "<Running>"


RUNNING = _Running()


NodeOutcome = OutputT_co | Exception

# Nodes begin pending, may become running, and are finally replaced by outcomes.
NodeSlot = NodeOutcome | _Pending | _Running


class GraphScheduleError(RuntimeError):
    """Report an invalid internal graph-scheduling state."""


@dataclass
class EvaluationState:
    """
    Mutable graph state for one objective evaluation.

    Args:
        graph: Compiled objective graph evaluated by the schedule.
        root_ctx: Context belonging directly to the root objective.

    """

    root_ctx: EvaluateContext
    contexts: list[EvaluateContext | None]
    node_slots: list[NodeSlot]
    remaining_dependencies: list[int]

    def __init__(self, graph: CallGraph, root_ctx: EvaluateContext):
        """Initialize all nodes as pending and register the root context."""

        self.contexts = [None] * len(graph.nodes)

        self.root_ctx = root_ctx
        self.contexts[graph.root_id] = root_ctx

        self.node_slots = [PENDING] * len(graph.nodes)
        self.graph = graph
        self.remaining_dependencies = [len(node.incoming) for node in graph.nodes]

    def assemble_node_input(self, node_id: int) -> tuple[Any, ...] | NodeOutcome:
        """Collect the inputs for this node_id."""
        node = self.graph.nodes[node_id]
        input_ids = node.inputs

        node_input: list[NodeOutcome] = [None] * len(input_ids)

        for idx, input_id in enumerate(input_ids):
            if input_id is ROOT:
                node_input[idx] = self.root_ctx.input
            else:
                slot = self.node_slots[cast("int", input_id)]
                if slot is PENDING or slot is RUNNING:
                    msg = (
                        "Tried to construct input from PENDING or already RUNNING node"
                    )
                    raise GraphScheduleError(msg)
                node_input[idx] = cast("NodeOutcome", slot)

        # FinishNodes always receive a tuple
        if isinstance(node, FinishNode):
            return tuple(node_input)

        # Other nodes receive one-element tuples in unpacked form
        return node_input[0] if len(node_input) == 1 else tuple(node_input)

    def mark_node_running(self, node_id: int) -> None:
        self.node_slots[node_id] = RUNNING

    def complete(self, node_id: int, outcome: NodeOutcome) -> None:
        """
        Store the raw outcome of a pending node and propagate its completion to its dependencies.

        Args:
            node_id: Identifier of the node that completed.
            outcome: Numerical value or exception produced by the node.

        Raises:
            RuntimeError: If the node is not pending.

        """

        node = self.graph.nodes[node_id]

        slot = self.node_slots[node_id]
        if slot is not RUNNING and slot is not PENDING:
            msg = f"Node {node_id} is not running or pending."
            raise GraphScheduleError(msg)

        self.node_slots[node_id] = outcome

        for nid in node.outgoing:
            self.dependency_completed(node_id, nid)

    def skip(self, node_id: int, exception: Exception) -> None:
        """Complete a node without executing its lifecycle."""
        self.complete(node_id, exception)

    def skip_subtree(self, node_id: int, exception: Exception) -> None:
        """Skip a component and all its descendants without running any lifecycle methods."""
        node = self.graph.nodes[node_id]

        self.skip(node_id, exception)

        if isinstance(node, StartNode):
            for child_id in node.contained_component_start_ids:
                self.skip_subtree(child_id, exception)

            self.skip(node.paired_end_node_id, exception)

    def abort_scope(self, node_id: int, exception: Exception) -> None:
        """Abort a structural component whose begin lifecycle failed."""
        node = self.graph.nodes[node_id]
        assert isinstance(node, StartNode)

        self.complete(node_id, exception)

        for child_id in node.contained_component_start_ids:
            self.skip_subtree(child_id, exception)

        ctx = self.contexts[node_id]
        assert ctx is not None

        # End hooks still run, but the begin failure remains authoritative.
        with suppress(Exception):
            node.objective._end_evaluation(exception, ctx)  # noqa: SLF001

        self.complete(node.paired_end_node_id, exception)

    def outcome(self, node_id: int) -> NodeOutcome:
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

        if isinstance(slot, (_Pending, _Running)):
            msg = f"Node {node_id} has no completed outcome."
            raise GraphScheduleError(msg)

        return slot

    def dependency_completed(self, source: int, target: int) -> None:
        """Record one dependency completion and reject duplicate propagation."""
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
        Return whether a node still needs to produce an outcome.

        Args:
            node_id: Identifier of the node to inspect.

        Returns:
            True if the node is active and incomplete.

        """

        return self.node_slots[node_id] is RUNNING

    def runnable_nodes(self) -> Sequence[int]:
        return [
            node_id
            for node_id, dep in enumerate(self.remaining_dependencies)
            if dep == 0 and self.is_pending(node_id)
        ]


@dataclass
class EvaluationRun(Generic[InputT]):
    """
    Associate one batch request with its root input and graph state.

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
    Describe one pending leaf evaluation independently of its backend.

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
    Report one completed node evaluation independently of its backend.

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
    def __init__(self, graph: CallGraph):
        """Initialize a reusable schedule for one compiled graph."""
        super().__init__()
        self.graph = graph

    def execute_start_node(self, node_id: int, run: EvaluationRun) -> None:
        node = cast("StartNode", self.graph.nodes[node_id])
        component = node.objective
        assert isinstance(component, StructuralComponent)

        # Check if we skip this node
        exception: Exception | None = None
        for source in node.inputs:
            if source is ROOT:
                continue

            outcome = run.state.outcome(cast("int", source))
            if isinstance(outcome, Exception):
                exception = outcome
                break

        if exception is not None:
            run.state.skip_subtree(node_id, exception)
            return

        component_input = run.state.assemble_node_input(node_id)
        component_ctx = run.state.contexts[node_id]

        # a start node and its end node need to share the same ctx
        run.state.contexts[node.paired_end_node_id] = component_ctx

        assert component_ctx is not None

        try:
            # start lifecycle
            component._begin_evaluation(component_input, component_ctx)  # noqa: SLF001
        except Exception as e:
            run.state.abort_scope(node_id, e)
            return

        # create component contexts
        for child_node_id in node.contained_component_start_ids:
            child_ctx = component_ctx.spawn_ctx()
            component_ctx.add_incoming(child_ctx)
            run.state.contexts[child_node_id] = child_ctx

        run.state.complete(node_id, component_input)

    def execute_finish_node(self, node_id: int, run: EvaluationRun) -> None:
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

        try:
            component._end_evaluation(outcome, component_ctx)  # noqa: SLF001
        except Exception as e:
            outcome = e

        run.state.complete(node_id, outcome)

    def advance_host_nodes(self, run: EvaluationRun) -> None:
        """Execute all currently runnable host nodes."""

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
        tasks: list[NodeTask] = []

        for run in runs:
            runnable_nodes = run.state.runnable_nodes()

            for node_id in runnable_nodes:
                node = self.graph.nodes[node_id]

                if isinstance(node, AtomicNode):
                    exception: Exception | None = None
                    for source in node.inputs:
                        if source is ROOT:
                            continue

                        outcome = run.state.outcome(cast("int", source))
                        if isinstance(outcome, Exception):
                            exception = outcome
                            break

                    if exception is not None:
                        run.state.skip(node_id, exception)
                        continue

                    task_input = run.state.assemble_node_input(node_id)

                    task_ctx = run.state.contexts[node_id]
                    assert task_ctx is not None
                    run.state.mark_node_running(node_id)
                    tasks.append(
                        NodeTask(
                            run_id=run.index,
                            node_id=node_id,
                            input=task_input,
                            ctx=task_ctx,
                        )
                    )
        return tasks

    def result_if_complete(self, run: EvaluationRun) -> EvaluationResult | None:
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
        """Submit tasks for execution without waiting for completion."""
        raise NotImplementedError

    def next_completion(
        self,
        batch_state: BatchState,
    ) -> NodeCompletion:
        """Block until one submitted task completes."""
        raise NotImplementedError

    def abort_batch(self, batch_state: BatchState) -> None:
        """Best effort cancellation of pending tasks."""

    def evaluate_many(  # noqa: PLR0912, PLR0915
        self, requests: Sequence[EvaluationRequest[InputT]]
    ) -> Iterator[EvaluationResult]:
        if self.closed:
            msg = "Prepared schedule is closed."
            raise RuntimeError(msg)
        batch_state = BatchState()
        runs: list[EvaluationRun[InputT]] = []

        try:
            # 1. Build the list of runs
            for idx, req in enumerate(requests):
                req.ctx.input = req.parameters
                eval_state = EvaluationState(
                    self.graph,
                    root_ctx=req.ctx,
                )
                runs.append(EvaluationRun(index=idx, state=eval_state))

            # 2. Initialize structural components
            for run in runs:
                self.advance_host_nodes(run)

                result = self.result_if_complete(run)
                if result is not None:
                    yield result

            # 3. Process worker completions
            tasks_in_flight = 0
            while not all(run.yielded for run in runs):
                # Submit newly available work
                tasks = self.collect_ready_tasks(runs)

                if tasks:
                    self.submit_tasks(tasks, batch_state)
                    tasks_in_flight += len(tasks)

                # Wait for exactly one completion.
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

                    # Record atomic completion and unlock downstream nodes
                    run.state.complete(completion.node_id, completion.outcome)

                    # Propagate through structural nodes immediately.
                    self.advance_host_nodes(run)

                    # Emit the result immediately if this evaluation completed.
                    result = self.result_if_complete(run)
                    if result is not None:
                        yield result
                else:
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
    def __init__(self, graph: CallGraph):
        """Initialize a serial graph schedule."""
        super().__init__(graph)

    def submit_tasks(
        self,
        tasks: Sequence[NodeTask],
        batch_state: BatchState,
    ) -> None:
        """Submit tasks for execution without waiting for completion."""
        if not hasattr(batch_state, "tasks"):
            batch_state.tasks = []

        batch_state.tasks.extend(tasks)

    def next_completion(
        self,
        batch_state: BatchState,
    ) -> NodeCompletion:
        """Return a completed task."""

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
