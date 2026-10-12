from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Generic, cast

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
    InputT_contra,
    OutputT_co,
    PreparedScheduleBase,
    Scheduler,
)


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

        return "<Running>"


RUNNING = _Running()


# @dataclass(frozen=True)
# class Success:
#     value: Any


# @dataclass(frozen=True)
# class Failure:
#     exception: Exception


# NodeOutcome = Success | Failure

NodeOutcome = OutputT_co | Exception

# Per-node execution state. Nodes begin INACTIVE, become PENDING when they are
# reachable for this run, and are finally replaced by their raw NodeOutcome.
NodeSlot = NodeOutcome | _Pending | _Running

# Propagation either reaches the root and produces its outcome, or the run is
# still waiting for other nodes to complete.
PropagationResult = NodeOutcome

SetupOutcome = PropagationResult


class GraphException(BaseException): ...


@dataclass
class EvaluationState:
    """
    Mutable tree state for one objective evaluation.

    Args:
        graph: Compiled objective graph evaluated by the schedule.
        root_ctx: Context belonging directly to the root objective.

    """

    root_ctx: EvaluateContext
    contexts: list[EvaluateContext | None]
    node_slots: list[NodeSlot]
    remaining_dependencies: list[int]

    def __init__(self, graph: CallGraph, root_ctx: EvaluateContext):
        """Initialize all nodes as inactive and register the root context."""

        self.contexts = [None] * len(graph.nodes)

        self.root_ctx = root_ctx
        self.contexts[graph.root_id] = root_ctx

        self.node_slots = [PENDING] * len(graph.nodes)  # all nodes start out as PENDING

        self.graph = graph

        # a node can only run if this reaches zero
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
                    raise GraphException(msg)
                node_input[idx] = cast("NodeOutcome", slot)

        # FinishNodes always receive a tuple
        if isinstance(node, FinishNode):
            return tuple(node_input)

        # Other nodes receive one-element tuples in unpacked form
        return node_input[0] if len(node_input) == 1 else tuple(node_input)

    def mark_node_running(self, node_id: int):
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
            raise GraphException(msg)

        self.node_slots[node_id] = outcome

        for nid in node.outgoing:
            self.dependency_completed(node_id, nid)

    def skip(self, node_id: int, exception: Exception) -> None:
        """Complete a node without executing its lifecycle."""
        self.complete(node_id, exception)

    def abort_scope(self, node_id: int, exception: Exception):
        self.complete(node_id, exception)

        node = self.graph.nodes[node_id]
        assert isinstance(node, StartNode)

        def skip_children_recursively(component_id: int, exception: Exception):
            component_node = self.graph.nodes[component_id]
            assert isinstance(component_node, StartNode)
            for child_id in component_node.contained_component_start_ids:
                child_node = self.graph.nodes[child_id]
                self.skip(child_id, exception)
                if isinstance(child_node, StartNode):
                    self.skip(child_node.paired_end_node_id, exception)
                    skip_children_recursively(child_id, exception)

        skip_children_recursively(node_id, exception)

        ctx = self.contexts[node_id]
        assert ctx is not None
        node.objective._end_evaluation(exception, ctx)  # noqa: SLF001

        end_node = node.paired_end_node_id
        self.complete(end_node, exception)

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
            raise GraphException(msg)

        return slot

    def dependency_completed(self, source: int, target: int) -> bool:
        """
        Record one dependency completion for a composite node.

        Args:
            node_id: Identifier of the parent composite node.

        Returns:
            True when every child of the node has completed.

        Raises:
            RuntimeError: If more completions are recorded than the node has
                children.

        """

        self.remaining_dependencies[target] -= 1

        if self.remaining_dependencies[target] < 0:
            msg = f"Node {target} has received a dependency completion from {source}, but it has zero open dependencies."
            raise GraphException(msg)

        return self.remaining_dependencies[target] == 0

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

    def count_pending(self) -> int:
        return sum(slot is PENDING for slot in self.node_slots)


@dataclass
class EvaluationRun(Generic[InputT_contra]):
    """
    Associate one batch request with its root input and graph state.

    Args:
        index: Position of the request in the original input batch.
        root_input: Parameter mapping for the evaluation.
        state: Mutable state of this evaluation's tree traversal.

    """

    index: int
    root_input: InputT_contra
    state: EvaluationState
    yielded: bool = False


@dataclass(frozen=True)
class NodeTask(Generic[InputT_contra]):
    """
    Describe one pending leaf evaluation independently of its backend.

    Args:
        run_id: Position of the associated run in the active run sequence.
        node_id: Identifier of the leaf in the compiled call tree.
        parameters: Parameter mapping for the associated evaluation run.
        ctx: Context belonging to this leaf evaluation.

    """

    run_id: int
    node_id: int
    input: InputT_contra
    ctx: EvaluateContext


@dataclass(frozen=True)
class NodeCompletion:
    """
    Report one completed node evaluation independently of its backend.

    Args:
        run_id: Position of the associated run in the active run sequence.
        node_id: Identifier of the completed leaf in the call tree.
        outcome: Numerical result or ordinary objective exception.
        ctx_result_state: Optional context state produced outside the
            coordinator process and requiring restoration before propagation.

    """

    run_id: int
    node_id: int
    outcome: NodeOutcome
    ctx_result_state: dict[str, Any] | None = None


class GraphScheduleBase(PreparedScheduleBase[InputT_contra, OutputT_co]):
    def __init__(self, graph: CallGraph):
        super().__init__()
        self.graph = graph

    def execute_start_node(self, node_id: int, run: EvaluationRun):
        node = cast("StartNode", self.graph.nodes[node_id])
        component = node.objective
        assert isinstance(component, StructuralComponent)
        component_input = run.state.assemble_node_input(node_id)
        component_ctx = run.state.contexts[node_id]

        # a start node and its end node need to share the same ctx
        run.state.contexts[node.paired_end_node_id] = component_ctx

        assert component_ctx is not None

        try:
            # start lifecycle
            component._begin_evaluation(component_input, component_ctx)  # noqa: SLF001

            # create component contexts
            for child_node_id in node.contained_component_start_ids:
                child_ctx = component_ctx.spawn_ctx()
                component_ctx.add_incoming(child_ctx)
                run.state.contexts[child_node_id] = child_ctx
        except Exception as e:
            run.state.abort_scope(node_id, e)

        run.state.complete(node_id, component_input)

    def execute_finish_node(self, node_id: int, run: EvaluationRun):
        node = cast("FinishNode", self.graph.nodes[node_id])
        component = node.objective
        assert isinstance(component, StructuralComponent)
        component_input = run.state.assemble_node_input(node_id)
        component_ctx = run.state.contexts[node_id]
        assert component_ctx is not None

        try:
            outcome = component.combine_outcomes(component_input, component_ctx)
        except Exception as e:
            outcome = e

        component._end_evaluation(outcome, component_ctx)  # noqa: SLF001
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
                node = self.graph.nodes[node_id]

                if isinstance(node, StartNode):
                    self.execute_start_node(node_id, run)
                else:
                    self.execute_finish_node(node_id, run)

    def collect_ready_tasks(
        self, runs: Sequence[EvaluationRun[InputT_contra]]
    ) -> list[NodeTask]:

        tasks: list[NodeTask] = []

        for run in runs:
            # list of node ids with zero pending dependencies
            runnable_nodes = run.state.runnable_nodes()

            for node_id in runnable_nodes:
                node = self.graph.nodes[node_id]

                if isinstance(node, AtomicNode):

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
                        run.state.skip(node_id, exception)
                        self.advance_host_nodes(run)
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

    def evaluate_many(self, requests: Sequence[EvaluationRequest[InputT_contra]]):

        batch_state = BatchState()
        runs: list[EvaluationRun[InputT_contra]] = []

        # 1. Build the list of runs
        for idx, req in enumerate(requests):
            req.ctx.input = req.parameters
            eval_state = EvaluationState(
                self.graph,
                root_ctx=req.ctx,
            )
            runs.append(
                EvaluationRun(index=idx, root_input=req.parameters, state=eval_state)
            )

        # 2. Initialize structural components.
        for run in runs:
            self.advance_host_nodes(run)

            result = self.result_if_complete(run)
            if result is not None:
                yield result

        # 3. Process worker completions
        while not all(run.yielded for run in runs):

            # Submit newly available work
            tasks = self.collect_ready_tasks(runs)

            if tasks:
                self.submit_tasks(tasks, batch_state)

            # Wait for exactly one completion.
            completion = self.next_completion(batch_state)

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


class SerialGraphSchedule(GraphScheduleBase):
    def __init__(self, graph: CallGraph):
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
        input = task.input
        ctx = task.ctx
        node = self.graph.nodes[task.node_id]

        try:
            outcome = node.objective(input, ctx)
        except Exception as e:
            outcome = e

        return NodeCompletion(
            run_id=task.run_id,
            node_id=task.node_id,
            outcome=outcome,
            ctx_result_state=ctx.to_result_state(),
        )


class SerialGraphScheduler(Scheduler[SerialGraphSchedule]):
    """Prepare synchronous tree schedules for objective functors."""

    def prepare(
        self,
        objective: ObjectiveComponent[InputT_contra, OutputT_co],
        /,
    ) -> SerialGraphSchedule:
        """
        Compile an objective functor into a serial tree schedule.

        Args:
            objective: Root ordinary or composite objective to schedule.

        Returns:
            Prepared serial schedule for the objective tree.

        """

        return SerialGraphSchedule(graph=objective_to_call_graph(objective))
