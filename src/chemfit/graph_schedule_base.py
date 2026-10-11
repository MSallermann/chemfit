from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Generic, TypeVar

from chemfit.abstract_objective_function import EvaluateContext, ObjectiveFunctor
from chemfit.callgraph import (
    CallGraph,
    StartNode,
    AtomicNode,
    FinishNode,
    ROOT,
    objective_to_call_graph,
)
from chemfit.scheduling import (
    EvaluationRequest,
    EvaluationResult,
    PreparedScheduleBase,
    Scheduler,
    OutputT_co,
    InputT_contra,
)
from typing import cast


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

NodeOutcome = OutputT_co | Exception

# Per-node execution state. Nodes begin INACTIVE, become PENDING when they are
# reachable for this run, and are finally replaced by their raw NodeOutcome.
NodeSlot = NodeOutcome | _Pending | _Inactive

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
        tree: Compiled objective tree evaluated by the schedule.
        root_ctx: Context belonging directly to the root objective.

    """

    contexts: list[EvaluateContext | None]
    node_slots: list[NodeSlot]
    remaining_children: list[int]

    def __init__(self, graph: CallGraph, root_ctx: EvaluateContext):
        """Initialize all nodes as inactive and register the root context."""

        self.contexts = [None] * len(graph.nodes)

        self.root_ctx = root_ctx
        self.contexts[0] = root_ctx

        self.node_slots = [INACTIVE] * len(
            graph.nodes
        )  # all nodes start out as INACTIVE

        self.graph = graph

        # a node can only run if this reaches zero
        self.remaining_dependencies = [len(node.incoming) for node in graph.nodes]

    def activate(self, node_id: int) -> None:
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
                if slot is PENDING or slot is INACTIVE:
                    msg = "Tried to construct input from PENDING or INACTIVE node"
                    raise GraphException(msg)
                node_input[idx] = cast("NodeOutcome", slot)

        if len(node_input) == 1:
            return node_input

        return tuple(node_input)

    def complete(self, node_id: int, outcome: NodeOutcome) -> None:
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

        if isinstance(slot, (_Pending, _Inactive)):
            msg = f"Node {node_id} has no completed outcome."
            raise RuntimeError(msg)

        return slot

    def child_completed(self, node_id: int) -> bool:
        """
        Record one child completion for a composite node.

        Args:
            node_id: Identifier of the parent composite node.

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

    def is_pending(self, node_id: int) -> bool:
        """
        Return whether a node still needs to produce an outcome.

        Args:
            node_id: Identifier of the node to inspect.

        Returns:
            True if the node is active and incomplete.

        """

        return self.node_slots[node_id] is PENDING
