# ruff: noqa: SLF001

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, cast

from chemfit.abstract_objective_function import (
    PARENT_INPUT,
    InputSource,
    InputT_contra,
    ObjectiveComponent,
    OutputT_co,
    ParentInput_,
    Skipped,
    StructuralComponent,
)
from chemfit.evaluate_context import EvaluateContext


class GraphComponent(StructuralComponent[InputT_contra, OutputT_co]):
    """
    A scheduler-visible directed acyclic graph of objective components.

    Graph components are normally created with :class:`GraphBuilder`. Their
    topology is an immutable snapshot: later changes to the builder do not
    affect an already-created component.
    """

    def __init__(
        self,
        components: Sequence[ObjectiveComponent[Any, Any]],
        component_dependencies: tuple[tuple[int, ...], ...],
        component_inputs: tuple[tuple[InputSource, ...], ...],
        combine_inputs: tuple[InputSource, ...],
    ) -> None:
        """
        Store an immutable snapshot of a builder's graph topology.

        Args:
            components: Child components in topological order.
            component_dependencies: Prerequisite child indices for each child.
            component_inputs: Ordered input sources for each child.
            combine_inputs: Sources exposed as the graph result.

        """
        super().__init__()
        self._components = tuple(components)
        self._component_dependencies = component_dependencies
        self._component_inputs = component_inputs
        self._combine_inputs = combine_inputs

    def components(self) -> Sequence[ObjectiveComponent[Any, Any]]:
        """Return the graph's child components in topological order."""
        return self._components

    def component_dependencies(self) -> tuple[tuple[int, ...], ...]:
        """Return execution dependencies for every child component."""
        return self._component_dependencies

    def component_inputs(self) -> tuple[tuple[InputSource, ...], ...]:
        """Return the ordered data sources consumed by every child."""
        return self._component_inputs

    def combine_inputs(self) -> tuple[InputSource, ...]:
        """Return the sources selected as graph outputs."""
        return self._combine_inputs

    def combine_outcomes(
        self,
        outcomes: tuple[Any, ...],
        ctx: EvaluateContext,  # noqa: ARG002
    ) -> OutputT_co:
        """
        Return one selected output directly or multiple outputs as a tuple.

        A failed or skipped sole output cannot produce a value, so its original
        exception is raised when available. Multiple outputs preserve exception
        and skip outcomes as tuple entries.
        """
        if len(outcomes) != 1:
            return cast("OutputT_co", outcomes)

        outcome = outcomes[0]

        if isinstance(outcome, Skipped):
            if outcome.reason is not None:
                raise outcome.reason
            msg = "Graph output was skipped."
            raise RuntimeError(msg)

        if isinstance(outcome, Exception):
            raise outcome

        return cast("OutputT_co", outcome)


@dataclass(frozen=True, eq=False)
class NodeRef(Generic[OutputT_co]):
    """Opaque reference to one node's output within a particular builder."""

    _component: ObjectiveComponent[Any, OutputT_co]
    _node_inputs: list[InputSource]
    _node_dependencies: list[int]
    _index: int


class GraphBuilder:
    """
    Incrementally construct a scheduler-visible directed acyclic graph.

    Add nodes in topological order with :meth:`call`, optionally add ordering
    constraints with :meth:`depends_on`, and finish with :meth:`output`.
    References belong to the builder that created them and cannot be reused in
    another graph.
    """

    def __init__(self) -> None:
        """Initialize an empty graph."""
        self._nodes: list[NodeRef[Any]] = []

    @property
    def input(self) -> ParentInput_:
        """Return a reference to the input received by the completed graph."""
        return PARENT_INPUT

    def _validate_ref(self, ref: NodeRef[Any]) -> None:
        if not 0 <= ref._index < len(self._nodes) or self._nodes[ref._index] is not ref:
            msg = "NodeRef belongs to another graph."
            raise ValueError(msg)

    def depends_on(self, a: NodeRef[Any], *others: NodeRef[Any]) -> None:
        """
        Make ``a`` wait for earlier nodes without passing their results.

        Args:
            a: Node whose execution should be delayed.
            *others: Earlier nodes that must finish before ``a`` may run.

        Raises:
            ValueError: If a reference is foreign, invalid, or does not precede
                ``a`` in construction order.

        """
        self._validate_ref(a)

        for nref in others:
            self._validate_ref(nref)

            if nref._index >= a._index:
                msg = "A node can only depend on nodes defined before it."
                raise ValueError(msg)

            if nref._index not in a._node_dependencies:
                a._node_dependencies.append(nref._index)

    def call(
        self,
        component: ObjectiveComponent[InputT_contra, OutputT_co],
        *inputs: ParentInput_ | NodeRef[Any],
    ) -> NodeRef[OutputT_co]:
        """
        Add a component invocation and return a reference to its result.

        One input source is passed to the component directly. Multiple sources
        are assembled into a tuple in the supplied order, and no sources
        produce an empty tuple. A node reference supplies both data and the
        execution dependency needed to make that data available.

        Args:
            component: Component to invoke.
            *inputs: Parent-input or earlier-node sources consumed by the
                component.

        Returns:
            A builder-local reference to the new node's result.

        Raises:
            ValueError: If an input reference does not belong to this builder.

        """
        for inp in inputs:
            if isinstance(inp, NodeRef):
                self._validate_ref(inp)

        node_inputs: list[InputSource] = [
            PARENT_INPUT if isinstance(inp, ParentInput_) else inp._index
            for inp in inputs
        ]
        node_dependencies = list(
            dict.fromkeys(
                inp._index for inp in inputs if not isinstance(inp, ParentInput_)
            )
        )
        index = len(self._nodes)

        node = NodeRef(
            _component=component,
            _node_inputs=node_inputs,
            _node_dependencies=node_dependencies,
            _index=index,
        )
        self._nodes.append(node)
        return node

    def output(
        self, *combine_nodes: NodeRef[Any] | ParentInput_
    ) -> GraphComponent[Any, Any]:
        """
        Create a graph component exposing the selected sources.

        A single source becomes the graph result directly. Multiple sources
        produce a tuple in the supplied order. The returned component snapshots
        the current nodes, inputs, and dependencies, so the builder may be
        extended afterward without changing it.

        Args:
            *combine_nodes: One or more parent-input or node-result sources.

        Returns:
            A structural component representing the current graph.

        Raises:
            ValueError: If no output is supplied or a reference is foreign or
                invalid.

        """
        if len(combine_nodes) == 0:
            msg = "Need at least one combine node"
            raise ValueError(msg)

        for nref in combine_nodes:
            if isinstance(nref, NodeRef):
                self._validate_ref(nref)

        combine_inputs = tuple(
            PARENT_INPUT if isinstance(node, ParentInput_) else node._index
            for node in combine_nodes
        )

        return GraphComponent(
            components=tuple(node._component for node in self._nodes),
            component_dependencies=tuple(
                tuple(node._node_dependencies) for node in self._nodes
            ),
            component_inputs=tuple(tuple(node._node_inputs) for node in self._nodes),
            combine_inputs=combine_inputs,
        )
