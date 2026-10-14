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
    def __init__(
        self,
        components: Sequence[ObjectiveComponent[Any, Any]],
        component_dependencies: tuple[tuple[int, ...], ...],
        component_inputs: tuple[tuple[InputSource, ...], ...],
        combine_inputs: tuple[InputSource, ...],
    ) -> None:
        """Store an immutable snapshot of a builder's graph topology."""
        super().__init__()
        self._components = tuple(components)
        self._component_dependencies = component_dependencies
        self._component_inputs = component_inputs
        self._combine_inputs = combine_inputs

    def components(self) -> Sequence[ObjectiveComponent[Any, Any]]:
        return self._components

    def component_dependencies(self) -> tuple[tuple[int, ...], ...]:
        return self._component_dependencies

    def component_inputs(self) -> tuple[tuple[InputSource, ...], ...]:
        return self._component_inputs

    def combine_inputs(self) -> tuple[InputSource, ...]:
        return self._combine_inputs

    def combine_outcomes(
        self,
        outcomes: tuple[Any, ...],
        ctx: EvaluateContext,  # noqa: ARG002
    ) -> OutputT_co:
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
    _component: ObjectiveComponent[Any, OutputT_co]
    _node_inputs: list[InputSource]
    _node_dependencies: list[int]
    _index: int


class GraphBuilder:
    def __init__(self) -> None:
        """Initialize an empty graph."""
        self._nodes: list[NodeRef[Any]] = []

    @property
    def input(self) -> ParentInput_:
        return PARENT_INPUT

    def _validate_ref(self, ref: NodeRef[Any]) -> None:
        if not 0 <= ref._index < len(self._nodes) or self._nodes[ref._index] is not ref:
            msg = "NodeRef belongs to another graph."
            raise ValueError(msg)

    def depends_on(self, a: NodeRef[Any], *others: NodeRef[Any]) -> None:
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
