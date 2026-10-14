from collections.abc import Sequence
from typing import Any, Generic
from typing_extensions import Self

from chemfit.abstract_objective_function import (
    InputT_contra,
    ObjectiveComponent,
    OutputT_co,
    StructuralComponent,
    InputSource,
    ParentInput_,
    PARENT_INPUT,
)
from chemfit.evaluate_context import EvaluateContext
from dataclasses import dataclass


class GraphComponent(StructuralComponent[InputT_contra, OutputT_co]):
    def __init__(
        self,
        components: Sequence[ObjectiveComponent[Any, Any]],
        component_dependencies: tuple[tuple[int, ...], ...],
        component_inputs: tuple[tuple[InputSource, ...], ...],
        combine_inputs: tuple[InputSource, ...],
    ) -> None:
        super().__init__()
        self._components = components
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
        self, outcomes: tuple[Any, ...], ctx: EvaluateContext
    ) -> OutputT_co:
        raise NotImplementedError


@dataclass(frozen=True)
class NodeRef(Generic[InputT_contra, OutputT_co]):
    _input: ParentInput_ | Self
    _component: ObjectiveComponent[InputT_contra, OutputT_co]


class GraphBuilder:

    def __init__(self) -> None:
        self._nodes: list[NodeRef[Any, Any]] = []

    @property
    def root_input(self):
        return PARENT_INPUT

    def call(
        self,
        component: ObjectiveComponent[InputT_contra, OutputT_co],
        input: ParentInput_ | NodeRef[InputT_contra, OutputT_co],
    ) -> NodeRef:

        if isinstance(input, NodeRef) and input not in self._nodes:
            msg = "Cannot input a NodeRef that is not already in the graph."
            raise ValueError(msg)

        node = NodeRef(_input=input, _component=component)
        self._nodes.append(node)
        return node

    def finish(self) -> GraphComponent[Any, Any]:
        ...
