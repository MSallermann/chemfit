from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias

from chemfit.abstract_objective_function import (
    PARENT_INPUT,
    ObjectiveComponent,
    ParentInput_,
    StructuralComponent,
)


class Root_: ...  # noqa: N801


ROOT = Root_()


@dataclass
class GraphNode:
    id: int
    objective: ObjectiveComponent[Any, Any]
    outgoing: tuple[int, ...] = ()
    incoming: tuple[int, ...] = ()

    # Input source, ROOT for the root parameters of
    # of the evaluation
    # or a node id for the output of a component
    # if the tuple contains more than one element
    # a tuple of inputs is passed
    # If it contains exactly one element that element
    # is passed without the surrounding tuple
    inputs: tuple[int | Root_, ...] = ()


class StartNode(GraphNode):
    pass


class FinishNode(GraphNode):
    pass


class AtomicNode(GraphNode):
    pass


@dataclass(frozen=True)
class CallGraph:
    nodes: tuple[GraphNode, ...]


def objective_to_call_graph(
    objective: ObjectiveComponent[Any, Any],
) -> CallGraph:

    nodes: list[GraphNode] = []

    def connect(source: int, target: int) -> None:
        """``source`` must exectute before ``target."""

        a, b = nodes[source], nodes[target]
        nodes[source].outgoing = (*a.outgoing, target)
        nodes[target].incoming = (*b.incoming, source)

    def add_input(source: int | Root_, target: int) -> None:
        """
        ``source``'s output is an input to ``target.

        Only call this after calling ``connect(source,target)``
        """

        tgt_node = nodes[target]

        if not isinstance(source, Root_):
            src_node = nodes[source]
            assert target in src_node.outgoing
            assert source in tgt_node.incoming

        nodes[target].inputs = (*tgt_node.inputs, source)

    def add(
        component: ObjectiveComponent[Any, Any],
    ) -> tuple[int, int]:

        if not isinstance(component, StructuralComponent):
            # non-structural components get a simple atomic node
            node_id = len(nodes)
            nodes.append(
                AtomicNode(
                    id=node_id,
                    objective=component,
                    incoming=(),
                    outgoing=(),
                )
            )
            return node_id, node_id
        else:
            # add the start node of the structural component
            start_id = len(nodes)
            nodes.append(StartNode(start_id, component))

            #######################################
            # Add contained components and finish node
            #######################################

            contained_components = component.components()
            component_start_ids = []
            component_end_ids = []
            # we dont know the node_ids of later stages yet
            # so it's not possible to populate outgoing/incoming
            # Therefore, we first collect all the component start and end ids
            # and add them to the nodes list
            for comp in contained_components:
                cur_start_id, cur_end_id = add(comp)
                component_start_ids.append(cur_start_id)
                component_end_ids.append(cur_end_id)

            # we can already append the finish node
            # it depends on all contained components finishing (duh)
            end_id = len(nodes)
            nodes.append(FinishNode(end_id, component))

            ####################################
            # Hook up the dependencies
            ####################################

            # 1. The contained components depend on the start node
            for comp_start_id in component_start_ids:
                connect(start_id, comp_start_id)

            # 2. the end node depends on the contained components
            for comp_end_id in component_end_ids:
                connect(comp_end_id, end_id)

            # 3. hook up incoming/outgoing for the contained components
            # ... query component dependencies from the containing structural component
            component_dependencies = component.component_dependencies()
            # ... query component inputs from the containing structural component
            component_inputs = component.component_inputs()

            for idx, comp_start_id in enumerate(component_start_ids):
                # Each component start depends on the end of its dependencies
                for d in component_dependencies[idx]:
                    connect(component_end_ids[d], comp_start_id)

                for i in component_inputs[idx]:
                    source: int = (
                        start_id
                        if isinstance(i, ParentInput_)
                        else component_end_ids[i]
                    )
                    add_input(source, comp_start_id)

            # return the start and end id of the structural component
            return start_id, end_id

    start_id, end_id = add(objective)
    add_input(ROOT, start_id)

    return CallGraph(nodes=tuple(nodes))


def call_graph_to_dot(
    graph: CallGraph,
    dot_output_path: Path | None = None,
    svg_output_path: Path | None = None,
) -> str:
    import subprocess

    """Generate a Graphviz DOT representation of a CallGraph."""

    lines = [
        "digraph CallGraph {",
        "    rankdir=LR;",
        '    node [shape=box, style=rounded, fontname="Helvetica"];',
        '    edge [fontname="Helvetica"];',
        '    ROOT [shape=ellipse, label="ROOT"];',
    ]

    for node in graph.nodes:
        kind = type(node).__name__.removesuffix("Node")
        component = type(node.objective).__name__

        lines.append(f'    n{node.id} [label="{node.id}: {kind}\\n{component}"];')

    # Execution dependencies
    for node in graph.nodes:
        for target in node.outgoing:
            lines.append(f"    n{node.id} -> n{target};")  # noqa: PERF401

    # Dataflow
    for node in graph.nodes:
        for index, source in enumerate(node.inputs):
            source_id = "ROOT" if isinstance(source, Root_) else f"n{source}"
            lines.append(
                f"    {source_id} -> n{node.id} "
                f'[style=dashed, color=blue, label="{index}"];'
            )

    lines.append("}")
    res = "\n".join(lines)

    if dot_output_path is not None:
        with dot_output_path.open("w") as f:
            f.write(res)

    if svg_output_path is not None:
        if dot_output_path is None:
            msg = "Have to set `dot_output_path` if `svg_output_path` is set."
            raise ValueError(msg)

        subprocess.run(
            ["dot", "-Tsvg", f"{dot_output_path}", "-o", f"{svg_output_path}"]
        )

    return res
