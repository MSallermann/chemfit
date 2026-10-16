import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chemfit.abstract_objective_function import (
    Component,
    ParentInput_,
    Skipped,
    StructuralComponent,
)


class Root_:  # noqa: N801
    """Sentinel denoting the caller-supplied input to the root component."""


ROOT = Root_()


@dataclass
class GraphNode:
    """
    Represent one executable phase and its dataflow in a compiled call graph.

    ``incoming`` and ``outgoing`` are execution-order edges. ``inputs`` is a
    separate ordered list of data sources consumed by the node. An integer
    input therefore normally also creates an execution edge, while a literal
    :class:`~chemfit.abstract_objective_function.Skipped` input does not.

    ``parent`` and ``index`` identify the immediate structural component that
    owns this node. They are set on child start nodes so the scheduler can call
    the parent's ``resolve_child_input()`` policy before entering the child.

    Args:
        id: Stable position of the node in :attr:`CallGraph.nodes`.
        objective: Logical component whose lifecycle this node represents.
        outgoing: Node IDs that wait for this node to complete.
        incoming: Node IDs that must complete before this node is runnable.
        inputs: Ordered root, node-outcome, or literal-skip data sources.
        parent: Start-node ID of the immediate structural parent, if any.
        index: Child index within the immediate structural parent, if any.

    """

    id: int
    objective: Component[Any, Any]
    outgoing: tuple[int, ...] = ()
    incoming: tuple[int, ...] = ()

    inputs: tuple[int | Root_ | Skipped, ...] = ()

    parent: int | None = None
    index: int | None = None

    def __repr__(self) -> str:
        """Return a compact node identifier for scheduler diagnostics."""
        return f"{type(self)}: id={self.id}, obj={type(self.objective)}"


@dataclass
class StartNode(GraphNode):
    """
    Represent entry into a structural component's evaluation lifecycle.

    The scheduler resolves the component input, runs its begin lifecycle, and
    creates immediate child contexts at this node. A separate
    :class:`~chemfit.callgraph.FinishNode` later reduces child outcomes and runs
    the end lifecycle.
    """

    paired_end_node_id: int = -1
    contained_component_start_ids: tuple[int, ...] = ()


@dataclass
class FinishNode(GraphNode):
    """
    Represent reduction and lifecycle finalization of a structural component.

    Its inputs are selected by ``combine_inputs()``, but its execution edges
    include every immediate child so finalization cannot overtake an unselected
    child's work or side effects.
    """

    paired_start_node_id: int = -1


class AtomicNode(GraphNode):
    """Represent one indivisible component call submitted to a backend."""


@dataclass(frozen=True)
class CallGraph:
    """
    Store the immutable executable graph for one root component.

    Args:
        nodes: Nodes in stable ID order; each node's ID equals its position.
        root_id: Node that consumes the caller-supplied input.
        result_id: Node whose outcome is the root component's final result.

    """

    nodes: tuple[GraphNode, ...]
    root_id: int
    result_id: int


def objective_to_call_graph(
    objective: Component[Any, Any],
) -> CallGraph:
    """
    Compile a component hierarchy into execution and dataflow relationships.

    Atomic components become a single :class:`~chemfit.callgraph.AtomicNode`. Structural
    components become a start/finish pair surrounding recursively compiled
    children. Splitting their lifecycle allows independent children to run in
    parallel while ensuring the structural reducer and post-hooks execute only
    after the whole immediate scope has settled.

    The graph records two related concepts:

    - execution edges determine when a node is runnable; and
    - ordered input sources determine which completed values it consumes.

    Integer data sources automatically create execution edges. Explicit
    ``component_dependencies()`` may add ordering without passing a value, and
    literal skip sources pass policy data without introducing an edge.

    Args:
        objective: Root component to compile.

    Returns:
        Immutable call graph reusable across evaluation requests.

    """
    nodes: list[GraphNode] = []

    def connect(source: int, target: int) -> None:
        """
        Add one execution edge, ignoring equivalent existing edges.

        The same edge can arise from both an explicit component dependency and
        an input reference. Deduplication is essential because the scheduler's
        dependency counter expects exactly one completion per incoming node.
        """

        a, b = nodes[source], nodes[target]
        if target in a.outgoing:
            return
        nodes[source].outgoing = (*a.outgoing, target)
        nodes[target].incoming = (*b.incoming, source)

    def add_input(source: int | Root_ | Skipped, target: int) -> None:
        """
        Append an ordered data source and its required execution edge.

        Only node references imply ordering. The external root is available at
        evaluation start, and a literal skip is already a complete value.
        """

        tgt_node = nodes[target]

        if isinstance(source, int):
            connect(source, target)

        nodes[target].inputs = (*tgt_node.inputs, source)

    def graph_source(
        source: int | ParentInput_[Any] | Skipped,
        start_id: int,
        component_end_ids: list[int],
    ) -> int | Skipped:
        """
        Translate a structural source from child indices to graph node IDs.

        ``PARENT_INPUT`` maps to the structural start node, whose outcome is
        its resolved input. A child index maps to that child's end node so a
        nested structural child contributes its finalized result, not merely
        its begin phase.
        """

        if isinstance(source, ParentInput_):
            return start_id
        if isinstance(source, Skipped):
            return source
        return component_end_ids[source]

    def add(
        component: Component[Any, Any],
    ) -> tuple[int, int]:
        """
        Recursively add a component and return its entry and result node IDs.

        Atomic components return the same ID twice. Structural components
        return their distinct start and finish IDs.
        """
        if not isinstance(component, StructuralComponent):
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

        # Reserve the start ID before recursion so descendants can refer to the
        # enclosing component's resolved input.
        start_id = len(nodes)
        start_node = StartNode(start_id, component)
        nodes.append(start_node)

        contained_components = component.components()
        component_start_ids = []
        component_end_ids = []

        # Recursion assigns all child IDs before the finish node is created.
        # Retaining both endpoints is necessary because nested structural
        # children are entered at their start but expose results at their end.
        for comp in contained_components:
            cur_start_id, cur_end_id = add(comp)
            component_start_ids.append(cur_start_id)
            component_end_ids.append(cur_end_id)

        end_id = len(nodes)
        end_node = FinishNode(end_id, component)
        end_node.paired_start_node_id = start_id
        nodes.append(end_node)

        start_node.contained_component_start_ids = tuple(component_start_ids)
        start_node.paired_end_node_id = end_id

        # Every child waits until its parent's begin lifecycle has succeeded
        # and its context has been created. Parent/index metadata lets the
        # scheduler invoke the correct child-input policy at that point.
        for idx, comp_start_id in enumerate(component_start_ids):
            connect(start_id, comp_start_id)
            nodes[comp_start_id].parent = start_id
            nodes[comp_start_id].index = idx

        # Empty structural scopes still require the begin phase before finish.
        connect(start_id, end_id)

        # combine_inputs() controls reducer data, not scope completion. Waiting
        # for every immediate child prevents post-hooks from running while an
        # unselected child is still executing.
        for component_end_id in component_end_ids:
            connect(component_end_id, end_id)

        for source in component.combine_inputs():
            add_input(graph_source(source, start_id, component_end_ids), end_id)

        component_dependencies = component.component_dependencies()
        component_inputs = component.component_inputs()

        for idx, comp_start_id in enumerate(component_start_ids):
            # Explicit dependencies can impose ordering without contributing a
            # value. Input references below also add edges, so a structural
            # component need not repeat data dependencies in both declarations.
            for d in component_dependencies[idx]:
                connect(component_end_ids[d], comp_start_id)

            for i in component_inputs[idx]:
                add_input(
                    graph_source(i, start_id, component_end_ids),
                    comp_start_id,
                )

        return start_id, end_id

    start_id, end_id = add(objective)

    # The root sentinel is data that exists outside the graph. add_input()
    # records it without creating an execution edge.
    add_input(ROOT, start_id)

    return CallGraph(nodes=tuple(nodes), root_id=start_id, result_id=end_id)


def call_graph_to_dot(
    graph: CallGraph,
    dot_output_path: Path | str | None = None,
    svg_output_path: Path | str | None = None,
    path_to_dot_exec: Path | str = Path("dot"),
) -> str:
    """
    Render a call graph as Graphviz DOT and optionally write DOT and SVG files.

    Solid gray arrows show execution dependencies. Dashed blue arrows show
    ordered dataflow, including the external root input and literal skipped
    values. Because a node reference can be both an execution dependency and a
    data source, both arrows may appear between the same nodes.

    Args:
        graph: Compiled call graph to render.
        dot_output_path: Optional destination for the generated DOT text.
        svg_output_path: Optional SVG destination rendered by Graphviz.
        path_to_dot_exec: Path to the Graphviz ``dot`` executable.

    Returns:
        Generated DOT source regardless of whether files were written.

    Raises:
        subprocess.CalledProcessError: If Graphviz rendering fails.

    """

    lines = [
        "digraph CallGraph {",
        "    rankdir=LR;",
        '    node [shape=box, style=rounded, fontname="Helvetica"];',
        '    edge [fontname="Helvetica"];',
        '    ROOT [shape=ellipse, label="ROOT"];',
        '    RESULT [shape=ellipse, label="RESULT"];',
    ]

    for node in graph.nodes:
        kind = type(node).__name__.removesuffix("Node")
        component = type(node.objective).__name__

        attributes = [
            f'label="{node.id}: {kind}\\n{component}"',
        ]

        if node.id in (graph.root_id, graph.result_id):
            attributes.append("penwidth=2")

        lines.append(f"    n{node.id} [{', '.join(attributes)}];")

    # Execution dependencies
    for node in graph.nodes:
        for target in node.outgoing:
            lines.append(f"    n{node.id} -> n{target};")  # noqa: PERF401

    # Dataflow
    for node in graph.nodes:
        for index, source in enumerate(node.inputs):
            if isinstance(source, Root_):
                source_id = "ROOT"
            elif isinstance(source, Skipped):
                source_id = f"skipped_{node.id}_{index}"
                reason = "" if source.reason is None else f"\\n{source.reason!r}"
                lines.append(
                    f"    {source_id} "
                    f"[shape=ellipse, label={json.dumps('Skipped' + reason)}];"
                )
            else:
                source_id = f"n{source}"

            attributes = ["style=dashed", "color=blue"]

            if len(node.inputs) > 1:
                attributes.append(f'label="{index}"')

            lines.append(f"    {source_id} -> n{node.id} [{', '.join(attributes)}];")

    # Final output
    lines.append(f"    n{graph.result_id} -> RESULT [style=dashed, color=blue];")

    lines.append("}")
    res = "\n".join(lines)

    if dot_output_path is not None:
        dot_output_path = Path(dot_output_path)
        dot_output_path.parent.mkdir(parents=True, exist_ok=True)
        dot_output_path.write_text(res, encoding="utf-8")

    if svg_output_path is not None:
        svg_output_path = Path(svg_output_path)
        svg_output_path.parent.mkdir(parents=True, exist_ok=True)

        subprocess.run(  # noqa: S603
            [
                str(path_to_dot_exec),
                "-Tsvg",
                "-o",
                str(svg_output_path),
            ],
            check=True,
            input=res,
            text=True,
            encoding="utf-8",
        )

    return res
