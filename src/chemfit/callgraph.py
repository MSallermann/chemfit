from dataclasses import dataclass
from typing import Any, TypeAlias

from chemfit.abstract_objective_function import ObjectiveFunctor
from chemfit.scheduling import SchedulableCompositeObjective

NodeId: TypeAlias = int


@dataclass(frozen=True)
class LeafNode:
    id: NodeId
    parent_id: NodeId | None
    child_idx: int | None
    objective: ObjectiveFunctor[Any]


@dataclass(frozen=True)
class CombineNode:
    id: NodeId
    parent_id: NodeId | None
    child_idx: int | None
    objective: SchedulableCompositeObjective[Any]
    children: tuple[NodeId, ...]


CallNode: TypeAlias = LeafNode | CombineNode


@dataclass(frozen=True)
class CallTree:
    root: NodeId
    nodes: tuple[CallNode, ...]


def build_nodes_from_objective(
    objective: ObjectiveFunctor[Any],
    nodes: list[CallNode] | None = None,
) -> list[CallNode]:
    """Build call-tree nodes for an ordinary or composite objective."""

    if nodes is None:
        nodes = []

    def add_objective(
        objective: ObjectiveFunctor[Any],
        parent_id: NodeId | None,
        child_idx: NodeId | None,
    ) -> NodeId:
        # the id of the current node is simply its position in the node list
        node_id = len(nodes)

        # if the current node is a leaf, we simply append it to the node list and return
        if not isinstance(objective, SchedulableCompositeObjective):
            nodes.append(
                LeafNode(
                    node_id,
                    parent_id,
                    child_idx,
                    objective,
                )
            )
            return node_id

        # Reserve a composite node's position before recursively adding its
        # children so that every node ID remains equal to its list index.
        nodes.append(
            CombineNode(
                node_id,
                parent_id,
                child_idx,
                objective,
                children=(),
            )
        )

        # Then we figure out its children by recursion
        children = tuple(
            add_objective(term, parent_id=node_id, child_idx=child_idx)
            for child_idx, term in enumerate(objective.child_objectives())
        )

        # Finally, we can replace the temporary node with a fresh one, now that child IDs are known.
        # (We cannot just overwrite .children since CombineNode is a frozen dataclass)
        nodes[node_id] = CombineNode(
            node_id,
            parent_id,
            child_idx=child_idx,
            objective=objective,
            children=children,
        )

        return node_id

    # On the first invocation we add the root node, so parent_id is None
    add_objective(objective, parent_id=None, child_idx=None)
    return nodes


def build_nodes_from_cob(
    cob: ObjectiveFunctor[Any],
    nodes: list[CallNode] | None = None,
) -> list[CallNode]:
    """Build nodes through the legacy generalized-builder entry point."""

    return build_nodes_from_objective(cob, nodes)


def print_call_tree(tree: CallTree) -> None:
    """Print a CallTree as a tree drawing."""

    def label(node_id: NodeId) -> str:
        node = tree.nodes[node_id]
        obj = node.objective

        name = getattr(obj, "__name__", obj.__class__.__name__)
        return (
            f"[{node_id}] {name} (parent={node.parent_id}, child_idx={node.child_idx})"
        )

    def visit(
        node_id: NodeId,
        prefix: str = "",
        is_last: bool = True,
        is_root: bool = False,
    ) -> None:
        node = tree.nodes[node_id]

        if is_root:
            print(label(node_id))
        else:
            branch = "└── " if is_last else "├── "
            print(prefix + branch + label(node_id))

        if not isinstance(node, CombineNode):
            return

        child_prefix = prefix
        if not is_root:
            child_prefix += "    " if is_last else "│   "

        for i, child_id in enumerate(node.children):
            visit(
                child_id,
                prefix=child_prefix,
                is_last=i == len(node.children) - 1,
            )

    visit(tree.root, is_root=True)


def objective_to_call_tree(objective: ObjectiveFunctor[Any]) -> CallTree:
    """Compile an ordinary or composite objective into a call tree."""

    return CallTree(0, tuple(build_nodes_from_objective(objective)))


def cob_to_call_tree(cob: ObjectiveFunctor[Any]) -> CallTree:
    """Compile an objective while preserving the legacy entry point."""

    return objective_to_call_tree(cob)
