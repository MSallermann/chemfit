"""Tests for static objective and quantity-computer resource annotations."""

from dataclasses import fields

from chemfit.abstract_objective_function import (
    EvaluateContext,
    ObjectiveFunctor,
    QuantityComputer,
    ResourceRequest,
)
from chemfit.callgraph import LeafNode, objective_to_call_tree
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.tree_schedule import LeafTask
from chemfit.wrap_funcs import objective, quantity

Parameters = dict[str, float]
Quantities = dict[str, float]


class PlainObjective(ObjectiveFunctor[Parameters]):
    """Objective relying on the default empty resource request."""

    def _evaluate(self, parameters: Parameters, ctx: EvaluateContext) -> float:
        """Return the input value."""

        del ctx
        return parameters["x"]


class PlainComputer(QuantityComputer[Parameters, Quantities]):
    """Computer relying on the default empty resource request."""

    def _compute(
        self,
        parameters: Parameters,
        ctx: EvaluateContext,
    ) -> Quantities:
        """Return the input as a quantity."""

        del ctx
        return {"x": parameters["x"]}


def test_core_resources_are_mutable_and_assignment_copies() -> None:
    """Base resource mappings are optional, mutable, and copied on assignment."""

    objective = PlainObjective()
    computer = PlainComputer()
    supplied = {"cpus": 4.0}

    assert objective.resources == {}
    assert computer.resources == {}

    objective.resources = supplied
    supplied["cpus"] = 16.0
    assert objective.resources == {"cpus": 4.0}

    computer.resources["gpus"] = 1.0  # type: ignore[index]
    assert computer.resources == {"gpus": 1.0}


def test_decorators_preserve_resources_through_bind_and_with_loss() -> None:
    """Wrapped computations retain and delegate their annotations."""

    requested: ResourceRequest = {
        "cpus": 4,
        "licenses.solver": 1,
    }

    @objective(resources=requested)
    def loss_objective(parameters: Parameters, scale: float) -> float:
        return scale * parameters["x"]

    requested_dict = requested
    assert loss_objective.resources == requested_dict
    assert loss_objective.resources is not requested_dict
    assert loss_objective.bind(scale=2.0).resources == requested_dict

    @quantity(resources={"cpus": 8, "gpus": 1, "memory_gb": 16})
    def simulation(parameters: Parameters, offset: float) -> Quantities:
        return {"x": parameters["x"] + offset}

    bound_simulation = simulation.bind(offset=1.0)
    quantity_objective = bound_simulation.with_loss(
        lambda quantities: quantities["x"] ** 2
    )

    expected = {"cpus": 8, "gpus": 1, "memory_gb": 16}
    assert simulation.resources == expected
    assert bound_simulation.resources == expected
    assert quantity_objective.resources is bound_simulation.resources

    quantity_objective.resources = {"mpi_ranks": 4}
    assert bound_simulation.resources == {"mpi_ranks": 4}


def test_resources_remain_static_tree_metadata() -> None:
    """Schedulers can compile leaf resources without changing LeafTask."""

    @objective(resources={"cpus": 2})
    def cpu_term(parameters: Parameters) -> float:
        return parameters["x"]

    @quantity(resources={"gpus": 1})
    def gpu_computer(parameters: Parameters) -> Quantities:
        return {"x": parameters["x"]}

    combined_objective = CombinedObjectiveFunction(
        [cpu_term, gpu_computer.with_loss(lambda quantities: quantities["x"])]
    )
    tree = objective_to_call_tree(combined_objective)
    resources_by_node = {
        node.id: node.objective.resources
        for node in tree.nodes
        if isinstance(node, LeafNode)
    }

    assert list(resources_by_node.values()) == [{"cpus": 2}, {"gpus": 1}]
    assert [field.name for field in fields(LeafTask)] == [
        "run_id",
        "node_id",
        "parameters",
        "ctx",
    ]
