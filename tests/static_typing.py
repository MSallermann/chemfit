# pyright: strict, reportUnnecessaryTypeIgnoreComment=true
"""Static API checks; run with ``pyright tests/static_typing.py``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import assert_type

    from chemfit.abstract_objective_function import (
        EvaluateContext,
        QuantityComputerObjectiveFunction,
    )
    from chemfit.api import FitResult, fit_nevergrad
    from chemfit.ase_objective_function import (
        ASEComputer,
        ASEEvaluator,
        AtomsFactory,
        CalculatorFactory,
        QuantityProcessor,
    )
    from chemfit.async_helpers import async_eval_many, async_eval_one
    from chemfit.combined_objective_function import CombinedObjectiveFunction
    from chemfit.external_computer import ExternalQuantityComputer
    from chemfit.fitter import Fitter
    from chemfit.tree_schedule import SerialTreeSchedule, SerialTreeScheduler
    from chemfit.wrap_funcs import (
        WrappedObjectiveFunctor,
        WrappedQuantityComputer,
        objective,
        quantity,
    )

    Parameters = dict[str, float]
    IncompatibleParameters = dict[str, str]
    Quantities = dict[str, float]

    # Objective wrappers preserve the concrete parameter dictionary type.
    @objective()
    def wrapped_objective(parameters: Parameters) -> float:
        return parameters["x"] ** 2

    assert_type(wrapped_objective, WrappedObjectiveFunctor[Parameters])
    assert_type(
        wrapped_objective.with_meta(dataset="training"),
        WrappedObjectiveFunctor[Parameters],
    )
    wrapped_objective({"x": "wrong"})  # pyright: ignore[reportArgumentType]

    root_leaf_schedule = SerialTreeScheduler().prepare(wrapped_objective)
    assert_type(root_leaf_schedule, SerialTreeSchedule[Parameters])

    # Context-aware callables are accepted without widening their parameters.
    def objective_with_context(parameters: Parameters, _ctx: EvaluateContext) -> float:
        return parameters["x"] ** 2

    context_objective = objective()(objective_with_context)
    assert_type(context_objective, WrappedObjectiveFunctor[Parameters])
    context_objective({"x": "wrong"})  # pyright: ignore[reportArgumentType]

    # Async helpers preserve the objective's parameter type for individual
    # evaluations and batches.
    async_one = async_eval_one(context_objective, {"x": 1.0}, EvaluateContext())
    invalid_async_one = async_eval_one(
        context_objective,
        {"x": "wrong"},  # pyright: ignore[reportArgumentType]
        EvaluateContext(),
    )
    async_many = async_eval_many(context_objective, [{"x": 1.0}], [EvaluateContext()])

    # Fitter carries the objective's parameter type through its public API.
    fitter = Fitter(wrapped_objective, {"x": 1.0})
    assert_type(fitter, Fitter[Parameters])
    fitter.objective_function({"x": "wrong"})  # pyright: ignore[reportArgumentType]

    fit_result = fit_nevergrad(
        wrapped_objective,
        {"x": 1.0},
        budget=1,
        initial_observations=[({"x": 0.0}, 0.0)],
    )
    assert_type(fit_result, FitResult[Parameters])

    # An unannotated lambda falls back to a dynamic dictionary, keeping the
    # common concise form usable without discarding precise function annotations.
    inferred_fitter = Fitter(lambda parameters: parameters["x"] ** 2, {"x": 0.0})
    assert_type(inferred_fitter, Fitter[dict[str, Any]])

    # A heterogeneous nested dictionary intentionally uses Any unless the user
    # supplies a more precise schema such as a TypedDict.
    def dynamic_objective(parameters: dict[str, Any]) -> float:
        return parameters["core"]["x"] ** 2

    nested_fitter = Fitter(
        dynamic_objective,
        {
            "model": "quadratic",
            "core": {"x": 1.0, "weights": [1.0, 2.0]},
            "metadata": "fixed",
        },
    )
    assert_type(nested_fitter, Fitter[dict[str, Any]])
    nested_fitter.initial_parameters["core"]["x"]

    # Quantity wrappers preserve both their parameter and result types. Binding
    # arguments and attaching a loss function must not erase either one.
    @quantity()
    def compute_quantities(parameters: Parameters, scale: float) -> Quantities:
        return {"x2": scale * parameters["x"] ** 2}

    def loss(quantities: Quantities, target: float) -> float:
        return (quantities["x2"] - target) ** 2

    assert_type(
        compute_quantities,
        WrappedQuantityComputer[Parameters, Quantities],
    )
    assert_type(
        compute_quantities.with_meta(dataset="training"),
        WrappedQuantityComputer[Parameters, Quantities],
    )
    compute_quantities({"x": "wrong"})  # pyright: ignore[reportArgumentType]

    bound_quantities = compute_quantities.bind(scale=2.0)
    quantity_objective = bound_quantities.with_loss(loss, target=1.0)
    assert_type(
        quantity_objective,
        QuantityComputerObjectiveFunction[Parameters, Quantities],
    )
    assert_type(
        quantity_objective.with_meta(observable="x2"),
        QuantityComputerObjectiveFunction[Parameters, Quantities],
    )
    quantity_objective({"x": "wrong"})  # pyright: ignore[reportArgumentType]

    # Combined objectives preserve a common parameter type with either the
    # simple reducer API or the context-aware aggregator API.
    def reduce_values(values: list[float]) -> float:
        return sum(values)

    def aggregate_values(
        values: list[float],
        _outputs: list[dict[str, Any] | None],
        _context: EvaluateContext,
    ) -> float:
        return sum(values)

    def handle_failure(
        _error: Exception, _context: EvaluateContext, _index: int
    ) -> float | None:
        return None

    combined = CombinedObjectiveFunction([wrapped_objective])
    assert_type(combined, CombinedObjectiveFunction[Parameters])
    assert_type(
        combined.with_weights([1.0]),
        CombinedObjectiveFunction[Parameters],
    )
    assert_type(
        combined.with_reduction(reduce_values),
        CombinedObjectiveFunction[Parameters],
    )
    assert_type(
        combined.with_aggregator(aggregate_values),
        CombinedObjectiveFunction[Parameters],
    )
    assert_type(
        combined.with_exception_handler(handle_failure),
        CombinedObjectiveFunction[Parameters],
    )
    combined({"x": "wrong"})  # pyright: ignore[reportArgumentType]

    combined_with_reducer = CombinedObjectiveFunction(
        [wrapped_objective],
        reduction=reduce_values,
        exception_handler=handle_failure,
    )
    assert_type(combined_with_reducer, CombinedObjectiveFunction[Parameters])

    combined_with_aggregator = CombinedObjectiveFunction(
        [wrapped_objective],
        aggregator=aggregate_values,
    )
    assert_type(combined_with_aggregator, CombinedObjectiveFunction[Parameters])

    # Aggregators must account for objectives that do not produce quantities.
    # Consequently, a callback accepting only dictionaries is too narrow.
    def aggregator_requiring_quantities(
        values: list[float],
        _outputs: list[dict[str, Any]],
        _context: EvaluateContext,
    ) -> float:
        return sum(values)

    CombinedObjectiveFunction(
        [wrapped_objective],
        aggregator=aggregator_requiring_quantities,  # pyright: ignore[reportArgumentType]
    )

    # Objectives with incompatible parameter value types must not be combined,
    # whether the combined objective's type is inferred or explicit.
    def incompatible_objective(parameters: IncompatibleParameters) -> float:
        return float(parameters["x"])

    mixed_parameter_combination = CombinedObjectiveFunction(
        [
            wrapped_objective,  # pyright: ignore[reportArgumentType]
            incompatible_objective,
        ]
    )

    explicitly_typed_mixed_combination = CombinedObjectiveFunction[Parameters](
        [wrapped_objective, incompatible_objective]  # pyright: ignore[reportArgumentType]
    )

    # External computers connect the command callbacks' parameter type to
    # the parsers' common output type.
    def parse_outputs(_output_file: Path) -> Quantities:
        return {"value": 1.0}

    def make_command(
        _parameters: Parameters,
        _workdir: Path,
        _ctx: EvaluateContext,
    ) -> list[str]:
        return ["true"]

    def prepare_input(
        _parameters: Parameters,
        _workdir: Path,
        _ctx: EvaluateContext,
    ) -> None:
        return None

    external = (
        ExternalQuantityComputer[Parameters, Quantities](
            base_working_directory="work",
        )
        .with_hook(prepare_input)
        .with_cmd(make_command)
        .with_parser(parse_outputs, "result.txt")
    )
    assert_type(
        external,
        ExternalQuantityComputer[Parameters, Quantities],
    )
    external({"x": "wrong"})  # pyright: ignore[reportArgumentType]

    # ASE protocol assignments check each callback independently. Constructing
    # the computer must then preserve the parameter and quantity types.
    def make_calculator(
        _parameters: Parameters,
        _atoms: Any,
        _ctx: EvaluateContext,
    ) -> Any:
        return object()

    def make_atoms() -> Any:
        return None

    def evaluate_atoms(
        _parameters: Parameters,
        _atoms: Any,
        _ctx: EvaluateContext,
    ) -> None:
        return None

    def process_quantities(
        _calculator: Any,
        _atoms: Any,
        _ctx: EvaluateContext,
    ) -> Quantities:
        return {"value": 1.0}

    calculator_factory: CalculatorFactory[Parameters] = make_calculator
    ase_evaluator: ASEEvaluator[Parameters] = evaluate_atoms
    atoms_factory: AtomsFactory = make_atoms
    quantity_processor: QuantityProcessor[Quantities] = process_quantities
    ase_computer = ASEComputer(
        atoms_factory=atoms_factory,
        calculator_factory=calculator_factory,
        quantity_processors=[quantity_processor],
    ).with_evaluator(ase_evaluator)
    assert_type(
        ase_computer,
        ASEComputer[Parameters, Quantities],
    )
    ase_computer({"x": "wrong"})  # pyright: ignore[reportArgumentType]

    # The default ASE quantity processor produces a heterogeneous dictionary;
    # omitting processors must not make the parameter type unknown.
    default_ase_computer = ASEComputer[Parameters, dict[str, Any]](
        atoms_factory=atoms_factory,
        calculator_factory=calculator_factory,
    )
    assert_type(
        default_ase_computer,
        ASEComputer[Parameters, dict[str, Any]],
    )

    def default_ase_loss(quantities: dict[str, Any]) -> float:
        return float(quantities["value"])

    default_ase_objective = default_ase_computer.with_loss(default_ase_loss)
    assert_type(
        default_ase_objective,
        QuantityComputerObjectiveFunction[Parameters, dict[str, Any]],
    )
