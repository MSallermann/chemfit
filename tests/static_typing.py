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
    from chemfit.ase_objective_function import (
        AtomsFactory,
        CalculatorFactory,
        ParameterApplier,
        QuantityProcessor,
        SinglePointASEComputer,
    )
    from chemfit.async_helpers import async_eval_many, async_eval_one
    from chemfit.combined_objective_function import CombinedObjectiveFunction
    from chemfit.executor_wrapper_cob import ExecutorWrapperCOB
    from chemfit.file_based_computer import FileBasedQuantityComputer
    from chemfit.fitter import Fitter
    from chemfit.wrap_funcs import (
        WrappedObjectiveFunctor,
        WrappedQuantityComputer,
        to_objective_functor,
        to_quantity_computer,
    )

    Parameters = dict[str, float]
    IncompatibleParameters = dict[str, str]
    Quantities = dict[str, float]

    # Objective wrappers preserve the concrete parameter dictionary type.
    @to_objective_functor()
    def objective(parameters: Parameters) -> float:
        return parameters["x"] ** 2

    assert_type(objective, WrappedObjectiveFunctor[Parameters])
    objective({"x": "wrong"})  # pyright: ignore[reportArgumentType]

    # Context-aware callables are accepted without widening their parameters.
    def objective_with_context(parameters: Parameters, _ctx: EvaluateContext) -> float:
        return parameters["x"] ** 2

    context_objective = to_objective_functor()(objective_with_context)
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
    fitter = Fitter(objective, {"x": 1.0})
    assert_type(fitter, Fitter[Parameters])
    fitter.objective_function(
        {"x": "wrong"}  # pyright: ignore[reportArgumentType]
    )

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
    @to_quantity_computer()
    def compute_quantities(parameters: Parameters, scale: float) -> Quantities:
        return {"x2": scale * parameters["x"] ** 2}

    def loss(quantities: Quantities, target: float) -> float:
        return (quantities["x2"] - target) ** 2

    assert_type(
        compute_quantities,
        WrappedQuantityComputer[Parameters, Quantities],
    )
    compute_quantities({"x": "wrong"})  # pyright: ignore[reportArgumentType]

    bound_quantities = compute_quantities.bind(scale=2.0)
    quantity_objective = bound_quantities.with_loss(loss, target=1.0)
    assert_type(
        quantity_objective,
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

    combined = CombinedObjectiveFunction([objective])
    assert_type(combined, CombinedObjectiveFunction[Parameters])
    combined({"x": "wrong"})  # pyright: ignore[reportArgumentType]

    combined_with_reducer = CombinedObjectiveFunction(
        [objective],
        reduction=reduce_values,
        exception_handler=handle_failure,
    )
    assert_type(combined_with_reducer, CombinedObjectiveFunction[Parameters])

    combined_with_aggregator = CombinedObjectiveFunction(
        [objective],
        reduction=aggregate_values,
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
        [objective],
        reduction=aggregator_requiring_quantities,  # pyright: ignore[reportArgumentType]
    )

    # Objectives with incompatible parameter value types must not be combined,
    # whether the combined objective's type is inferred or explicit.
    def incompatible_objective(parameters: IncompatibleParameters) -> float:
        return float(parameters["x"])

    mixed_parameter_combination = CombinedObjectiveFunction(
        [
            objective,  # pyright: ignore[reportArgumentType]
            incompatible_objective,
        ]
    )

    explicitly_typed_mixed_combination = CombinedObjectiveFunction[Parameters](
        [objective, incompatible_objective]  # pyright: ignore[reportArgumentType]
    )

    # Executor wrappers retain the combined objective's parameter type.
    executor_wrapped = ExecutorWrapperCOB(combined)
    assert_type(executor_wrapped, ExecutorWrapperCOB[Parameters])
    executor_wrapped({"x": "wrong"})  # pyright: ignore[reportArgumentType]

    # File-based computers connect the command callbacks' parameter type to
    # the parsers' common output type.
    def parse_outputs(_output_files: list[Path]) -> Quantities:
        return {"value": 1.0}

    def make_command(_parameters: Parameters, _workdir: Path) -> list[str]:
        return ["true"]

    def presubmit(_parameters: Parameters, _workdir: Path) -> None:
        return None

    external = FileBasedQuantityComputer(
        output_files=["result.txt"],
        output_parsers=parse_outputs,
        base_working_directory="work",
        executable_cmd=make_command,
        presubmit_hook=presubmit,
    )
    assert_type(
        external,
        FileBasedQuantityComputer[Parameters, Quantities],
    )
    external({"x": "wrong"})  # pyright: ignore[reportArgumentType]

    # ASE protocol assignments check each callback independently. Constructing
    # the computer must then preserve the parameter and quantity types.
    def apply_parameters(_atoms: Any, _parameters: Parameters, /) -> None:
        return None

    def make_calculator(_atoms: Any) -> None:
        return None

    def make_atoms() -> Any:
        return None

    def process_quantities(_calculator: Any, _atoms: Any) -> Quantities:
        return {"value": 1.0}

    parameter_applier: ParameterApplier[Parameters] = apply_parameters
    calculator_factory: CalculatorFactory = make_calculator
    atoms_factory: AtomsFactory = make_atoms
    quantity_processor: QuantityProcessor[Quantities] = process_quantities
    ase_computer = SinglePointASEComputer(
        calc_factory=calculator_factory,
        param_applier=parameter_applier,
        atoms_factory=atoms_factory,
        quantity_processors=[quantity_processor],
    )
    assert_type(
        ase_computer,
        SinglePointASEComputer[Parameters, Quantities],
    )
    ase_computer({"x": "wrong"})  # pyright: ignore[reportArgumentType]

    # The default ASE quantity processor produces a heterogeneous dictionary;
    # omitting processors must not make the parameter type unknown.
    default_ase_computer = SinglePointASEComputer(
        calc_factory=calculator_factory,
        param_applier=parameter_applier,
        atoms_factory=atoms_factory,
    )
    assert_type(
        default_ase_computer,
        SinglePointASEComputer[Parameters, dict[str, Any]],
    )

    def default_ase_loss(quantities: dict[str, Any]) -> float:
        return float(quantities["value"])

    default_ase_objective = default_ase_computer.with_loss(default_ase_loss)
    assert_type(
        default_ase_objective,
        QuantityComputerObjectiveFunction[Parameters, dict[str, Any]],
    )
