# pyright: strict, reportUnnecessaryTypeIgnoreComment=true
"""Static public-API checks; run with ``pyright tests/static_typing.py``."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import assert_type

    from chemfit.abstract_objective_function import (
        EvaluateContext,
        OpaquePipeline,
        WrappedComponent,
        objective,
        quantity,
    )
    from chemfit.api import FitResult, fit_nevergrad
    from chemfit.combined_objective_function import CombinedObjectiveFunction
    from chemfit.executor_graph_schedule import (
        ExecutorGraphSchedule,
        ExecutorGraphScheduler,
    )
    from chemfit.fitter import Fitter
    from chemfit.graph_schedule_base import SerialGraphSchedule, SerialGraphScheduler

    Parameters = dict[str, float]
    Quantities = dict[str, float]

    @objective()
    def wrapped_objective(parameters: Parameters) -> float:
        return parameters["x"] ** 2

    assert_type(wrapped_objective, WrappedComponent[Parameters, float])
    assert_type(
        wrapped_objective.with_meta(dataset="training"),
        WrappedComponent[Parameters, float],
    )
    wrapped_objective({"x": "wrong"})  # pyright: ignore[reportArgumentType]

    serial_schedule = SerialGraphScheduler().prepare(wrapped_objective)
    assert_type(serial_schedule, SerialGraphSchedule)

    def check_executor_scheduler(executor_scheduler: ExecutorGraphScheduler) -> None:
        executor_schedule = executor_scheduler.prepare(wrapped_objective)
        assert_type(executor_schedule, ExecutorGraphSchedule)

    fitter = Fitter(wrapped_objective, {"x": 1.0})
    assert_type(fitter, Fitter[Parameters])

    fit_result = fit_nevergrad(wrapped_objective, {"x": 1.0}, budget=1)
    assert_type(fit_result, FitResult[Parameters])

    @quantity()
    def compute(parameters: Parameters, scale: float) -> Quantities:
        return {"x2": scale * parameters["x"] ** 2}

    def loss(quantities: Quantities, target: float) -> float:
        return (quantities["x2"] - target) ** 2

    assert_type(compute, WrappedComponent[Parameters, Quantities])
    term = compute.bind(scale=2.0).with_loss(loss, target=1.0)
    assert_type(term, OpaquePipeline[Parameters, float])

    combined = CombinedObjectiveFunction([wrapped_objective, term])
    assert_type(combined, CombinedObjectiveFunction[Parameters])

    def context_objective(parameters: Parameters, *, ctx: EvaluateContext) -> float:
        ctx.meta["x"] = parameters["x"]
        return parameters["x"]

    wrapped_with_context = objective(pass_ctx=True)(context_objective)
    assert_type(wrapped_with_context, WrappedComponent[Parameters, float])

    dynamic = Fitter(lambda parameters: parameters["x"], {"x": 1.0})
    assert_type(dynamic, Fitter[dict[str, Any]])
