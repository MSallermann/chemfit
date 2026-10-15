# pyright: strict, reportUnnecessaryTypeIgnoreComment=true
"""Static public-API checks; run with ``pyright tests/static_typing.py``."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import assert_type

    from chemfit import Component, DirectSchedule, DirectScheduler, FitContext
    from chemfit.abstract_objective_function import (
        EvaluateContext,
        OpaquePipeline,
        PipelineComponent,
        PostEvaluationHook,
        PreEvaluationHook,
        WrappedComponent,
        component,
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
    from chemfit.graph_component import GraphBuilder, GraphComponent, NodeRef
    from chemfit.graph_schedule_base import SerialGraphSchedule, SerialGraphScheduler
    from chemfit.scheduling import EvaluationRequest

    Parameters = dict[str, float]
    Quantities = dict[str, float]

    @objective()
    def wrapped_objective(parameters: Parameters) -> float:
        return parameters["x"] ** 2

    assert_type(wrapped_objective, WrappedComponent[Parameters, float])
    public_component: Component[Parameters, float] = wrapped_objective

    class RenamedContextHook:
        def pre_eval(self, context: EvaluateContext) -> None:
            context.meta["started"] = True

        def post_eval(self, context: EvaluateContext) -> None:
            context.meta["finished"] = True

    pre_hook: PreEvaluationHook = RenamedContextHook()
    post_hook: PostEvaluationHook = RenamedContextHook()
    wrapped_objective.register_eval_hook(hook=pre_hook)
    wrapped_objective.register_eval_hook(hook=post_hook)

    assert_type(
        wrapped_objective.with_meta(dataset="training"),
        WrappedComponent[Parameters, float],
    )
    assert_type(
        wrapped_objective.with_context_configurator(lambda _ctx: None),
        WrappedComponent[Parameters, float],
    )
    wrapped_objective({"x": "wrong"})  # pyright: ignore[reportArgumentType]

    fit_context = FitContext()
    assert_type(fit_context, FitContext)

    request = EvaluationRequest(input={"x": 1.0}, ctx=fit_context)
    assert_type(request.input, dict[str, float])

    direct_scheduler: DirectScheduler[Parameters, float] = DirectScheduler()
    direct_schedule = direct_scheduler.prepare(wrapped_objective)
    assert_type(direct_schedule, DirectSchedule[Parameters, float])

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
    structural_term = combined.with_loss(lambda value: value)
    assert_type(structural_term, PipelineComponent[Parameters, float])

    def context_objective(parameters: Parameters, *, ctx: EvaluateContext) -> float:
        ctx.meta["x"] = parameters["x"]
        return parameters["x"]

    wrapped_with_context = objective(pass_ctx=True)(context_objective)
    assert_type(wrapped_with_context, WrappedComponent[Parameters, float])

    @component()
    def label(parameters: Parameters) -> str:
        return str(parameters["x"])

    @component()
    def combine_graph_values(values: tuple[float, str]) -> str:
        return f"{values[0]}: {values[1]}"

    graph_builder = GraphBuilder()
    number_node = graph_builder.call(wrapped_objective, graph_builder.input)
    label_node = graph_builder.call(label, graph_builder.input)
    graph_result = graph_builder.call(
        combine_graph_values,
        number_node,
        label_node,
    )
    assert_type(number_node, NodeRef[float])
    assert_type(label_node, NodeRef[str])
    assert_type(graph_result, NodeRef[str])
    assert_type(graph_builder.output(graph_result), GraphComponent[Any, Any])

    dynamic = Fitter(lambda parameters: parameters["x"], {"x": 1.0})
    assert_type(dynamic, Fitter[dict[str, Any]])
