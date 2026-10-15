# pyright: strict, reportUnnecessaryTypeIgnoreComment=true
"""Static public-API checks; run with ``pyright tests/static_typing.py``."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import assert_type

    from chemfit import (
        Component,
        DirectSchedule,
        DirectScheduler,
        FitContext,
        GraphBuilder,
    )
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
    from chemfit.graph_component import GraphComponent, NodeRef
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

    parameters: Parameters = {"x": 1.0}
    parameters1: Parameters = {"x": 1.0}
    parameters2: Parameters = {"x": 2.0}
    single: float = fitter.evaluate(parameters)
    batch: list[float] = fitter.evaluate([parameters1, parameters2])
    assert_type(single, float)
    assert_type(batch, list[float])

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
    assert_type(graph_builder.output(graph_result), GraphComponent[Any, str])
    assert_type(
        graph_builder.output(number_node, label_node),
        GraphComponent[Any, tuple[float, str]],
    )

    @component()
    def reduce_graph_values(values: Sequence[float]) -> float:
        return sum(values)

    @component()
    def times_two(value: float) -> float:
        return 2 * value

    float_graph_builder = GraphBuilder[float]()
    first_float = float_graph_builder.call(times_two, float_graph_builder.input)
    second_float = float_graph_builder.call(times_two, first_float)
    third_float = float_graph_builder.call(times_two, first_float)
    reduced_node = float_graph_builder.call(
        reduce_graph_values,
        third_float,
        second_float,
        first_float,
    )
    assert_type(reduced_node, NodeRef[float])
    assert_type(
        float_graph_builder.output(
            reduced_node,
            reduced_node,
            float_graph_builder.input,
        ),
        GraphComponent[float, tuple[float, float, float]],
    )

    typed_graph_builder = GraphBuilder[Parameters]()
    typed_number_node = typed_graph_builder.call(
        wrapped_objective,
        typed_graph_builder.input,
    )
    assert_type(typed_number_node, NodeRef[float])
    assert_type(
        typed_graph_builder.output(typed_number_node),
        GraphComponent[Parameters, float],
    )
    assert_type(
        typed_graph_builder.output(typed_graph_builder.input),
        GraphComponent[Parameters, Parameters],
    )

    @component()
    def text_length(value: str) -> int:
        return len(value)

    typed_graph_builder.call(
        text_length,
        typed_graph_builder.input,  # pyright: ignore[reportArgumentType]
    )
    graph_builder.call(  # pyright: ignore[reportCallIssue]
        combine_graph_values,
        number_node,
        number_node,  # pyright: ignore[reportArgumentType]
    )

    dynamic = Fitter(lambda parameters: parameters["x"], {"x": 1.0})
    assert_type(dynamic, Fitter[dict[str, Any]])
