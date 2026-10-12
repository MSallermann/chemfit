"""Behavioral tests for SerialGraphSchedule and structural exception propagation."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from loky import ProcessPoolExecutor

from chemfit.abstract_objective_function import (
    NO_OUTPUT,
    EvaluateContext,
    PipelineComponent,
    component,
    quantity,
)
from chemfit.combined_objective_function import (
    CombinedObjectiveFunction,
    skip_exception_handler,
)
from chemfit.executor_graph_schedule import ExecutorGraphScheduler
from chemfit.graph_schedule_base import SerialGraphScheduler
from chemfit.scheduling import EvaluationRequest


def evaluate_graph(objective, value):
    ctx = EvaluateContext()
    # result = SerialGraphScheduler().prepare(objective)(value, ctx)
    # result = ExecutorGraphScheduler(executor_factory=ThreadPoolExecutor).prepare(
    #     objective
    # )(value, ctx)
    result = ExecutorGraphScheduler(executor_factory=ProcessPoolExecutor).prepare(
        objective
    )(value, ctx)

    return result, ctx


def test_successful_nested_pipeline_and_join():
    """Graph and direct evaluation agree on values and child contexts."""

    @component()
    def double(x):
        return 2 * x

    @component()
    def add_three(x):
        return x + 3

    @quantity()
    def measure(x, scale):
        return {"value": x * scale}

    cob = CombinedObjectiveFunction(
        [
            measure.bind(scale=1).with_loss(lambda q: q["value"]),
            measure.bind(scale=2).with_loss(lambda q: q["value"]),
        ]
    )

    @component()
    def subtract_one(x):
        return x - 1

    objective = PipelineComponent(double, add_three).then(cob).then(subtract_one)

    direct_ctx = EvaluateContext()
    direct_result = objective(2, direct_ctx)
    graph_result, graph_ctx = evaluate_graph(objective, 2)

    assert direct_result == graph_result == 20
    assert [ctx.output for ctx in direct_ctx.incoming] == [4, 7, 21, 20]
    assert [ctx.output for ctx in graph_ctx.incoming] == [4, 7, 21, 20]
    assert [ctx.output for ctx in graph_ctx.incoming[2].incoming] == [7, 14]
    assert graph_ctx.output == 20


def test_failed_atomic_stage_skips_downstream_stage():
    @component()
    def fail(x):
        raise ValueError("atomic failure")

    @component()
    def must_not_run(x):
        raise AssertionError("downstream stage should have been skipped")

    objective = PipelineComponent(fail, must_not_run)
    ctx = EvaluateContext()

    with pytest.raises(ValueError, match="atomic failure"):
        SerialGraphScheduler().prepare(objective)(3, ctx)

    assert isinstance(ctx.incoming[0].exception, ValueError)
    assert str(ctx.incoming[0].exception) == "atomic failure"
    assert ctx.incoming[1].output is NO_OUTPUT
    assert ctx.incoming[1].exception is None


def test_join_recovers_from_one_failed_term_and_runs_sibling():
    @component()
    def fail(x):
        raise ValueError("term failure")

    @component()
    def succeed(x):
        return 2 * x

    objective = CombinedObjectiveFunction([fail, succeed]).with_exception_handler(
        skip_exception_handler
    )

    result, ctx = evaluate_graph(objective, 4)

    assert result == 8
    assert isinstance(ctx.incoming[0].exception, ValueError)
    assert str(ctx.incoming[0].exception) == "term failure"
    assert ctx.incoming[1].output == 8
    assert ctx.incoming[1].exception is None


def test_failed_structural_begin_aborts_only_its_own_scope():
    # Only structural hooks run in the coordinator process.
    calls = []

    @component()
    def first_child(x):
        raise AssertionError("first child should have been skipped")

    @component()
    def second_child(x):
        raise AssertionError("second child should have been skipped")

    class FailingStartPipeline(PipelineComponent):
        def _begin_evaluation(self, value, ctx):
            super()._begin_evaluation(value, ctx)
            raise RuntimeError("begin failure")

    failed_branch = FailingStartPipeline(first_child, second_child)
    failed_branch.register_eval_hook(
        pre=lambda ctx: calls.append("begin hook"),
        post=lambda ctx: calls.append("end hook"),
    )

    @component()
    def healthy_sibling(x):
        return 5

    objective = CombinedObjectiveFunction(
        [failed_branch, healthy_sibling]
    ).with_exception_handler(skip_exception_handler)

    result, ctx = evaluate_graph(objective, 2)

    assert result == 5
    assert calls == ["begin hook", "end hook"]
    assert isinstance(ctx.incoming[0].exception, RuntimeError)
    assert str(ctx.incoming[0].exception) == "begin failure"
    assert ctx.incoming[0].incoming == ()  # No child evaluations started.
    assert ctx.incoming[1].output == 5
    assert ctx.incoming[1].exception is None


def test_failed_input_skips_entire_nested_structural_component():
    # Structural hooks run in the coordinator; a skipped scope invokes neither.
    calls = []

    @component()
    def fail(x):
        raise ValueError("upstream failure")

    @component()
    def inner_first(x):
        raise AssertionError("inner first should have been skipped")

    @component()
    def inner_second(x):
        raise AssertionError("inner second should have been skipped")

    inner = PipelineComponent(inner_first, inner_second)
    inner.register_eval_hook(
        pre=lambda ctx: calls.append("inner pre"),
        post=lambda ctx: calls.append("inner post"),
    )
    objective = PipelineComponent(fail, inner)
    ctx = EvaluateContext()

    with pytest.raises(ValueError, match="upstream failure"):
        SerialGraphScheduler().prepare(objective)(2, ctx)

    assert isinstance(ctx.incoming[0].exception, ValueError)
    assert ctx.incoming[1].output is NO_OUTPUT
    assert ctx.incoming[1].exception is None
    assert ctx.incoming[1].incoming == ()
    assert calls == []


def test_structural_end_failure_is_reported_without_aborting_siblings():
    calls = []

    @component()
    def first(x):
        return x + 1

    @component()
    def second(x):
        return 2 * x

    failed_branch = PipelineComponent(first, second)

    def fail_on_end(ctx):
        calls.append("failed branch post")
        raise RuntimeError("end failure")

    failed_branch.register_eval_hook(post=fail_on_end)

    @component()
    def healthy_sibling(x):
        return 7

    objective = CombinedObjectiveFunction(
        [failed_branch, healthy_sibling]
    ).with_exception_handler(skip_exception_handler)

    result, ctx = evaluate_graph(objective, 2)

    assert result == 7
    assert calls == ["failed branch post"]
    assert [child.output for child in ctx.incoming[0].incoming] == [3, 6]
    # A failed post-hook does not rewrite the successful evaluation context.
    assert ctx.incoming[0].output == 6
    assert ctx.incoming[0].exception is None
    assert ctx.incoming[1].output == 7
    assert ctx.incoming[1].exception is None


def test_batch_mixed_success_and_failure_keeps_request_indices():
    """An unhandled failure in one request does not prevent other results."""

    @component()
    def possibly_fail(x):
        if x < 0:
            raise ValueError(f"bad input: {x}")
        return 2 * x

    @component()
    def downstream(x):
        return x + 1

    schedule = SerialGraphScheduler().prepare(
        PipelineComponent(possibly_fail, downstream)
    )
    inputs = [2, -1, 5, -3]
    contexts = [EvaluateContext() for _ in inputs]
    requests = [
        EvaluationRequest(parameters=value, ctx=ctx)
        for value, ctx in zip(inputs, contexts, strict=True)
    ]

    results = list(schedule.evaluate_many(requests))
    assert len(results) == len(inputs)
    assert {result.index for result in results} == set(range(len(inputs)))

    by_index = {result.index: result.value for result in results}
    assert by_index[0] == 5
    assert by_index[2] == 11
    for index, value in [(1, -1), (3, -3)]:
        assert isinstance(by_index[index], ValueError)
        assert str(by_index[index]) == f"bad input: {value}"

    assert [ctx.input for ctx in contexts] == inputs
    for index, expected_first, expected_result in [(0, 4, 5), (2, 10, 11)]:
        assert contexts[index].incoming[0].output == expected_first
        assert contexts[index].incoming[1].output == expected_result
        assert contexts[index].output == expected_result

    for index in (1, 3):
        ctx = contexts[index]
        assert isinstance(ctx.incoming[0].exception, ValueError)
        assert ctx.incoming[1].output is NO_OUTPUT
        assert ctx.incoming[1].exception is None


def test_batch_recovered_failures_are_isolated_per_request():
    """A COB recovers failures independently for each request in the batch."""

    @component()
    def sometimes_fail(x):
        if x % 2 == 0:
            raise ValueError(f"failed: {x}")
        return x

    @component()
    def healthy_sibling(x):
        return 2 * x

    objective = CombinedObjectiveFunction(
        [sometimes_fail, healthy_sibling]
    ).with_exception_handler(skip_exception_handler)
    schedule = SerialGraphScheduler().prepare(objective)

    inputs = [1, 2, 3, 4]
    contexts = [EvaluateContext() for _ in inputs]
    results = list(
        schedule.evaluate_many(
            [
                EvaluationRequest(parameters=value, ctx=ctx)
                for value, ctx in zip(inputs, contexts, strict=True)
            ]
        )
    )

    assert len(results) == len(inputs)
    assert {result.index for result in results} == set(range(len(inputs)))
    assert {result.index: result.value for result in results} == {
        0: 3,
        1: 4,
        2: 9,
        3: 8,
    }

    for index, (value, ctx) in enumerate(zip(inputs, contexts, strict=True)):
        assert ctx.output == {0: 3, 1: 4, 2: 9, 3: 8}[index]
        assert ctx.incoming[1].output == 2 * value
        assert ctx.incoming[1].exception is None
        if value % 2 == 0:
            assert isinstance(ctx.incoming[0].exception, ValueError)
            assert str(ctx.incoming[0].exception) == f"failed: {value}"
        else:
            assert ctx.incoming[0].output == value
            assert ctx.incoming[0].exception is None
