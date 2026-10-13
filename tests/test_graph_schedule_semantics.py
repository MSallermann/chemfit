"""Behavioral tests for SerialGraphSchedule and structural exception propagation."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from loky import ProcessPoolExecutor

from chemfit.abstract_objective_function import (
    NO_OUTPUT,
    PARENT_INPUT,
    SKIPPED,
    EvaluateContext,
    PipelineComponent,
    Skipped,
    StructuralComponent,
    component,
    quantity,
)
from chemfit.callgraph import call_graph_to_dot, objective_to_call_graph
from chemfit.combined_objective_function import (
    CombinedObjectiveFunction,
    skip_exception_handler,
)
from chemfit.executor_graph_schedule import ExecutorGraphScheduler
from chemfit.graph_schedule_base import SerialGraphScheduler
from chemfit.scheduling import EvaluationRequest

try:
    from mpi4py import MPI

    from chemfit.mpi_graph_schedule import MPIGraphScheduler
except ImportError:
    MPI = None


LOCAL_SCHEDULERS = (
    pytest.param(SerialGraphScheduler, id="serial"),
    pytest.param(
        lambda: ExecutorGraphScheduler(executor_factory=ThreadPoolExecutor),
        id="thread-pool",
    ),
    pytest.param(
        lambda: ExecutorGraphScheduler(executor_factory=ProcessPoolExecutor),
        id="process-pool",
    ),
)
SCHEDULERS = (
    (pytest.param(MPIGraphScheduler, id="mpi"),)
    if MPI is not None and MPI.COMM_WORLD.Get_size() > 1
    else LOCAL_SCHEDULERS
)


@pytest.fixture(params=SCHEDULERS)
def scheduler(request):
    return request.param()


def prepare_schedule(scheduler, objective):
    schedule = scheduler.prepare(objective)
    if getattr(schedule, "rank", 0) != 0:
        with schedule:
            schedule.worker_loop()
        pytest.skip("MPI worker rank")
    return schedule


def evaluate_graph(scheduler, objective, value):
    ctx = EvaluateContext()
    with prepare_schedule(scheduler, objective) as schedule:
        result = schedule(value, ctx)

    return result, ctx


def test_successful_nested_pipeline_and_join(scheduler):
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
    graph_result, graph_ctx = evaluate_graph(scheduler, objective, 2)

    assert direct_result == graph_result == 20
    assert [ctx.output for ctx in direct_ctx.incoming] == [4, 7, 21, 20]
    assert [ctx.output for ctx in graph_ctx.incoming] == [4, 7, 21, 20]
    assert [ctx.output for ctx in graph_ctx.incoming[2].incoming] == [7, 14]
    assert graph_ctx.output == 20


def test_failed_atomic_stage_skips_downstream_stage(scheduler):
    @component()
    def fail(_x):
        msg = "atomic failure"
        raise ValueError(msg)

    @component()
    def must_not_run(_x):
        msg = "downstream stage should have been skipped"
        raise AssertionError(msg)

    objective = PipelineComponent(fail, must_not_run)
    ctx = EvaluateContext()

    with (
        prepare_schedule(scheduler, objective) as schedule,
        pytest.raises(ValueError, match="atomic failure"),
    ):
        schedule(3, ctx)

    assert isinstance(ctx.incoming[0].output, ValueError)
    assert str(ctx.incoming[0].output) == "atomic failure"
    assert ctx.incoming[1].output is NO_OUTPUT


def test_join_recovers_from_one_failed_term_and_runs_sibling(scheduler):
    @component()
    def fail(_x):
        msg = "term failure"
        raise ValueError(msg)

    @component()
    def succeed(x):
        return 2 * x

    objective = CombinedObjectiveFunction([fail, succeed]).with_exception_handler(
        skip_exception_handler
    )

    result, ctx = evaluate_graph(scheduler, objective, 4)

    assert result == 8
    assert isinstance(ctx.incoming[0].output, ValueError)
    assert str(ctx.incoming[0].output) == "term failure"
    assert ctx.incoming[1].output == 8


def test_failed_structural_begin_aborts_only_its_own_scope(scheduler):
    # Only structural hooks run in the coordinator process.
    calls = []

    @component()
    def first_child(_x):
        msg = "first child should have been skipped"
        raise AssertionError(msg)

    @component()
    def second_child(_x):
        msg = "second child should have been skipped"
        raise AssertionError(msg)

    failed_branch = PipelineComponent(first_child, second_child)

    def fail_pre_hook(_ctx):
        calls.append("begin hook")
        msg = "begin failure"
        raise RuntimeError(msg)

    failed_branch.register_eval_hook(
        pre=fail_pre_hook, post=lambda _ctx: calls.append("end hook")
    )

    @component()
    def healthy_sibling(_x):
        return 5

    objective = CombinedObjectiveFunction(
        [failed_branch, healthy_sibling]
    ).with_exception_handler(skip_exception_handler)

    result, ctx = evaluate_graph(scheduler, objective, 2)

    assert result == 5
    assert calls == ["begin hook", "end hook"]
    assert isinstance(ctx.incoming[0].output, RuntimeError)
    assert str(ctx.incoming[0].output) == "begin failure"
    assert ctx.incoming[0].incoming == ()  # No child evaluations started.
    assert ctx.incoming[1].output == 5


def test_failed_input_skips_entire_nested_structural_component(scheduler):
    # Structural hooks run in the coordinator; a skipped scope invokes neither.
    calls = []

    @component()
    def fail(_x):
        msg = "upstream failure"
        raise ValueError(msg)

    @component()
    def inner_first(_x):
        msg = "inner first should have been skipped"
        raise AssertionError(msg)

    @component()
    def inner_second(_x):
        msg = "inner second should have been skipped"
        raise AssertionError(msg)

    inner = PipelineComponent(inner_first, inner_second)
    inner.register_eval_hook(
        pre=lambda _ctx: calls.append("inner pre"),
        post=lambda _ctx: calls.append("inner post"),
    )
    objective = PipelineComponent(fail, inner)
    ctx = EvaluateContext()

    with (
        prepare_schedule(scheduler, objective) as schedule,
        pytest.raises(ValueError, match="upstream failure"),
    ):
        schedule(2, ctx)

    assert isinstance(ctx.incoming[0].output, ValueError)
    assert ctx.incoming[1].output is NO_OUTPUT
    assert ctx.incoming[1].incoming == ()
    assert calls == []


def test_structural_end_failure_is_reported_without_aborting_siblings(scheduler):
    calls = []

    @component()
    def first(x):
        return x + 1

    @component()
    def second(x):
        return 2 * x

    failed_branch = PipelineComponent(first, second)

    def fail_on_end(_ctx):
        calls.append("failed branch post")
        msg = "end failure"
        raise RuntimeError(msg)

    failed_branch.register_eval_hook(post=fail_on_end)

    @component()
    def healthy_sibling(_x):
        return 7

    objective = CombinedObjectiveFunction(
        [failed_branch, healthy_sibling]
    ).with_exception_handler(skip_exception_handler)

    result, ctx = evaluate_graph(scheduler, objective, 2)

    assert result == 7
    assert calls == ["failed branch post"]
    assert [child.output for child in ctx.incoming[0].incoming] == [3, 6]
    # A failed post-hook becomes the finalized outcome.
    assert isinstance(ctx.incoming[0].output, RuntimeError)
    assert str(ctx.incoming[0].output) == "end failure"
    assert ctx.incoming[1].output == 7


def test_pre_hook_short_circuit_and_post_hook_transform_match_direct_call(scheduler):
    @component()
    def must_not_run(value):
        msg = f"child unexpectedly received {value}"
        raise AssertionError(msg)

    objective = PipelineComponent(must_not_run, must_not_run)
    objective.register_eval_hook(pre=lambda _ctx: 4)
    objective.register_eval_hook(post=lambda ctx: setattr(ctx, "output", 9))

    direct_ctx = EvaluateContext()
    graph_result, graph_ctx = evaluate_graph(scheduler, objective, 2)

    assert objective(2, direct_ctx) == graph_result == 9
    assert direct_ctx.incoming == graph_ctx.incoming == ()
    assert graph_ctx.output == 9


def test_atomic_post_hook_recovery_is_used_by_downstream_component(scheduler):
    @component()
    def first(value):
        msg = f"failed with {value}"
        raise ValueError(msg)

    first.register_eval_hook(post=lambda ctx: setattr(ctx, "output", 3 * ctx.input))

    @component()
    def second(value):
        return value + 1

    objective = PipelineComponent(first, second)
    direct_result = objective(2)
    graph_result, ctx = evaluate_graph(scheduler, objective, 2)

    assert direct_result == graph_result == 7
    assert [child.output for child in ctx.incoming] == [6, 7]


def test_literal_and_resolution_skips_reach_structural_reducer(scheduler):
    calls = []

    @component()
    def must_not_run(value):
        calls.append(value)
        return value

    class SkipPolicy(StructuralComponent):
        def components(self):
            return (must_not_run, must_not_run)

        def component_dependencies(self):
            return ((), ())

        def component_inputs(self):
            return ((SKIPPED,), (PARENT_INPUT,))

        def resolve_child_input(self, index, inputs, ctx):
            if index == 1:
                msg = "resolution failed"
                raise ValueError(msg)
            return super().resolve_child_input(index, inputs, ctx)

        def combine_inputs(self):
            return (0, 1, SKIPPED)

        def combine_outcomes(self, outcomes, _ctx):
            assert all(isinstance(outcome, Skipped) for outcome in outcomes)
            return tuple(
                None if outcome.reason is None else str(outcome.reason)
                for outcome in outcomes
            )

    objective = SkipPolicy()
    assert "Skipped" in call_graph_to_dot(objective_to_call_graph(objective))
    direct_result = objective(3)
    graph_result, ctx = evaluate_graph(scheduler, objective, 3)

    assert direct_result == graph_result == (None, "resolution failed", None)
    assert calls == []
    assert [child.output for child in ctx.incoming] == [NO_OUTPUT, NO_OUTPUT]


def test_finish_waits_for_children_not_selected_by_combine_inputs(scheduler):
    @component()
    def selected(value):
        return value + 1

    @component()
    def unselected(value):
        return value + 2

    class SelectFirst(StructuralComponent):
        def components(self):
            return (selected, unselected)

        def component_dependencies(self):
            return ((), ())

        def component_inputs(self):
            # The data reference must create an ordering edge even though it
            # is intentionally omitted from component_dependencies().
            return ((PARENT_INPUT,), (0,))

        def combine_inputs(self):
            return (0,)

        def combine_outcomes(self, outcomes, _ctx):
            return outcomes[0]

    result, ctx = evaluate_graph(scheduler, SelectFirst(), 3)

    assert result == 4
    assert [child.output for child in ctx.incoming] == [4, 6]


def test_batch_mixed_success_and_failure_keeps_request_indices(scheduler):
    """An unhandled failure in one request does not prevent other results."""

    @component()
    def possibly_fail(x):
        if x < 0:
            msg = f"bad input: {x}"
            raise ValueError(msg)
        return 2 * x

    @component()
    def downstream(x):
        return x + 1

    schedule = prepare_schedule(scheduler, PipelineComponent(possibly_fail, downstream))
    inputs = [2, -1, 5, -3]
    contexts = [EvaluateContext() for _ in inputs]
    requests = [
        EvaluationRequest(parameters=value, ctx=ctx)
        for value, ctx in zip(inputs, contexts, strict=True)
    ]

    with schedule:
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
        assert isinstance(ctx.incoming[0].output, ValueError)
        assert ctx.incoming[1].output is NO_OUTPUT


def test_batch_recovered_failures_are_isolated_per_request(scheduler):
    """A COB recovers failures independently for each request in the batch."""

    @component()
    def sometimes_fail(x):
        if x % 2 == 0:
            msg = f"failed: {x}"
            raise ValueError(msg)
        return x

    @component()
    def healthy_sibling(x):
        return 2 * x

    objective = CombinedObjectiveFunction(
        [sometimes_fail, healthy_sibling]
    ).with_exception_handler(skip_exception_handler)
    schedule = prepare_schedule(scheduler, objective)

    inputs = [1, 2, 3, 4]
    contexts = [EvaluateContext() for _ in inputs]
    with schedule:
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
        if value % 2 == 0:
            assert isinstance(ctx.incoming[0].output, ValueError)
            assert str(ctx.incoming[0].output) == f"failed: {value}"
        else:
            assert ctx.incoming[0].output == value
