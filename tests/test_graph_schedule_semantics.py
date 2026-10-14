"""Behavioral tests for SerialGraphSchedule and structural exception propagation."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from loky import ProcessPoolExecutor

from chemfit.abstract_objective_function import (
    NO_OUTPUT,
    PARENT_INPUT,
    SKIPPED,
    EvaluateContext,
    OpaquePipeline,
    PipelineComponent,
    Skipped,
    StructuralComponent,
    component,
    quantity,
)
from chemfit.api import join
from chemfit.callgraph import AtomicNode, call_graph_to_dot, objective_to_call_graph
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
    assert [ctx.output for ctx in direct_ctx.children] == [4, 7, 21, 20]
    assert [ctx.output for ctx in graph_ctx.children] == [4, 7, 21, 20]
    assert [ctx.output for ctx in graph_ctx.children[2].children] == [7, 14]
    for ctx in (direct_ctx, graph_ctx):
        terms = ctx.children[2].children
        assert [stage.output for stage in terms[0].children] == [
            {"value": 7},
            7,
        ]
        assert [stage.output for stage in terms[1].children] == [
            {"value": 14},
            14,
        ]
    assert graph_ctx.output == 20


def test_nested_pipeline_and_join_support_repeated_component_instances(scheduler):
    @component()
    def double(value):
        return 2 * value

    @component()
    def add_one(value):
        return value + 1

    @component()
    def total(values):
        return sum(values)

    repeated_join = join(double, double, add_one, add_one)
    branch = PipelineComponent(repeated_join, total)
    objective = join(branch, branch)

    direct_ctx = EvaluateContext()
    direct_result = objective(2, direct_ctx)
    graph_result, graph_ctx = evaluate_graph(scheduler, objective, 2)

    assert direct_result == graph_result == (14, 14)
    assert [child.output for child in graph_ctx.children] == [14, 14]

    occurrence_contexts = []
    for branch_ctx in graph_ctx.children:
        repeated_join_ctx = branch_ctx.children[0]
        assert [ctx.output for ctx in repeated_join_ctx.children] == [4, 4, 3, 3]
        occurrence_contexts.extend(repeated_join_ctx.children)

    assert len({id(ctx) for ctx in occurrence_contexts}) == 8


def test_structural_with_loss_remains_graph_visible(scheduler):
    @component()
    def add_one(value):
        return value + 1

    @component()
    def double(value):
        return 2 * value

    joined = join(add_one, double).with_meta(scope="joined")

    def mark_post_hook(ctx):
        ctx.meta["post_hook_ran"] = True

    joined.register_eval_hook(post=mark_post_hook)

    def loss(values, *, offset):
        return sum(values) + offset

    objective = joined.with_loss(loss, offset=1)
    assert isinstance(objective, PipelineComponent)

    graph = objective_to_call_graph(objective)
    assert sum(isinstance(node, AtomicNode) for node in graph.nodes) == 3

    direct_ctx = EvaluateContext()
    direct_result = objective(2, direct_ctx)
    graph_result, graph_ctx = evaluate_graph(scheduler, objective, 2)

    assert direct_result == graph_result == 8
    assert [ctx.output for ctx in graph_ctx.children] == [(3, 4), 8]
    assert [ctx.output for ctx in graph_ctx.children[0].children] == [3, 4]
    assert graph_ctx.children[0].meta == {
        "scope": "joined",
        "post_hook_ran": True,
    }


def test_atomic_with_loss_remains_opaque(scheduler):
    @component()
    def add_one(value):
        return value + 1

    atomic_objective = add_one.with_loss(
        lambda value, *, multiplier: value * multiplier,
        multiplier=3,
    )
    assert isinstance(atomic_objective, OpaquePipeline)
    atomic_graph = objective_to_call_graph(atomic_objective)
    assert len(atomic_graph.nodes) == 1
    assert isinstance(atomic_graph.nodes[0], AtomicNode)

    atomic_direct_ctx = EvaluateContext()
    atomic_direct_result = atomic_objective(2, atomic_direct_ctx)
    atomic_graph_result, atomic_graph_ctx = evaluate_graph(
        scheduler, atomic_objective, 2
    )

    assert atomic_direct_result == atomic_graph_result == 9
    assert [ctx.output for ctx in atomic_direct_ctx.children] == [3, 9]
    assert [ctx.output for ctx in atomic_graph_ctx.children] == [3, 9]


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

    assert isinstance(ctx.children[0].output, ValueError)
    assert str(ctx.children[0].output) == "atomic failure"
    assert ctx.children[1].output is NO_OUTPUT


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
    assert isinstance(ctx.children[0].output, ValueError)
    assert str(ctx.children[0].output) == "term failure"
    assert ctx.children[1].output == 8


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
    assert isinstance(ctx.children[0].output, RuntimeError)
    assert str(ctx.children[0].output) == "begin failure"
    assert ctx.children[0].children == []  # No child evaluations started.
    assert ctx.children[1].output == 5


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

    assert isinstance(ctx.children[0].output, ValueError)
    assert ctx.children[1].output is NO_OUTPUT
    assert ctx.children[1].children == []
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
    assert [child.output for child in ctx.children[0].children] == [3, 6]
    # A failed post-hook becomes the finalized outcome.
    assert isinstance(ctx.children[0].output, RuntimeError)
    assert str(ctx.children[0].output) == "end failure"
    assert ctx.children[1].output == 7


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
    assert direct_ctx.children == graph_ctx.children == []
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
    assert [child.output for child in ctx.children] == [6, 7]


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
    assert [child.output for child in ctx.children] == [NO_OUTPUT, NO_OUTPUT]


def test_custom_input_policy_skips_subtree_and_unlocks_dependency_chain(scheduler):
    @component()
    def fail(value):
        msg = f"cannot evaluate {value}"
        raise ValueError(msg)

    @component()
    def must_not_run(_value):
        msg = "skipped subtree executed"
        raise AssertionError(msg)

    @component()
    def continue_after_skip(value):
        return value + 10

    skipped_branch = PipelineComponent(must_not_run, must_not_run)

    class SkipFailedBranch(StructuralComponent):
        def components(self):
            return (fail, skipped_branch, continue_after_skip)

        def component_dependencies(self):
            return ((), (0,), (1,))

        def component_inputs(self):
            return ((PARENT_INPUT,), (0,), (PARENT_INPUT,))

        def resolve_child_input(self, index, inputs, ctx):
            if index == 1 and isinstance(inputs[0], ValueError):
                ctx.meta["skipped_failed_branch"] = True
                return Skipped(reason=inputs[0])
            return super().resolve_child_input(index, inputs, ctx)

        def combine_inputs(self):
            return (2,)

        def combine_outcomes(self, outcomes, _ctx):
            return outcomes[0]

    objective = SkipFailedBranch()
    direct_ctx = EvaluateContext()
    direct_result = objective(5, direct_ctx)
    graph_result, graph_ctx = evaluate_graph(scheduler, objective, 5)

    assert direct_result == graph_result == 15
    assert direct_ctx.meta == graph_ctx.meta == {"skipped_failed_branch": True}
    assert isinstance(graph_ctx.children[0].output, ValueError)
    assert graph_ctx.children[1].output is NO_OUTPUT
    assert graph_ctx.children[1].children == []
    assert graph_ctx.children[2].output == 15


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
    assert [child.output for child in ctx.children] == [4, 6]


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
        assert contexts[index].children[0].output == expected_first
        assert contexts[index].children[1].output == expected_result
        assert contexts[index].output == expected_result

    for index in (1, 3):
        ctx = contexts[index]
        assert isinstance(ctx.children[0].output, ValueError)
        assert ctx.children[1].output is NO_OUTPUT


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
        assert ctx.children[1].output == 2 * value
        if value % 2 == 0:
            assert isinstance(ctx.children[0].output, ValueError)
            assert str(ctx.children[0].output) == f"failed: {value}"
        else:
            assert ctx.children[0].output == value
