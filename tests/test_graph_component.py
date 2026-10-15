"""Focused tests for user-built structural graphs."""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from loky import ProcessPoolExecutor

from chemfit.abstract_objective_function import (
    NO_OUTPUT,
    PARENT_INPUT,
    EvaluateContext,
    ParentInput_,
    Skipped,
    component,
)
from chemfit.api import graph, join
from chemfit.executor_graph_schedule import ExecutorGraphScheduler
from chemfit.graph_component import NodeRef
from chemfit.graph_schedule_base import SerialGraphScheduler

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


def evaluate_scheduled(scheduler, objective, value):
    ctx = EvaluateContext()
    schedule = scheduler.prepare(objective)
    if getattr(schedule, "rank", 0) != 0:
        with schedule:
            schedule.worker_loop()
        pytest.skip("MPI worker rank")

    with schedule:
        result = schedule(value, ctx)
    return result, ctx


def test_shared_intermediate_is_evaluated_once_for_multiple_outputs(scheduler):
    @component()
    def unique_token(_value):
        return uuid.uuid4().hex

    @component()
    def consume(token):
        return token

    builder = graph()
    shared = builder.call(unique_token, builder.input)
    left = builder.call(consume, shared)
    right = builder.call(consume, shared)
    objective = builder.output(left, right)

    direct_result = objective(1)
    scheduled_result, _ = evaluate_scheduled(scheduler, objective, 1)

    assert direct_result[0] == direct_result[1]
    assert scheduled_result[0] == scheduled_result[1]


def test_explicit_dependency_orders_work_without_passing_its_result(
    scheduler, tmp_path
):
    marker = tmp_path / "dependency-finished"

    @component()
    def first(path):
        time.sleep(0.05)
        Path(path).touch()
        return "unused"

    @component()
    def second(path):
        if not Path(path).exists():
            msg = "ordering dependency did not finish"
            raise RuntimeError(msg)
        return Path(path).name

    builder = graph()
    first_ref = builder.call(first, builder.input)
    second_ref = builder.call(second, builder.input)
    builder.depends_on(second_ref, first_ref)
    objective = builder.output(second_ref)

    assert objective.component_dependencies() == ((), (0,))
    assert objective.component_inputs() == ((PARENT_INPUT,), (PARENT_INPUT,))
    assert objective(marker) == marker.name
    marker.unlink()

    result, _ = evaluate_scheduled(scheduler, objective, marker)
    assert result == marker.name


def test_nested_structural_components_match_direct_evaluation(scheduler):
    @component()
    def add_one(value):
        return value + 1

    @component()
    def double(value):
        return value * 2

    @component()
    def total(values):
        return sum(values)

    @component()
    def negate(value):
        return -value

    inner_builder = graph()
    branches = inner_builder.call(join(add_one, double), inner_builder.input)
    inner_total = inner_builder.call(total, branches)
    inner = inner_builder.output(inner_total)

    outer_builder = graph()
    nested = outer_builder.call(inner, outer_builder.input)
    result_ref = outer_builder.call(negate, nested)
    objective = outer_builder.output(result_ref)

    direct_ctx = EvaluateContext()
    direct_result = objective(3, direct_ctx)
    scheduled_result, scheduled_ctx = evaluate_scheduled(scheduler, objective, 3)

    assert direct_result == scheduled_result == -10
    assert direct_ctx.to_summary(recursive=True) == scheduled_ctx.to_summary(
        recursive=True
    )
    assert "chemfit.structure" in direct_ctx.meta
    assert "chemfit.structure" in direct_ctx.children[0].meta
    assert "chemfit.structure" in direct_ctx.children[0].children[0].meta
    json.dumps(direct_ctx.to_summary(recursive=True))


def failure_graphs():
    @component()
    def fail(_value):
        msg = "boom"
        raise LookupError(msg)

    @component()
    def should_not_run(_value):
        msg = "skipped component ran"
        raise AssertionError(msg)

    builder = graph()
    failed = builder.call(fail, builder.input)
    skipped = builder.call(should_not_run, failed)
    return builder.output(skipped), builder.output(failed, skipped)


def test_single_output_raises_upstream_failure(scheduler):
    single_output, _ = failure_graphs()

    with pytest.raises(LookupError, match="boom"):
        single_output(1)
    with pytest.raises(LookupError, match="boom"):
        evaluate_scheduled(scheduler, single_output, 1)


def test_multiple_outputs_preserve_failures_and_skips(scheduler):
    _, multiple_outputs = failure_graphs()

    direct_ctx = EvaluateContext()
    direct_result = multiple_outputs(1, direct_ctx)
    scheduled_result, scheduled_ctx = evaluate_scheduled(scheduler, multiple_outputs, 1)

    for result in (direct_result, scheduled_result):
        assert isinstance(result[0], LookupError)
        assert str(result[0]) == "boom"
        assert isinstance(result[1], Skipped)
        assert isinstance(result[1].reason, LookupError)
        assert str(result[1].reason) == "boom"
    assert direct_ctx.children[1].output is NO_OUTPUT
    assert scheduled_ctx.children[1].output is NO_OUTPUT


def test_foreign_and_invalid_node_references_are_rejected():
    @component()
    def identity(value):
        return value

    first_builder = graph()
    foreign = first_builder.call(identity, first_builder.input)

    second_builder = graph()
    local = second_builder.call(identity, second_builder.input)

    with pytest.raises(ValueError, match="another graph"):
        second_builder.call(identity, foreign)
    with pytest.raises(ValueError, match="another graph"):
        second_builder.output(foreign)
    with pytest.raises(ValueError, match="another graph"):
        second_builder.depends_on(local, foreign)

    later = second_builder.call(identity, second_builder.input)
    with pytest.raises(ValueError, match="defined before"):
        second_builder.depends_on(local, later)
    with pytest.raises(ValueError, match="defined before"):
        second_builder.depends_on(local, local)

    invalid = NodeRef(identity, [], [], -100)
    with pytest.raises(ValueError, match="another graph"):
        second_builder.output(invalid)


@pytest.mark.parametrize("invalid", [None, object(), ParentInput_()])
def test_unsupported_graph_references_are_rejected(invalid):
    @component()
    def identity(value):
        return value

    builder = graph()

    with pytest.raises(TypeError, match="builder.input or a NodeRef"):
        builder.call(identity, invalid)
    with pytest.raises(TypeError, match="builder.input or a NodeRef"):
        builder.output(invalid)


def test_output_snapshots_topology_and_keeps_inputs_separate_from_dependencies():
    @component()
    def identity(value):
        return value

    @component()
    def pair(values):
        return values

    builder = graph()
    shared = builder.call(identity, builder.input)
    repeated_input = builder.call(pair, shared, shared)
    snapshot = builder.output(repeated_input)

    assert snapshot.components()[0] is identity
    identity.static_meta_data["after_snapshot"] = True
    assert snapshot.components()[0].static_meta_data["after_snapshot"] is True

    ordered = builder.call(identity, builder.input)
    builder.depends_on(ordered, repeated_input)
    extended = builder.output(ordered)

    assert len(snapshot.components()) == 2
    assert snapshot.component_dependencies() == ((), (0,))
    assert snapshot.component_inputs() == ((PARENT_INPUT,), (0, 0))
    assert snapshot.combine_inputs() == (1,)

    assert len(extended.components()) == 3
    assert extended.component_dependencies() == ((), (0,), (1,))
    assert extended.component_inputs() == (
        (PARENT_INPUT,),
        (0, 0),
        (PARENT_INPUT,),
    )
    assert extended.combine_inputs() == (2,)

    ctx = EvaluateContext()
    assert extended(3, ctx) == 3
    assert ctx.meta["chemfit.structure"] == {
        "children": [
            {"index": index, "name": name}
            for index, name in enumerate(("identity", "pair", "identity"))
        ],
        "inputs": [
            [{"kind": "parent_input"}],
            [{"kind": "child", "index": 0}, {"kind": "child", "index": 0}],
            [{"kind": "parent_input"}],
        ],
        "dependencies": [[], [0], [1]],
        "outputs": [{"kind": "child", "index": 2}],
    }

    ctx.meta["chemfit.structure"]["children"].clear()
    assert extended(4, ctx) == 4
    assert len(ctx.meta["chemfit.structure"]["children"]) == 3
    assert len(ctx.children) == 3
