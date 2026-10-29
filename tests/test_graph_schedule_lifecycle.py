import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from chemfit.abstract_objective_function import (
    PARENT_INPUT,
    Component,
    EvaluateContext,
    PipelineComponent,
    StructuralComponent,
    component,
)
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.executor_graph_schedule import ExecutorGraphScheduler
from chemfit.graph_schedule_base import SerialGraphScheduler
from chemfit.scheduling import (
    DirectScheduler,
    EvaluationRequest,
    PreparedScheduleBase,
)


class FailingCloseSchedule(PreparedScheduleBase[object, object]):
    def evaluate_many(self, _requests):
        return iter(())

    def close(self) -> None:
        super().close()
        msg = "close failed"
        raise OSError(msg)


class SpecializedContext(EvaluateContext):
    pass


class SpecializedContextComponent(Component[int, int]):
    def _create_context(self) -> EvaluateContext:
        return SpecializedContext()

    def _run(self, value: int, /, ctx: EvaluateContext) -> int:
        assert isinstance(ctx, SpecializedContext)
        return 2 * value


def test_owned_and_borrowed_executor_lifecycles():
    @component()
    def identity(value):
        return value

    owned_executor = ThreadPoolExecutor(max_workers=1)
    owned_schedule = ExecutorGraphScheduler(
        executor_factory=lambda: owned_executor
    ).prepare(identity)
    owned_schedule.close()
    with pytest.raises(RuntimeError, match="shutdown"):
        owned_executor.submit(lambda: None)

    borrowed_executor = ThreadPoolExecutor(max_workers=1)
    try:
        borrowed_schedule = ExecutorGraphScheduler(executor=borrowed_executor).prepare(
            identity
        )
        borrowed_schedule.close()
        assert borrowed_executor.submit(lambda: 3).result() == 3
    finally:
        borrowed_executor.shutdown()


def test_context_manager_cleanup_does_not_mask_body_failure(
    caplog: pytest.LogCaptureFixture,
):
    schedule = FailingCloseSchedule()

    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(ValueError, match="body failed") as exception_info,
        schedule,
    ):
        msg = "body failed"
        raise ValueError(msg)

    assert schedule.closed
    assert "close failed" in caplog.text
    if hasattr(exception_info.value, "__notes__"):
        assert "close failed" in "\n".join(exception_info.value.__notes__)


def test_context_manager_propagates_cleanup_failure_after_success():
    schedule = FailingCloseSchedule()

    with pytest.raises(OSError, match="close failed"), schedule:
        pass

    assert schedule.closed


def test_closing_partially_consumed_generator_closes_schedule():
    @component()
    def identity(value):
        return value

    schedule = SerialGraphScheduler().prepare(identity)
    results = schedule.evaluate_many(
        [EvaluationRequest(value, EvaluateContext()) for value in (1, 2, 3)]
    )

    assert next(results).value == 1
    results.close()

    assert schedule.closed
    with pytest.raises(RuntimeError, match="closed"):
        schedule(4)


def test_direct_and_serial_graph_schedulers_produce_the_same_result():
    @component()
    def double(value):
        return 2 * value

    pipeline = PipelineComponent(double, double)
    direct_ctx = EvaluateContext()
    graph_ctx = EvaluateContext()

    with DirectScheduler().prepare(pipeline) as direct_schedule:
        assert direct_schedule(3, direct_ctx) == 12

    with SerialGraphScheduler().prepare(pipeline) as graph_schedule:
        assert graph_schedule(3, graph_ctx) == 12

    assert [child.output for child in direct_ctx.children] == [6, 12]
    assert [child.output for child in graph_ctx.children] == [6, 12]


@pytest.mark.parametrize("scheduler", [DirectScheduler(), SerialGraphScheduler()])
def test_prepared_schedule_uses_component_default_context(scheduler):
    with scheduler.prepare(SpecializedContextComponent()) as schedule:
        assert schedule(3) == 6


def test_repeated_batches_replace_context_hierarchy():
    @component()
    def double(value):
        return 2 * value

    @component()
    def add_one(value):
        return value + 1

    ctx = EvaluateContext()
    with SerialGraphScheduler().prepare(PipelineComponent(double, add_one)) as schedule:
        assert schedule(input=2, ctx=ctx) == 5
        first_children = tuple(ctx.children)
        first_structure = ctx.meta["chemfit.structure"]
        first_structure["children"][0]["name"] = "mutated"
        assert schedule(4, ctx) == 9

    assert ctx.output == 9
    assert [child.output for child in ctx.children] == [8, 9]
    assert ctx.meta["chemfit.structure"] is not first_structure
    assert ctx.meta["chemfit.structure"]["children"][0]["name"] == "double"
    assert all(
        current is not previous
        for current, previous in zip(ctx.children, first_children, strict=True)
    )


def test_extending_pipeline_rebuilds_structure_metadata_blueprint():
    @component()
    def double(value):
        return 2 * value

    pipeline = PipelineComponent(double, double)
    assert pipeline(2) == 8

    extended = pipeline.then(double)
    ctx = EvaluateContext()

    assert extended(2, ctx) == 16
    assert len(ctx.meta["chemfit.structure"]["children"]) == 3


def test_custom_structure_without_super_init_builds_metadata_on_demand():
    @component()
    def double(value):
        return 2 * value

    class CustomStructure(StructuralComponent[int, int]):
        def __init__(self) -> None:
            Component.__init__(self)

        def components(self):
            return (double,)

        def component_dependencies(self):
            return ((),)

        def component_inputs(self):
            return ((PARENT_INPUT,),)

        def combine_outcomes(self, outcomes, ctx):
            del ctx
            return outcomes[0]

    ctx = EvaluateContext()

    assert CustomStructure()(3, ctx) == 6
    assert ctx.meta["chemfit.structure"]["children"] == [{"index": 0, "name": "double"}]


def test_thread_backend_executes_independent_nodes_concurrently():
    barrier = threading.Barrier(2)

    @component()
    def synchronized(value):
        barrier.wait(timeout=2)
        return value

    objective = CombinedObjectiveFunction([synchronized, synchronized])
    with ExecutorGraphScheduler(
        executor_factory=lambda: ThreadPoolExecutor(max_workers=2)
    ).prepare(objective) as schedule:
        assert schedule(3) == 6


def test_thread_batch_results_are_yielded_in_completion_order():
    release_first = threading.Event()

    @component()
    def controlled(value):
        if value == 0:
            assert release_first.wait(timeout=2)
        return value

    requests = [EvaluationRequest(value, EvaluateContext()) for value in (0, 1)]
    with ExecutorGraphScheduler(
        executor_factory=lambda: ThreadPoolExecutor(max_workers=2)
    ).prepare(controlled) as schedule:
        results = schedule.evaluate_many(requests)

        first = next(results)
        assert (first.index, first.value) == (1, 1)

        release_first.set()
        second = next(results)
        assert (second.index, second.value) == (0, 0)

        with pytest.raises(StopIteration):
            next(results)


def test_catastrophic_worker_failure_closes_schedule_but_not_borrowed_executor():
    @component()
    def interrupt(_value):
        msg = "worker interrupted"
        raise KeyboardInterrupt(msg)

    executor = ThreadPoolExecutor(max_workers=1)
    schedule = ExecutorGraphScheduler(executor=executor).prepare(interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="worker interrupted"):
            schedule(1)

        assert schedule.closed
        assert executor.submit(lambda: 7).result() == 7
    finally:
        executor.shutdown()
