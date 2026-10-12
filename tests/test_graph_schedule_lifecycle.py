import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from chemfit.abstract_objective_function import (
    EvaluateContext,
    PipelineComponent,
    component,
)
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.executor_graph_schedule import ExecutorGraphScheduler
from chemfit.graph_schedule_base import SerialGraphScheduler
from chemfit.scheduling import EvaluationRequest


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


def test_repeated_batches_replace_context_graph_state():
    @component()
    def double(value):
        return 2 * value

    @component()
    def add_one(value):
        return value + 1

    ctx = EvaluateContext()
    with SerialGraphScheduler().prepare(PipelineComponent(double, add_one)) as schedule:
        assert schedule(2, ctx) == 5
        assert schedule(4, ctx) == 9

    assert ctx.output == 9
    assert [child.output for child in ctx.incoming] == [8, 9]


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


def test_catastrophic_worker_failure_closes_schedule_but_not_borrowed_executor():
    @component()
    def interrupt(_value):
        raise KeyboardInterrupt("worker interrupted")

    executor = ThreadPoolExecutor(max_workers=1)
    schedule = ExecutorGraphScheduler(executor=executor).prepare(interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="worker interrupted"):
            schedule(1)

        assert schedule.closed
        assert executor.submit(lambda: 7).result() == 7
    finally:
        executor.shutdown()
