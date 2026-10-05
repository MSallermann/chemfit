import asyncio
import time
from concurrent.futures import Executor, Future, ThreadPoolExecutor

import numpy as np
import pytest

from chemfit.abstract_objective_function import EvaluateContext
from chemfit.async_helpers import async_eval_many
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.executor_scheduler import ExecutorTreeScheduler
from chemfit.tree_schedule import BatchState
from chemfit.wrap_funcs import objective


class MockExecutor(ThreadPoolExecutor):
    def __init__(self, *args, **kwargs):
        """Mock Executor that counts the number of submits."""

        self.n_submit = 0
        super().__init__(*args, **kwargs)

    def submit(self, fn, /, *args, **kwargs):  # noqa: ANN001
        self.n_submit += 1
        return super().submit(fn, *args, **kwargs)


class RecordingFuture(Future[object]):
    def __init__(self) -> None:
        """Initialize cancellation-call tracking."""
        super().__init__()
        self.cancel_calls = 0

    def cancel(self) -> bool:
        self.cancel_calls += 1
        return super().cancel()


class CatastrophicExecutor(Executor):
    def __init__(self, failure: BaseException) -> None:
        """Initialize an executor whose first future fails catastrophically."""

        self.failure = failure
        self.futures: list[RecordingFuture] = []

    def submit(self, fn, /, *args, **kwargs):  # noqa: ANN001
        del fn, args, kwargs
        future = RecordingFuture()
        self.futures.append(future)
        if len(self.futures) == 1:
            future.set_exception(self.failure)
        return future


def test_async_cob():
    @objective(pass_ctx=False)
    def a(p: dict[str, float]) -> float:
        time.sleep(0.5)
        return p["x"] ** 2

    @objective()
    def b(p: dict[str, float]) -> float:
        time.sleep(0.5)
        return p["y"] ** 2

    params = {"x": 1.0, "y": 2.0}

    # Compute a serial reference before preparing the executor scheduler.
    cob = CombinedObjectiveFunction([a, a, b, b])

    ctx_sync = EvaluateContext()
    res_sync = cob(params, ctx_sync)

    with (
        MockExecutor(max_workers=5) as executor,
        ExecutorTreeScheduler(executor=executor).prepare(cob) as schedule,
    ):
        ctx_async = EvaluateContext()
        res_async = schedule.evaluate(params, ctx_async)

        assert executor.n_submit == cob.n_terms()
        assert np.isclose(res_sync, res_async)

        # Evaluate the prepared schedule concurrently from several threads.
        params_list = [{"x": float(i), "y": float(2) - i} for i in range(5)]

        contexts = [EvaluateContext() for _ in params_list]
        results = asyncio.run(async_eval_many(schedule.evaluate, params_list, contexts))

        results_expected = [cob(p) for p in params_list]
        assert results == results_expected


def test_catastrophic_executor_failure_cancels_pending_futures() -> None:
    failure = KeyboardInterrupt("executor catastrophe")
    executor = CatastrophicExecutor(failure)
    objective = CombinedObjectiveFunction(
        [lambda _parameters: 1.0, lambda _parameters: 2.0]
    )
    schedule = ExecutorTreeScheduler(executor=executor).prepare(objective)

    with pytest.raises(KeyboardInterrupt, match="executor catastrophe") as raised:
        schedule.evaluate({}, EvaluateContext())

    assert raised.value is failure
    assert schedule.closed
    assert len(executor.futures) == 2
    assert executor.futures[1].cancel_calls == 1
    assert executor.futures[1].cancelled()
    with pytest.raises(RuntimeError, match="Prepared schedule is closed"):
        schedule.evaluate({}, EvaluateContext())


def test_catastrophic_executor_setup_failure_before_futures_exist() -> None:
    failure = KeyboardInterrupt("executor setup catastrophe")

    def fail_during_setup(
        idx_child_ctx: int,
        child_ctx: EvaluateContext,
        num_children: int,
        parent_ctx: EvaluateContext,
    ) -> None:
        del idx_child_ctx, child_ctx, num_children, parent_ctx
        raise failure

    executor = CatastrophicExecutor(AssertionError("submit should not be called"))
    objective = CombinedObjectiveFunction(
        [lambda _parameters: 1.0],
        child_context_configurator=fail_during_setup,
    )
    schedule = ExecutorTreeScheduler(executor=executor).prepare(objective)

    with pytest.raises(KeyboardInterrupt, match="executor setup catastrophe") as raised:
        schedule.evaluate({}, EvaluateContext())

    assert raised.value is failure
    assert getattr(failure, "__notes__", []) == []
    assert executor.futures == []
    assert schedule.closed
    schedule.abort_batch(BatchState())
