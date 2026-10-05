from concurrent.futures import Executor, Future, ThreadPoolExecutor

import pytest

from chemfit.abstract_objective_function import EvaluateContext
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.executor_scheduler import ExecutorTreeScheduler
from chemfit.tree_schedule import BatchState


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
