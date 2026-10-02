from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING

from chemfit import abstract_objective_function, wrap_funcs
from chemfit.abstract_objective_function import EvaluateContext
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.executor_scheduler import ExecutorTreeScheduler

if TYPE_CHECKING:
    from collections.abc import Callable


def _result_or_cancel(fut: MyFuture, timeout: float | None = None):
    try:
        try:
            return fut.result(timeout)
        finally:
            fut.cancel()
    finally:
        # Break a reference cycle with the exception in self._exception
        del fut


class MyFuture(Future):
    """Immediately completed Future used by MyExecutor."""


class MyExecutor:
    def submit(self, fn: Callable, /, *args, **kwargs) -> MyFuture:
        print(f"Submit with args {args}")

        fut = MyFuture()

        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            fut.set_exception(exc)
        else:
            fut.set_result(result)

        return fut


class MyFunctor(abstract_objective_function.ObjectiveFunctor):
    def _evaluate(
        self,
        parameters: dict[str, float],
        ctx: EvaluateContext,
    ) -> float:
        ctx.parameters = parameters
        return parameters["a"] ** 2 - parameters["b"]


def my_func(parameters: dict[str, float]):
    return parameters["a"] ** 2 - parameters["b"]


def a(p: dict):
    time.sleep(0.5)
    return p["a"] ** 2


def b(p: dict):
    time.sleep(0.5)
    return p["b"] ** 2


# We create a combined objective function
cob = CombinedObjectiveFunction([a, a, b, b])


def test_executors():
    executors = [MyExecutor(), ProcessPoolExecutor(), ThreadPoolExecutor()]

    for executor in executors:
        cob.set_scheduler(ExecutorTreeScheduler(executor=executor))
        func = wrap_funcs.WrappedObjectiveFunctor(my_func)
        ctx = EvaluateContext(executor=executor)
        params = {"a": 2.0, "b": -1.0}

        assert ctx.executor is not None
        fut = ctx.executor.submit(func, params, ctx)

        print(fut.result())
        print(func(params, ctx))
        print(ctx.loss)
        print(ctx.parameters)
        print(ctx.executor)

        cob(params, ctx)
