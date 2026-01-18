from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable

from chemfit import abstract_objective_function, wrap_funcs
from chemfit.abstract_objective_function import EvaluateContext
from chemfit.async_wrapper_cob import AsyncWrapperCOB
from chemfit.combined_objective_function import CombinedObjectiveFunction


def _result_or_cancel(fut: MyFuture, timeout: float | None = None):
    try:
        try:
            return fut.result(timeout)
        finally:
            fut.cancel()
    finally:
        # Break a reference cycle with the exception in self._exception
        del fut


class MyFuture(abstract_objective_function.Future):
    def __init__(self, func: Callable, args: Any) -> None:
        """Initialize a future."""

        self.func = func
        self.args = args

    def result(self, timeout: float | None = None):  # noqa: ARG002
        return self.func(*self.args)

    def cancel(self): ...


class MyExecutor(abstract_objective_function.Executor):
    def submit(self, fn: Callable, *args) -> MyFuture:
        print(f"Submit with args {args}")
        return MyFuture(fn, args)

    def map(self, fn: Callable, *iterables, timeout: float | None = None):
        if timeout is not None:
            end_time = timeout + time.monotonic()

        fs = [self.submit(fn, *args) for args in zip(*iterables)]

        # Yield must be hidden in closure so that the futures are submitted
        # before the first iterator value is required.
        def result_iterator():
            try:
                # reverse to keep finishing order
                fs.reverse()
                while fs:
                    # Careful not to keep a reference to the popped future
                    if timeout is None:
                        yield _result_or_cancel(fs.pop())
                    else:
                        yield _result_or_cancel(fs.pop(), end_time - time.monotonic())
            finally:
                for future in fs:
                    future.cancel()

        return result_iterator()


class MyFunctor(abstract_objective_function.ObjectiveFunctor):
    def __call__(
        self,
        parameters: dict[str, float],
        ctx: EvaluateContext | None = None,
    ) -> float:
        ctx.loss = 99
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
# ... and an async wrapper around it
async_cob = AsyncWrapperCOB(cob)


def test_executors():
    executors = [MyExecutor(), ProcessPoolExecutor(), ThreadPoolExecutor()]

    for executor in executors:
        func = wrap_funcs.WrappedObjectiveFunctor(my_func)
        ctx = EvaluateContext(executor=executor)
        params = {"a": 2.0, "b": -1.0}
        fut = ctx.executor.submit(func, params, ctx)

        print(fut.result())
        print(func(params, ctx))
        print(ctx.loss)
        print(ctx.parameters)
        print(ctx.executor)

        async_cob(params, ctx)
