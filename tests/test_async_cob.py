import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from chemfit.abstract_objective_function import EvaluateContext
from chemfit.async_helpers import async_eval_many
from chemfit.async_wrapper_cob import AsyncWrapperCOB
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.wrap_funcs import to_objective_functor


class MockExecutor(ThreadPoolExecutor):
    def __init__(self, *args, **kwargs):
        """Mock Executor that counts the number of submits."""

        self.n_submit = 0
        super().__init__(*args, **kwargs)

    def submit(self, fn, /, *args, **kwargs):  # noqa: ANN001
        self.n_submit += 1
        return super().submit(fn, *args, **kwargs)


def test_async_cob():
    @to_objective_functor
    def a(p: dict):
        time.sleep(0.5)
        return p["x"] ** 2

    @to_objective_functor
    def b(p: dict):
        time.sleep(0.5)
        return p["y"] ** 2

    params = {"x": 1, "y": 2}

    # We create a combined objective function
    cob = CombinedObjectiveFunction([a, a, b, b])
    # ... and an async wrapper around it
    async_cob = AsyncWrapperCOB(cob)

    # Here we make sure that the async result matches the syn result and that the executor was used
    ctx_sync = EvaluateContext()
    res_sync = cob(params, ctx_sync)

    ctx_async = EvaluateContext()
    ctx_async.executor = MockExecutor(max_workers=5)
    res_async = async_cob(params, ctx_async)

    assert ctx_async.executor.n_submit == cob.n_terms()
    assert np.isclose(res_sync, res_async)

    # Now we test if we can evaluate the objective function for many parameters at the same time
    params_list = [{"x": i, "y": 2 - i} for i in range(5)]

    contexts = [EvaluateContext(executor=MockExecutor(2)) for _ in params_list]
    results = asyncio.run(async_eval_many(async_cob, params_list, contexts))

    results_expected = [cob(p) for p in params_list]
    assert results == results_expected
