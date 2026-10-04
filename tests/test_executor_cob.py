import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from chemfit.abstract_objective_function import EvaluateContext
from chemfit.async_helpers import async_eval_many
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.executor_scheduler import ExecutorTreeScheduler
from chemfit.wrap_funcs import objective


class MockExecutor(ThreadPoolExecutor):
    def __init__(self, *args, **kwargs):
        """Mock Executor that counts the number of submits."""

        self.n_submit = 0
        super().__init__(*args, **kwargs)

    def submit(self, fn, /, *args, **kwargs):  # noqa: ANN001
        self.n_submit += 1
        return super().submit(fn, *args, **kwargs)


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
