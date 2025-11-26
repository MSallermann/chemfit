import asyncio
import time

import numpy as np

from chemfit.abstract_objective_function import EvaluateContext
from chemfit.async_helpers import async_eval_many
from chemfit.async_wrapper_cob import AsyncWrapperCOB
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.wrap_funcs import to_objective_functor


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

    cob = CombinedObjectiveFunction([a, a, b, b])
    async_cob = AsyncWrapperCOB(cob)

    ctx_sync = EvaluateContext()
    res_sync = cob(params, ctx_sync)

    ctx_async = EvaluateContext()
    res_async = async_cob(params, ctx_async)

    assert np.isclose(res_sync, res_async)

    params_list = [{"x": i, "y": 2 - i} for i in range(5)]
    contexts = [EvaluateContext() for _ in params_list]

    results = asyncio.run(async_eval_many(async_cob, params_list, contexts))

    print(results)
    print([c.to_meta_data() for c in contexts])
