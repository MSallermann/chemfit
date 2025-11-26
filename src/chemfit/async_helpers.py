import asyncio
from collections.abc import Iterable
from typing import Any

from chemfit.abstract_objective_function import EvaluateContext, ObjectiveFunctor


async def async_eval_one(
    obj: ObjectiveFunctor, params: dict[str, Any], ctx: EvaluateContext
):
    """
    Run a single evaluation "asynchronously" using a fresh EvaluateContext.

    This uses asyncio.to_thread here, but you could also use a process pool
    or real async if your internals are I/O-bound.
    """

    return await asyncio.to_thread(obj, params, ctx)


async def async_eval_many(
    obj: ObjectiveFunctor,
    params_list: Iterable[dict[str, Any]],
    ctxs: Iterable[EvaluateContext],
):
    tasks = [
        async_eval_one(obj, p, ctx) for p, ctx in zip(params_list, ctxs, strict=True)
    ]
    return await asyncio.gather(*tasks)
