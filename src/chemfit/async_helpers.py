import asyncio
from collections.abc import Iterable
from typing import Any, Callable

from chemfit.abstract_objective_function import EvaluateContext


async def async_eval_one(
    obj: Callable[[dict[str, Any], EvaluateContext], float],
    params: dict[str, Any],
    ctx: EvaluateContext,
):
    """
    Evaluate one objective call asynchronously.

    This helper runs a single evaluation of an objective function in a
    background thread using ``asyncio.to_thread``. It is intended for
    concurrent or parallel evaluation of stateless objectives that accept
    a `(params, ctx)` signature.

    Args:
        obj (Callable[[dict[str, Any], EvaluateContext], float]):
            Objective-like callable. Typically an ``ObjectiveFunctor`` or a
            wrapper around one. Must be synchronous and thread-safe when
            provided distinct ``EvaluateContext`` instances.
        params (dict[str, Any]):
            Parameter dictionary for this evaluation.
        ctx (EvaluateContext):
            Context that will be populated during evaluation.

    Returns:
        float: The scalar loss computed by ``obj``.

    Notes:
        - This function does **not** create the context; callers must supply
          one context per evaluation.
        - ``async_eval_one`` does not modify ``obj``. If ``obj`` stores
          per-evaluation state internally, it is **not** safe to use with
          this function.

    """
    return await asyncio.to_thread(obj, params, ctx)


async def async_eval_many(
    obj: Callable[[dict[str, Any], EvaluateContext], float],
    params_list: Iterable[dict[str, Any]],
    ctxs: Iterable[EvaluateContext],
):
    """
    Evaluate multiple objective calls concurrently.

    This helper schedules a batch of evaluations of the same objective
    using ``asyncio.gather``. Each evaluation receives its own distinct
    ``EvaluateContext``. All evaluations run concurrently via
    ``asyncio.to_thread`` and therefore do not block the event loop.

    Args:
        obj (Callable[[dict[str, Any], EvaluateContext], float]):
            Objective-like callable. Must be compatible with concurrent
            calls when provided separate contexts.
        params_list (Iterable[dict[str, Any]]):
            Iterable of parameter dictionaries. One evaluation is performed
            per entry.
        ctxs (Iterable[EvaluateContext]):
            Iterable of contexts. Must have the same length as
            ``params_list``. Each context is populated independently.

    Returns:
        list[float]: A list of scalar loss values in the same order as
        ``params_list``.

    Raises:
        ValueError: If ``params_list`` and ``ctxs`` have mismatched lengths
            when zipped with ``strict=True``.

    Notes:
        - The caller is responsible for allocating exactly one
          ``EvaluateContext`` per evaluation.
        - This function does not modify shared state on ``obj``; if the
          objective stores per-evaluation data in ``self`` instead of in
          ``ctx``, it cannot be used safely.

    """
    tasks = [
        async_eval_one(obj, p, ctx) for p, ctx in zip(params_list, ctxs, strict=True)
    ]
    return await asyncio.gather(*tasks)
