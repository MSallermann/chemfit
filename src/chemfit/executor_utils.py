from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

from chemfit.abstract_objective_function import EvaluateContext, ExecutorLike

if TYPE_CHECKING:
    from collections.abc import Iterable

T = TypeVar("T")


def attach_context_as_return_value(func: Callable[..., T]):
    def wrapped(*params) -> tuple[T, dict[str, Any]]:
        ctx = cast("EvaluateContext", params[-1])
        return (func(*params), ctx.__getstate__())

    return wrapped


def map_with_context(
    executor: ExecutorLike,
    fn: Callable[..., T],
    *iterables: Iterable[Any],
    ctxs: Iterable[EvaluateContext],
    timeout: float | None = None,
    chunksize: int = 1,
) -> Iterable[T]:
    fn_with_ctx_ret = attach_context_as_return_value(func=fn)

    return_vals_with_ctx = executor.map(
        fn_with_ctx_ret, *iterables, ctxs, timeout=timeout, chunksize=chunksize
    )

    return_vals = []
    for ctx_in, (r, ctx_r) in zip(ctxs, return_vals_with_ctx):
        ctx_in.__setstate__(ctx_r)
        return_vals.append(r)

    return return_vals
