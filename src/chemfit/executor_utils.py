from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    ParamSpec,
    TypeVar,
    cast,
)

from chemfit.abstract_objective_function import EvaluateContext, ExecutorLike

if TYPE_CHECKING:
    from collections.abc import Iterable

T_co = TypeVar("T_co", covariant=True)
P = ParamSpec("P")


class AttachContextAsReturnValue(Generic[T_co]):
    def __init__(self, func: Callable[..., T_co]) -> None:
        """Wrap a function and return the pickled EvaluateContext state."""
        self.func = func

    def __call__(self, *args: Any) -> tuple[T_co, dict[str, Any]]:
        ctx = cast("EvaluateContext", args[-1])
        assert isinstance(ctx, EvaluateContext)
        return (self.func(*args), ctx.__getstate__())


def map_with_context(
    executor: ExecutorLike,
    fn: Callable[..., T_co],
    *iterables: Iterable[Any],
    ctxs: Iterable[EvaluateContext],
    timeout: float | None = None,
    chunksize: int = 1,
) -> Iterable[T_co]:
    fn_with_ctx_ret = AttachContextAsReturnValue(func=fn)

    return_vals_with_ctx = executor.map(
        fn_with_ctx_ret, *iterables, ctxs, timeout=timeout, chunksize=chunksize
    )

    return_vals = []
    for ctx_in, (r, ctx_r) in zip(ctxs, return_vals_with_ctx):
        ctx_in.__setstate__(ctx_r)
        return_vals.append(r)

    return return_vals
