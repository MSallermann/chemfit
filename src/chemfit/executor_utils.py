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
    """
    Wrap a callable so it also returns serialized context state.

    This helper is primarily used when executing functions through an
    executor that may run in a separate process. In that case,
    mutations to an ``EvaluateContext`` made inside the worker are not
    reflected in the caller's original context object. This wrapper
    makes those side effects explicit by returning the function result
    together with ``ctx.__getstate__()``.

    The wrapped callable is expected to receive an ``EvaluateContext``
    as its final positional argument.
    """

    def __init__(self, func: Callable[..., T_co]) -> None:
        """
        Initialize the wrapper.

        Args:
            func: Callable whose final positional argument must be an
                ``EvaluateContext``.

        """
        self.func = func

    def __call__(self, *args: Any) -> tuple[T_co, dict[str, Any]]:
        """
        Call the wrapped function and return its result with context state.

        Args:
            *args: Positional arguments forwarded to the wrapped callable.
                The final argument must be an ``EvaluateContext``.

        Returns:
            A tuple containing:
                - The return value of the wrapped callable.
                - The serialized state of the provided ``EvaluateContext``.

        Raises:
            AssertionError: If the final positional argument is not an
                ``EvaluateContext``.

        """

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
) -> list[T_co]:
    """
    Map a function over iterables and propagate context side effects.

    This helper behaves like ``executor.map(...)`` for callables whose
    final positional argument is an ``EvaluateContext``. Each worker
    returns both the function result and the serialized state of its
    context. After execution, the input context objects are updated in
    place via ``EvaluateContext.__setstate__()`` so that caller-visible
    context state reflects mutations performed inside the executor.

    This is especially useful for executors that may run work in
    separate processes, where in-worker mutations to context objects
    would otherwise not be visible to the caller.

    Args:
        executor: Executor used to evaluate the mapped calls.
        fn: Callable to evaluate. Its final positional argument must be
            an ``EvaluateContext``.
        *iterables: Iterables supplying the non-context positional
            arguments for each mapped call.
        ctxs: Iterable of evaluation contexts, one per mapped call.
        timeout: Maximum number of seconds to wait for results. If
            ``None``, wait indefinitely.
        chunksize: Approximate chunk size passed through to
            ``executor.map(...)``.

    Returns:
        A list of function return values in executor map order.

    Side Effects:
        Updates each context in ``ctxs`` in place using the serialized
        state returned from the corresponding worker execution.

    """

    fn_with_ctx_ret = AttachContextAsReturnValue(func=fn)

    return_vals_with_ctx = executor.map(
        fn_with_ctx_ret, *iterables, ctxs, timeout=timeout, chunksize=chunksize
    )

    return_vals = []
    for ctx_in, (r, ctx_r) in zip(ctxs, return_vals_with_ctx):
        ctx_in.__setstate__(ctx_r)
        return_vals.append(r)

    return return_vals
