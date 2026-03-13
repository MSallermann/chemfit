from __future__ import annotations

from typing import Any, Callable

from chemfit.abstract_objective_function import (
    EvaluateContext,
    ObjectiveFunctor,
    QuantityComputer,
)

WrappableObjFunction = (
    Callable[[dict[str, Any]], float]
    | Callable[[dict[str, Any], EvaluateContext], float]
)


class WrappedObjectiveFunctor(ObjectiveFunctor):
    def __init__(self, func: WrappableObjFunction, pass_ctx: bool = False):
        """
        Initialize a wrapped objective functor.

        Args:
            func: Callable to wrap as an ``ObjectiveFunctor``. The callable may
                either accept only ``parameters`` or accept both
                ``parameters`` and ``ctx``.
            pass_ctx: If ``True``, call ``func(parameters, ctx)``. If
                ``False``, call ``func(parameters)``.

        """
        super().__init__()
        self.func = func
        self.pass_ctx = pass_ctx

    def __call__(
        self, parameters: dict[str, Any], ctx: EvaluateContext | None = None
    ) -> float:
        """
        Evaluate the wrapped callable as an objective functor.

        If ``pass_ctx`` is ``True``, the wrapped callable receives both the
        parameter dictionary and the evaluation context. Otherwise, it
        receives only the parameter dictionary.

        Args:
            parameters: Parameter dictionary for the current evaluation.
            ctx: Optional evaluation context. If ``None``, a new
                ``EvaluateContext`` is created.

        Returns:
            Scalar loss value returned by the wrapped callable.

        Side Effects:
            - Stores ``parameters`` in ``ctx.parameters``.
            - Stores the returned loss in ``ctx.loss``.

        """

        if ctx is None:
            ctx = EvaluateContext()

        ctx.parameters = parameters

        if self.pass_ctx:
            ctx.loss = self.func(parameters, ctx)
        else:
            ctx.loss = self.func(parameters)

        return ctx.loss


def to_objective_functor(pass_ctx: bool = False):
    """
    Create a decorator that wraps a callable as an objective functor.

    Args:
        pass_ctx: If ``True``, the decorated callable is expected to accept
            ``(parameters, ctx)``. Otherwise, it is expected to accept only
            ``(parameters)``.

    Returns:
        Decorator that converts a compatible callable into a
        ``WrappedObjectiveFunctor``.

    """

    def wrap(func: WrappableObjFunction):
        return WrappedObjectiveFunctor(func, pass_ctx=pass_ctx)

    return wrap


WrappableQuantFunction = (
    Callable[[dict[str, Any]], dict[str, Any]]
    | Callable[[dict[str, Any], EvaluateContext], dict[str, Any]]
)


class WrappedQuantityComputer(QuantityComputer):
    def __init__(self, func: WrappableQuantFunction, pass_ctx: bool = False):
        """
        Initialize a wrapped quantity computer.

        Args:
            func: Callable to wrap as a ``QuantityComputer``. The callable may
                either accept only ``parameters`` or accept both
                ``parameters`` and ``ctx``.
            pass_ctx: If ``True``, call ``func(parameters, ctx)``. If
                ``False``, call ``func(parameters)``.

        """

        super().__init__()
        self.func = func
        self.pass_ctx = pass_ctx

    def _compute(
        self,
        parameters: dict[str, Any],
        ctx: EvaluateContext,
    ) -> dict[str, Any]:
        """
        Compute quantities using the wrapped callable.

        If ``pass_ctx`` is ``True``, the wrapped callable receives both the
        parameter dictionary and the evaluation context. Otherwise, it
        receives only the parameter dictionary.

        Args:
            parameters: Parameter dictionary for the current evaluation.
            ctx: Evaluation context for the current call.

        Returns:
            Quantity dictionary returned by the wrapped callable.

        """

        if self.pass_ctx:
            return self.func(parameters, ctx)

        return self.func(parameters)


def to_quantity_computer(pass_ctx: bool = False):
    """
    Create a decorator that wraps a callable as a quantity computer.

    Args:
        pass_ctx: If ``True``, the decorated callable is expected to accept
            ``(parameters, ctx)``. Otherwise, it is expected to accept only
            ``(parameters)``.

    Returns:
        Decorator that converts a compatible callable into a
        ``WrappedQuantityComputer``.

    """

    def wrap(func: WrappableQuantFunction):
        return WrappedQuantityComputer(func, pass_ctx=pass_ctx)

    return wrap
