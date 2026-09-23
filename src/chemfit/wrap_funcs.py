from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Generic, TypeVar

from typing_extensions import Concatenate

from chemfit.abstract_objective_function import (
    EvaluateContext,
    ObjectiveFunctor,
    QuantityComputer,
)

# These mirror the input/output directions of the abstract interfaces:
# wrapped functions consume parameters and produce quantities.  Keeping that
# variance lets decorators preserve a user's concrete callback annotations.
ParametersT_contra = TypeVar(
    "ParametersT_contra", bound=Mapping[str, object], contravariant=True
)
QuantitiesT_co = TypeVar("QuantitiesT_co", bound=dict[str, Any], covariant=True)

# Concatenate preserves the first ChemFit argument while allowing bind() to
# carry arbitrary additional positional or keyword parameters.
WrappableObjFunction = Callable[Concatenate[ParametersT_contra, ...], float]


class WrappedObjectiveFunctor(
    ObjectiveFunctor[ParametersT_contra], Generic[ParametersT_contra]
):
    def __init__(
        self,
        func: WrappableObjFunction[ParametersT_contra],
        pass_ctx: bool = False,
        func_args: tuple[Any, ...] | None = None,
        func_kwargs: dict[str, Any] | None = None,
    ):
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

        if func_args is None:
            self.func_args = ()
        else:
            self.func_args = func_args

        if func_kwargs is None:
            self.func_kwargs = {}
        else:
            self.func_kwargs = func_kwargs

    def bind(
        self, /, *args: Any, **kwargs: Any
    ) -> WrappedObjectiveFunctor[ParametersT_contra]:
        """
        Return a new quantity computer with extra arguments bound.

        The bound arguments are passed to the wrapped function in addition
        to the usual ChemFit arguments.

        Args:
            *args: Positional arguments to bind after ``parameters`` (and
                after ``ctx`` as well if ``pass_ctx=True``).
            **kwargs: Keyword arguments to bind.

        Returns:
            A new wrapped quantity computer with the requested arguments
            pre-applied.

        """
        return type(self)(
            func=self.func, pass_ctx=self.pass_ctx, func_args=args, func_kwargs=kwargs
        )

    def __call__(
        self, parameters: ParametersT_contra, ctx: EvaluateContext | None = None
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
            loss = self.func(parameters, *self.func_args, **self.func_kwargs, ctx=ctx)
        else:
            loss = self.func(parameters, *self.func_args, **self.func_kwargs)

        ctx.loss = loss
        return loss


def to_objective_functor(
    pass_ctx: bool = False,
) -> Callable[
    [WrappableObjFunction[ParametersT_contra]],
    WrappedObjectiveFunctor[ParametersT_contra],
]:
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

    def wrap(
        func: WrappableObjFunction[ParametersT_contra],
    ) -> WrappedObjectiveFunctor[ParametersT_contra]:
        return WrappedObjectiveFunctor(func, pass_ctx=pass_ctx)

    return wrap


WrappableQuantFunction = Callable[Concatenate[ParametersT_contra, ...], QuantitiesT_co]


class WrappedQuantityComputer(
    QuantityComputer[ParametersT_contra, QuantitiesT_co],
    Generic[ParametersT_contra, QuantitiesT_co],
):
    def __init__(
        self,
        func: WrappableQuantFunction[ParametersT_contra, QuantitiesT_co],
        pass_ctx: bool = False,
        func_args: tuple[Any, ...] | None = None,
        func_kwargs: dict[str, Any] | None = None,
    ):
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

        if func_args is None:
            self.func_args = ()
        else:
            self.func_args = func_args

        if func_kwargs is None:
            self.func_kwargs = {}
        else:
            self.func_kwargs = func_kwargs

    def bind(
        self, /, *args: Any, **kwargs: Any
    ) -> WrappedQuantityComputer[ParametersT_contra, QuantitiesT_co]:
        """
        Return a new quantity computer with extra arguments bound.

        The bound arguments are passed to the wrapped function in addition
        to the usual ChemFit arguments.

        Args:
            *args: Positional arguments to bind after ``parameters`` (and
                after ``ctx`` as well if ``pass_ctx=True``).
            **kwargs: Keyword arguments to bind.

        Returns:
            A new wrapped quantity computer with the requested arguments
            pre-applied.

        """
        return type(self)(
            func=self.func, pass_ctx=self.pass_ctx, func_args=args, func_kwargs=kwargs
        )

    def _compute(
        self,
        parameters: ParametersT_contra,
        ctx: EvaluateContext,
    ) -> QuantitiesT_co:
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
            return self.func(parameters, *self.func_args, **self.func_kwargs, ctx=ctx)

        return self.func(parameters, *self.func_args, **self.func_kwargs)


def to_quantity_computer(
    pass_ctx: bool = False,
) -> Callable[
    [WrappableQuantFunction[ParametersT_contra, QuantitiesT_co]],
    WrappedQuantityComputer[ParametersT_contra, QuantitiesT_co],
]:
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

    def wrap(
        func: WrappableQuantFunction[ParametersT_contra, QuantitiesT_co],
    ) -> WrappedQuantityComputer[ParametersT_contra, QuantitiesT_co]:
        return WrappedQuantityComputer(func, pass_ctx=pass_ctx)

    return wrap
