from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar

# Python 3.10 needs the backport for Concatenate[T, ...].
from typing_extensions import Concatenate  # noqa: UP035

from chemfit.abstract_objective_function import (
    EvaluateContext,
    ObjectiveFunctor,
    QuantityComputer,
    ResourceRequest,
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
        resources: ResourceRequest | None = None,
    ):
        """
        Initialize a wrapped objective functor.

        Args:
            func: Callable to wrap as an ``ObjectiveFunctor``. The callable may
                either accept only ``parameters`` or accept both
                ``parameters`` and ``ctx``.
            pass_ctx: If ``True``, call ``func(parameters, ctx)``. If
                ``False``, call ``func(parameters)``.
            resources: Static resources required for one evaluation.

        """
        super().__init__()
        self.func = func
        self.pass_ctx = pass_ctx
        if resources is not None:
            self.resources = resources

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
        Return a new objective functor with extra arguments bound.

        The bound arguments are passed to the wrapped function in addition
        to the usual ChemFit arguments. Static metadata and registered hook
        objects are retained, while the returned functor has independent hook
        registration lists.

        Args:
            *args: Positional arguments to bind after ``parameters`` (and
                after ``ctx`` as well if ``pass_ctx=True``).
            **kwargs: Keyword arguments to bind.

        Returns:
            A new wrapped objective functor with the requested arguments
            pre-applied.

        """
        new = type(self)(
            func=self.func,
            pass_ctx=self.pass_ctx,
            func_args=args,
            func_kwargs=kwargs,
            resources=self.resources,
        )
        new.static_meta_data = self.static_meta_data.copy()
        new.pre_eval_hooks = self.pre_eval_hooks.copy()
        new.post_eval_hooks = self.post_eval_hooks.copy()
        return new

    def _evaluate(self, parameters: ParametersT_contra, ctx: EvaluateContext) -> float:
        """
        Evaluate the wrapped callable as an objective functor.

        If ``pass_ctx`` is ``True``, the wrapped callable receives both the
        parameter dictionary and the evaluation context. Otherwise, it
        receives only the parameter dictionary.

        Args:
            parameters: Parameter dictionary for the current evaluation.
            ctx: Evaluation context for the current call.

        Returns:
            Scalar loss value returned by the wrapped callable.

        """

        if self.pass_ctx:
            loss = self.func(parameters, *self.func_args, **self.func_kwargs, ctx=ctx)
        else:
            loss = self.func(parameters, *self.func_args, **self.func_kwargs)

        return loss


def objective(
    *,
    pass_ctx: bool = False,
    resources: ResourceRequest | None = None,
) -> Callable[
    [WrappableObjFunction[ParametersT_contra]],
    WrappedObjectiveFunctor[ParametersT_contra],
]:
    """
    Wrap a Python loss function as a ChemFit objective.

    The decorated callable receives the parameter mapping and returns a scalar
    loss. The resulting :class:`WrappedObjectiveFunctor` participates in
    scheduling, evaluation hooks, static metadata, and objective composition.
    Extra function arguments can be configured later with ``bind()``.

    Args:
        pass_ctx: If ``True``, pass the current context as the keyword argument
            ``ctx`` in addition to the parameter mapping. Otherwise the
            callable receives only the parameters and any bound arguments.
        resources: Static resources required for one evaluation.

    Returns:
        A decorator that converts a compatible callable into a
        :class:`WrappedObjectiveFunctor`.

    Examples:
        Define and tag a directly computed objective term::

            @chemfit.objective(resources={"cpu": 1})
            def regularization(parameters, *, strength):
                return strength * parameters["epsilon"] ** 2

            term = (
                regularization.bind(strength=0.01)
                .with_meta(kind="regularization")
            )

        Request the evaluation context when the function needs to record
        metadata::

            @chemfit.objective(pass_ctx=True)
            def monitored_loss(parameters, *, ctx):
                ctx.meta["model"] = "lj"
                return parameters["epsilon"] ** 2

    Notes:
        ``with_meta()`` and ``bind()`` return fluent variants without changing
        the source objective. Hook registrations already present on the source
        are retained, while later registrations on a variant are independent.

    """

    def wrap(
        func: WrappableObjFunction[ParametersT_contra],
    ) -> WrappedObjectiveFunctor[ParametersT_contra]:
        return WrappedObjectiveFunctor(
            func,
            pass_ctx=pass_ctx,
            resources=resources,
        )

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
        resources: ResourceRequest | None = None,
    ):
        """
        Initialize a wrapped quantity computer.

        Args:
            func: Callable to wrap as a ``QuantityComputer``. The callable may
                either accept only ``parameters`` or accept both
                ``parameters`` and ``ctx``.
            pass_ctx: If ``True``, call ``func(parameters, ctx)``. If
                ``False``, call ``func(parameters)``.
            resources: Static resources required for one computation.

        """

        super().__init__()
        self.func = func
        self.pass_ctx = pass_ctx
        if resources is not None:
            self.resources = resources

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
        new = type(self)(
            func=self.func,
            pass_ctx=self.pass_ctx,
            func_args=args,
            func_kwargs=kwargs,
            resources=self.resources,
        )
        new.static_meta_data = self.static_meta_data.copy()
        return new

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


def quantity(
    *,
    pass_ctx: bool = False,
    resources: ResourceRequest | None = None,
) -> Callable[
    [WrappableQuantFunction[ParametersT_contra, QuantitiesT_co]],
    WrappedQuantityComputer[ParametersT_contra, QuantitiesT_co],
]:
    """
    Wrap a Python function as a ChemFit quantity computer.

    The decorated callable converts a parameter mapping into a quantity
    dictionary. The returned :class:`WrappedQuantityComputer` can be configured
    fluently with bound arguments and static metadata, then converted into an
    objective with ``with_loss()``.

    Args:
        pass_ctx: If ``True``, pass the current context as the keyword argument
            ``ctx`` in addition to the parameter mapping. Otherwise the
            callable receives only the parameters and any bound arguments.
        resources: Static resources required for one computation.

    Returns:
        A decorator that converts a compatible callable into a
        :class:`WrappedQuantityComputer`.

    Examples:
        Build a reusable quantity stage and attach a loss::

            @chemfit.quantity(resources={"cpu": 1})
            def model(parameters, *, scale):
                return {"prediction": scale * parameters["x"]}

            term = (
                model.bind(scale=2.0)
                .with_meta(dataset="training")
                .with_loss(squared_error, target=4.0)
                .with_meta(observable="prediction")
            )

    Notes:
        Static metadata is merged into the shared ``ctx.meta`` mapping. Existing
        context metadata is updated first, followed by quantity-computer
        metadata and then objective metadata, so the objective wins on key
        collisions.

    """

    def wrap(
        func: WrappableQuantFunction[ParametersT_contra, QuantitiesT_co],
    ) -> WrappedQuantityComputer[ParametersT_contra, QuantitiesT_co]:
        return WrappedQuantityComputer(
            func,
            pass_ctx=pass_ctx,
            resources=resources,
        )

    return wrap
