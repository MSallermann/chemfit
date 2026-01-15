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
        """Wrap a generic callable in an objective functor."""
        super().__init__()
        self.func = func
        self.pass_ctx = pass_ctx

    def __call__(
        self, parameters: dict[str, Any], ctx: EvaluateContext | None = None
    ) -> float:
        if ctx is None:
            ctx = EvaluateContext()

        if self.pass_ctx:
            ctx.loss = self.func(parameters, ctx)
        else:
            ctx.loss = self.func(parameters)

        ctx.parameters = parameters

        return ctx.loss


def to_objective_functor(pass_ctx: bool = False):
    def wrap(func: WrappableObjFunction):
        return WrappedObjectiveFunctor(func, pass_ctx=pass_ctx)

    return wrap


WrappableQuantFunction = (
    Callable[[dict[str, Any]], dict[str, Any]]
    | Callable[[dict[str, Any], EvaluateContext], dict[str, Any]]
)


class WrappedQuantityComputer(QuantityComputer):
    def __init__(self, func: WrappableQuantFunction, pass_ctx: bool = False):
        """Wrap a generic callable in a quantity computer."""
        super().__init__()
        self.func = func
        self.pass_ctx = pass_ctx

    def _compute(
        self,
        parameters: dict[str, Any],
        ctx: EvaluateContext,
    ) -> dict[str, Any]:
        if self.pass_ctx:
            ctx.quantities = self.func(parameters, ctx)
        else:
            ctx.quantities = self.func(parameters)

        return ctx.quantities


def to_quantity_computer(pass_ctx: bool = False):
    def wrap(func: WrappableQuantFunction):
        return WrappedQuantityComputer(func, pass_ctx=pass_ctx)

    return wrap
