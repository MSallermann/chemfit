from __future__ import annotations

from typing import Any, Callable

from chemfit.abstract_objective_function import (
    EvaluateContext,
    ObjectiveFunctor,
    QuantityComputer,
)


class WrappedObjectiveFunctor(ObjectiveFunctor):
    def __init__(self, func: Callable[[dict[str, Any]], float]):
        """Wrap a generic callable in an objective functor."""
        super().__init__()
        self.func = func
        self.last_ctx: EvaluateContext | None = None

    def __call__(
        self, parameters: dict[str, Any], ctx: EvaluateContext | None = None
    ) -> float:
        if ctx is None:
            ctx = EvaluateContext()

        loss = self.func(parameters)
        ctx.loss = loss
        ctx.parameters = parameters

        return ctx.loss


def to_objective_functor(func: Callable[[dict[str, Any]], float]):
    return WrappedObjectiveFunctor(func)


class WrappedQuantityComputer(QuantityComputer):
    def __init__(self, func: Callable[[dict[str, Any]], dict[str, Any]]):
        """Wrap a generic callable in a quantity computer."""
        super().__init__()
        self.func = func

    def _compute(
        self,
        parameters: dict[str, Any],
        ctx: EvaluateContext,  # noqa: ARG002
    ) -> dict[str, Any]:
        return self.func(parameters)


def to_quantity_computer(func: Callable[[dict[str, Any]], dict[str, Any]]):
    return WrappedQuantityComputer(func)
