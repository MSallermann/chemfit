from __future__ import annotations

import abc
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class SupportsGetMetaData(Protocol):
    def get_meta_data(self) -> dict[str, Any]: ...


@dataclass
class EvaluateContext:
    quantities: dict[str, Any] | None = None
    parameters: dict[str, Any] | None = None
    loss: float | None = None
    static: dict[str, Any] | None = None

    def to_meta_data(self) -> dict[str, Any]:
        return asdict(self)


class ObjectiveFunctor(abc.ABC):
    @abc.abstractmethod
    def get_meta_data(self) -> dict[str, Any]:
        """Get meta data."""
        ...

    def evaluate(self, parameters: dict[str, Any], ctx: EvaluateContext) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, parameters: dict[str, Any]) -> float:
        """
        Compute the objective value given a set of parameters.

        Args:
            parameters: Dictionary of parameter names to float values.

        Returns:
            float: Computed objective value (e.g., error metric).

        """
        ...


class QuantityComputer(abc.ABC):
    def __init__(self):
        """Initialize the QuantityComputer."""
        self.static_meta_data: dict[str, Any] | None = None  # For static meta data

    def evaluate(
        self, parameters: dict[str, Any], ctx: EvaluateContext
    ) -> dict[str, Any]:
        """Evaluate the quantities without changing internal state."""

        ctx.parameters = parameters
        ctx.quantities = self._compute(parameters, ctx)

        if ctx.static is None:
            ctx.static = self.static_meta_data
        elif self.static_meta_data is not None:
            ctx.static.update(self.static_meta_data)

        return ctx.quantities

    @abc.abstractmethod
    def _compute(
        self, parameters: dict[str, Any], ctx: EvaluateContext
    ) -> dict[str, Any]:
        """Compute dictionary of quantities for a given set of new parameters."""
        ...


class QuantityComputerObjectiveFunction(ObjectiveFunctor):
    def __init__(
        self,
        loss_function: Callable[[dict[str, Any]], float]
        | Callable[[dict[str, Any], dict[str, Any]], float],
        quantity_computer: QuantityComputer,
    ) -> None:
        """Initialize the objective function with a quantity computer."""

        super().__init__()
        self.quantity_computer = quantity_computer
        self.static_meta_data: dict[str, Any] | None = None
        self.loss_function = loss_function
        self.last_ctx: EvaluateContext | None = None

    def get_meta_data(self) -> dict[str, Any]:
        if self.last_ctx is None:
            return {}

        return self.last_ctx.to_meta_data()

    def evaluate(self, parameters: dict[str, Any], ctx: EvaluateContext) -> float:
        quantities = self.quantity_computer.evaluate(parameters, ctx)

        # Update or set static meta data if needed
        if ctx.static is None:
            ctx.static = self.static_meta_data
        elif self.static_meta_data is not None:
            ctx.static.update(self.static_meta_data)

        try:
            ctx.loss = self.loss_function(quantities)  # pyright: ignore[reportCallIssue] # we actually handle this with the signature checking
        except TypeError:
            ctx.loss = self.loss_function(quantities, parameters)  # pyright: ignore[reportCallIssue] # we actually handle this with the signature checking

        return ctx.loss

    def __call__(self, parameters: dict[str, Any]) -> float:
        """Evaluate the quantities."""
        ctx = EvaluateContext()
        self.last_ctx = ctx
        return self.evaluate(parameters, ctx)
