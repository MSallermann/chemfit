from __future__ import annotations

import abc
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class SupportsGetMetaData(Protocol):
    def get_meta_data(self) -> dict[str, Any]: ...


class ObjectiveFunctor(abc.ABC):
    @abc.abstractmethod
    def get_meta_data(self) -> dict[str, Any]:
        """Get meta data."""
        ...

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

        self._last_quantities: dict[str, Any] | None = None

    def get_meta_data(self) -> dict[str, Any]:
        """Get meta data."""
        return {"last": self._last_quantities}

    def __call__(self, parameters: dict[str, Any]) -> dict[str, Any]:
        self._last_quantities = self._compute(parameters)
        return self._last_quantities

    @abc.abstractmethod
    def _compute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Compute dictionary of quantities for a given set of new parameters."""
        ...


class QuantityComputerObjectiveFunction(ObjectiveFunctor):
    def __init__(
        self,
        loss_function: Callable[[dict[str, Any]], float] | ObjectiveFunctor,
        quantity_computer: QuantityComputer,
    ) -> None:
        """Initialize the objective function with a quantity computer."""

        super().__init__()
        self.quantity_computer = quantity_computer
        self.loss_function = loss_function
        self._last_loss: float | None = None

    def get_meta_data(self) -> dict[str, Any]:
        meta_data = {
            "computer": self.quantity_computer.get_meta_data(),
            "last_loss": self._last_loss,
        }

        if isinstance(self.loss_function, SupportsGetMetaData):
            meta_data["loss_function"] = self.loss_function.get_meta_data()

        return meta_data

    def __call__(self, parameters: dict[str, Any]) -> float:
        quantities = self.quantity_computer(parameters)
        self._last_loss = self.loss_function(quantities)

        return self._last_loss
