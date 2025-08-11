import abc
from typing import Any, Protocol, runtime_checkable


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


@runtime_checkable
class SupportsGetMetaData(Protocol):
    def get_meta_data(self) -> dict[str, Any]: ...
