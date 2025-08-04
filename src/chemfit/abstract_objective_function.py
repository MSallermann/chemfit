import abc


class ObjectiveFunctor(abc.ABC):

    @abc.abstractmethod
    def get_meta_data(self) -> dict:
        """Get meta data"""
        ...

    @abc.abstractmethod
    def __call__(self, parameters: dict) -> float:
        """Compute the objective value given a set of parameters.

        Args:
            parameters: Dictionary of parameter names to float values.

        Returns:
            float: Computed objective value (e.g., error metric).

        """
        ...
