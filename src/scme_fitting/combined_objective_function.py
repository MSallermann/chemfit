from typing import Sequence, Callable, Optional, Self, Union, Dict
from collections.abc import Sequence as ABCSequence


class CombinedObjectiveFunction:
    """
    Represents a weighted sum of multiple objective functions.

    Each objective function accepts a dictionary of parameters (str → float) and returns a float.
    Internally, each function is paired with a non-negative weight. Calling the instance returns
    the weighted sum of all objective-function evaluations.
    """

    def __init__(
        self,
        objective_functions: Sequence[Callable[[Dict[str, float]], float]],
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Initialize a CombinedObjectiveFunction.

        Args:
            objective_functions (Sequence[Callable[[Dict[str, float]], float]]):
                A sequence of callables. Each callable must accept a dictionary mapping parameter
                names (str) to values (float) and return a float.
            weights (Sequence[float], optional):
                A sequence of non-negative floats specifying the weight for each objective function.
                If None, all weights default to 1.0.

        Raises:
            AssertionError: If `weights` is provided but its length differs from the number of
                objective functions, or if any weight is negative.
        """
        # Convert to list internally for mutability
        self.objective_functions: list[Callable[[Dict[str, float]], float]] = list(
            objective_functions
        )

        if weights is None:
            # Default each weight to 1.0
            self.weights: list[float] = [1.0] * len(self.objective_functions)
        else:
            self.weights = list(weights)

        # Ensure alignment between objective functions and weights
        assert len(self.weights) == len(self.objective_functions), (
            "Number of weights must match number of objective functions."
        )
        # Ensure all weights are non-negative
        assert all(w >= 0 for w in self.weights), "All weights must be non-negative."

    def n_terms(self) -> int:
        """
        Return the number of objective terms.

        Returns:
            int: The number of (function, weight) pairs stored internally.
        """
        return len(self.weights)

    def add(
        self,
        obj_funcs: Union[
            Sequence[Callable[[Dict[str, float]], float]],
            Callable[[Dict[str, float]], float],
        ],
        weights: Union[Sequence[float], float] = 1.0,
    ) -> Self:
        """
        Add one or more objective functions (and corresponding weights) to this instance.

        If `obj_funcs` is a single callable, it is appended; if it is a sequence of callables,
        each is appended in order. The `weights` argument must align:
        - If `weights` is a single float, that same weight is used for each newly added function.
        - If `weights` is a sequence, its length must match the number of functions being added.

        Args:
            obj_funcs (Callable[[Dict[str, float]], float]
                or Sequence[Callable[[Dict[str, float]], float]]):
                Either a single objective-function callable or a sequence of such callables. Each callable
                must accept a `Dict[str, float]` and return a float.
            weights (float or Sequence[float], optional):
                Either a float (used for every new function) or a sequence of non-negative floats.
                If a sequence, its length must equal the number of functions in `obj_funcs`.
                Defaults to 1.0.

        Returns:
            Self: The current instance (allows chaining).

        Raises:
            AssertionError: If `weights` is a sequence but its length does not match the number
                of functions in `obj_funcs`, or if any provided weight is negative.
        """

        # Determine how many new functions are being added
        if isinstance(obj_funcs, ABCSequence) and not callable(obj_funcs):
            funcs_to_add = list(obj_funcs)  # type: ignore[assignment]
        else:
            funcs_to_add = [obj_funcs]  # type: ignore[assignment]

        # Append each new objective function
        for fn in funcs_to_add:
            self.objective_functions.append(fn)

        # Handle weights
        if isinstance(weights, ABCSequence) and not isinstance(weights, (str, bytes)):
            weights_to_add = list(weights)  # type: ignore[assignment]
            # Must match number of new functions
            assert len(weights_to_add) == len(funcs_to_add), (
                "Length of weights sequence must equal number of functions added."
            )
        else:
            # Single weight repeated for each new function
            weights_to_add = [float(weights) for _ in funcs_to_add]

        # Ensure all new weights are non-negative
        assert all(w >= 0 for w in weights_to_add), "All weights must be non-negative."

        # Append the new weights
        self.weights.extend(weights_to_add)

        # Final sanity check that lists remain aligned
        assert len(self.weights) == len(self.objective_functions), (
            "After adding, weights and objective_functions must remain the same length."
        )

        return self

    @classmethod
    def add_flat(
        cls,
        combined_objective_functions_list: Sequence[Self],
        weights: Sequence[float],
    ) -> Self:
        """
        Create a new, "flat" CombinedObjectiveFunction by merging multiple existing instances.

        Each input instance is scaled by its corresponding weight, and all internal objective functions
        are concatenated into a single-level structure.

        Args:
            combined_objective_functions_list (Sequence[CombinedObjectiveFunction]):
                A sequence of CombinedObjectiveFunction instances to combine.
            weights (Sequence[float]):
                A sequence of non-negative floats, one per CombinedObjectiveFunction. Each sub-instance's
                internal weights are multiplied by its associated weight.

        Returns:
            CombinedObjectiveFunction: A new instance whose `objective_functions` list is the
                concatenation of all sub-instances' objective functions, and whose `weights` list
                is the scaled and concatenated weights.

        Raises:
            AssertionError: If the lengths of `combined_objective_functions_list` and `weights` differ,
                or if any weight is negative.
        """
        # Ensure we have one scaling weight per sub-instance
        assert len(combined_objective_functions_list) == len(weights), (
            "Must supply exactly one weight per CombinedObjectiveFunction."
        )
        # Ensure all scaling weights are non-negative
        assert all(w >= 0 for w in weights), "All scaling weights must be non-negative."

        total_objective_functions: list[Callable[[Dict[str, float]], float]] = []
        total_weights: list[float] = []

        for sub_cob, scale in zip(combined_objective_functions_list, weights):
            total_objective_functions.extend(sub_cob.objective_functions)
            # Scale each internal weight
            total_weights.extend([w * scale for w in sub_cob.weights])

        # Ensure no negative weights after scaling
        assert all(w >= 0 for w in total_weights), (
            "Resulting weights must be non-negative."
        )

        return cls(total_objective_functions, total_weights)

    def __call__(self, params: Dict[str, float]) -> float:
        """
        Evaluate the combined objective at a given parameter dictionary.

        Each individual objective function is called (with a shallow copy of `params`), multiplied
        by its weight, and summed into a single scalar result.

        Args:
            params (Dict[str, float]): A dictionary mapping parameter names (str) to values (float).
                A copy is made for each objective function call to guard against in-place modifications.

        Returns:
            float: The weighted sum of all objective-function evaluations.
        """
        total: float = 0.0

        for idx, weight in enumerate(self.weights):
            p_copy = params.copy()
            total += self.objective_functions[idx](p_copy) * weight

        return total
