from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Callable, Protocol

from typing_extensions import Self

from chemfit.abstract_objective_function import (
    EvaluateContext,
    ObjectiveFunctor,
)
from chemfit.wrap_funcs import WrappedObjectiveFunctor

DEFAULT_SLICE = slice(None, None, None)


def transform_generic_callables(
    list_of_callables: Sequence[Callable[[dict[str, Any]], float]],
) -> list[ObjectiveFunctor]:
    res = []
    for func in list_of_callables:
        if isinstance(func, ObjectiveFunctor):
            res.append(func)
        else:
            res.append(WrappedObjectiveFunctor(func))
    return res


class Reducer(Protocol):
    def __call__(self, terms: Sequence[float]) -> float: ...


def sum_reducer(terms: Sequence[float]) -> float:
    return sum(terms)


def mean_reducer(terms: Sequence[float]) -> float:
    return sum(terms) / len(terms)


def root_mean_reducer(terms: Sequence[float]) -> float:
    return math.sqrt(mean_reducer(terms))


class ExceptionHandler(Protocol):
    def __call__(self, exception: Exception) -> float | None: ...


def raising_exception_handler(exception: Exception) -> float | None:
    raise exception


def nan_exception_handler(exception: Exception) -> float | None:  # noqa: ARG001
    return math.nan


def skip_exception_handler(exception: Exception) -> float | None:  # noqa: ARG001
    return None


class ContextModifer(Protocol):
    """
    Protocol to modify the list of child contexts after they have been spawned by the parent context.

    `idx_child_ctx` and `num_ctx` refer to the absolute index of the child context within all terms, **even if a slice is used**.

    If really needed, the *relative* index within the slice can be computed from the information in the `parent_ctx` as follows:

    >>> indices = range(parent_ctx.meta["n_terms"])
    >>> start, stop, step = parent_ctx.meta["slice_start"], parent_ctx.meta["slice_stop"], parent_ctx.meta["slice_step"]
    >>> idx_relative = indices[start:stop:step]

    """

    def __call__(
        self,
        idx_child_ctx: int,
        child_ctx: EvaluateContext,
        num_ctx: int,
        parent_ctx: EvaluateContext,
    ): ...


class CombinedObjectiveFunction(ObjectiveFunctor):
    def __init__(
        self,
        objective_functions: Sequence[Callable[[dict[str, Any]], float]],
        weights: Sequence[float] | None = None,
        context_modifier: ContextModifer | None = None,
        reduction: Reducer = sum_reducer,
        exception_handler: ExceptionHandler = raising_exception_handler,
    ) -> None:
        """
        Weighted sum of multiple objective functions.

        A `CombinedObjectiveFunction` aggregates several `ObjectiveFunctor`
        instances (or generic callables) and evaluates them as a weighted sum.
        Each term receives its own `EvaluateContext`, while the main context
        accumulates metadata and the final combined loss.

        The evaluation is sliceable: callers may evaluate only a subset of
        terms using the ``idx_slice`` argument in ``__call__``.

        Args:
            objective_functions (Sequence[Callable[[dict], float]]):
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
        self.objective_functions: list[ObjectiveFunctor] = transform_generic_callables(
            objective_functions
        )

        self.context_modifier = context_modifier
        self.reduction = reduction
        self.exception_handler = exception_handler

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
        obj_funcs: (
            Sequence[Callable[[dict[str, Any]], float]]
            | Callable[[dict[str, Any]], float]
        ),
        weights: Sequence[float] | float = 1.0,
    ) -> Self:
        """
        Add one or more objective functions (and corresponding weights) to this instance.

        If `obj_funcs` is a single callable, it is appended; if it is a sequence of callables,
        each is appended in order. The `weights` argument must align:
        - If `weights` is a single float, that same weight is used for each newly added function.
        - If `weights` is a sequence, its length must match the number of functions being added.

        Args:
            obj_funcs (Callable[dict], float]
                or Sequence[Callable[[dict], float]]):
                Either a single objective-function callable or a sequence of such callables. Each callable
                must accept a `dict` and return a float.
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
        if isinstance(obj_funcs, Sequence) and not callable(obj_funcs):
            funcs_to_add = list(obj_funcs)  # type: ignore[assignment]
        else:
            funcs_to_add = [obj_funcs]  # type: ignore[assignment]

        funcs_to_add = transform_generic_callables(funcs_to_add)

        # Append each new objective function
        for fn in funcs_to_add:
            self.objective_functions.append(fn)

        # Handle weights
        if isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)):
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
        weights: Sequence[float] | None = None,
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
        if weights is None:
            weights = [1.0 for _ in combined_objective_functions_list]

        # Ensure we have one scaling weight per sub-instance
        assert len(combined_objective_functions_list) == len(weights), (
            "Must supply exactly one weight per CombinedObjectiveFunction."
        )

        # Ensure all scaling weights are non-negative
        assert all(w >= 0 for w in weights), "All scaling weights must be non-negative."

        total_objective_functions: list[Callable[[dict[str, Any]], float]] = []
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

    def prepare_evaluation(
        self,
        parameters: dict[str, Any],
        ctx: EvaluateContext,
        idx_slice: slice = DEFAULT_SLICE,
    ) -> tuple[list[EvaluateContext], list[int]]:
        """
        Prepare per-term contexts for evaluation.

        This method allocates one `EvaluateContext` per selected term and
        writes basic metadata into the main context.

        Args:
            parameters (dict[str, Any]): Parameter dictionary for the evaluation.
            ctx (EvaluateContext): Main context that will accumulate metadata
                and final loss.
            idx_slice (slice): Slice selecting which terms to evaluate.

        Returns:
            tuple[list[EvaluateContext], list[int]]:
                - List of new per-term contexts.
                - List of indices corresponding to the selected terms.

        """

        ctx.loss = 0.0
        ctx.parameters = parameters
        ctx.meta.update(
            {
                "n_terms": self.n_terms(),
                "slice_start": idx_slice.start,
                "slice_stop": idx_slice.stop,
                "slice_step": idx_slice.step,
            }
        )

        idx_list = list(range(self.n_terms()))
        sliced_idx_list = idx_list[idx_slice]

        contexts = ctx.spawn_children(n_children=len(sliced_idx_list))

        if self.context_modifier is not None:
            [
                self.context_modifier(
                    idx_child_ctx=idx_cur_ctx,
                    child_ctx=child_ctx,
                    num_ctx=self.n_terms(),
                    parent_ctx=ctx,
                )
                for idx_cur_ctx, child_ctx in zip(sliced_idx_list, contexts)
            ]

        return contexts, idx_list

    def evaluate_term(
        self, parameters: dict[str, Any], ctx_term: EvaluateContext, idx: int
    ) -> float | None:
        try:
            return (
                self.objective_functions[idx](parameters, ctx_term) * self.weights[idx]
            )
        except Exception as e:
            return self.exception_handler(e)

    def evaluate_terms(
        self,
        parameters: dict[str, Any],
        ctx: EvaluateContext,
        idx_slice: slice = DEFAULT_SLICE,
    ) -> list[float]:
        contexts, idx_list = self.prepare_evaluation(
            parameters=parameters, ctx=ctx, idx_slice=idx_slice
        )

        terms = []

        for idx, ctx_term in zip(idx_list[idx_slice], contexts, strict=True):
            terms.append(self.evaluate_term(parameters, ctx_term, idx))

        return [t for t in terms if t is not None]

    def __call__(
        self,
        parameters: dict[str, Any],
        ctx: EvaluateContext | None = None,
    ) -> float:
        """
        Evaluate the weighted sum of all selected objective terms.

        Each selected term receives its own `EvaluateContext` (created by
        `prepare_evaluation`), and its contribution is accumulated into
        the main context's `loss` value.

        Args:
            parameters (dict[str, Any]):
                Parameter dictionary for this evaluation.
            ctx (EvaluateContext | None):
                Evaluation context. If None, a new one is created.
            idx_slice (slice):
                Slice selecting which terms to evaluate. Defaults to all terms.

        Returns:
            float: Weighted sum of the selected objective-function values.

        Notes:
            - Each term uses the **same** parameter dictionary.
            - Each term uses a **distinct** `EvaluateContext`.
            - Per-term data is merged into the main context after all terms
              have been evaluated.

        """

        if ctx is None:
            ctx = EvaluateContext()

        terms = self.evaluate_terms(
            parameters=parameters, ctx=ctx, idx_slice=DEFAULT_SLICE
        )

        ctx.collect_child_meta_data()

        ctx.loss = self.reduction(terms)
        return ctx.loss
