from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Protocol

from typing_extensions import Self

from chemfit.abstract_objective_function import (
    EvaluateContext,
    ObjectiveFunctor,
)
from chemfit.wrap_funcs import WrappedObjectiveFunctor


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
    def __call__(
        self, exception: Exception, ctx: EvaluateContext, idx: int
    ) -> float | None: ...


def raising_exception_handler(
    exception: Exception,
    ctx: EvaluateContext,  # noqa: ARG001
    idx: int,  # noqa: ARG001
) -> float | None:
    raise exception


def nan_exception_handler(
    exception: Exception,  # noqa: ARG001
    ctx: EvaluateContext,  # noqa: ARG001
    idx: int,  # noqa: ARG001
) -> float | None:
    return math.nan


def skip_exception_handler(
    exception: Exception,  # noqa: ARG001
    ctx: EvaluateContext,  # noqa: ARG001
    idx: int,  # noqa: ARG001
) -> float | None:
    return None


class ChildContextConfigurator(Protocol):
    """
    Protocol for configuring child evaluation contexts before term evaluation.

    The configurator is called once for each child context after the
    parent context has spawned them in ``prepare_evaluation()``. It may
    mutate the child context or the parent context in place to configure
    term-specific metadata, execution resources, or other evaluation
    settings.

    The ``idx_child_ctx`` and ``num_children`` arguments refer to the
    absolute term index and the total number of terms in the combined
    objective.

    """

    def __call__(
        self,
        idx_child_ctx: int,
        child_ctx: EvaluateContext,
        num_children: int,
        parent_ctx: EvaluateContext,
    ): ...


class CombinedObjectiveFunction(ObjectiveFunctor):
    def __init__(
        self,
        objective_functions: Sequence[Callable[[dict[str, Any]], float]],
        weights: Sequence[float] | None = None,
        child_context_configurator: ChildContextConfigurator | None = None,
        reduction: Reducer = sum_reducer,
        exception_handler: ExceptionHandler = raising_exception_handler,
    ) -> None:
        """
        Initialize a combined objective from multiple weighted terms.

        Each objective term is evaluated independently in its own child
        context. The resulting term values are multiplied by their
        corresponding weights, optionally filtered through
        ``exception_handler`` if evaluation fails, and then combined using
        ``reduction``.

        Generic callables are automatically wrapped as ``ObjectiveFunctor``
        instances.

        Args:
            objective_functions: Sequence of objective functors or compatible
                callables.
            weights: Optional non-negative weight for each objective term. If
                ``None``, all weights default to ``1.0``.
            child_context_configurator: Optional callable used to configure
                each spawned child context before term evaluation.
            reduction: Callable used to reduce the list of weighted term
                values to a single scalar loss.
            exception_handler: Callable used to handle exceptions raised
                during term evaluation. It may return a replacement value or
                ``None`` to skip the term entirely.

        Raises:
            AssertionError: If the number of weights does not match the
                number of objective functions, or if any weight is negative.

        """

        # Convert to list internally for mutability
        self.objective_functions: list[ObjectiveFunctor] = transform_generic_callables(
            objective_functions
        )

        self.child_context_configurator = child_context_configurator
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
        """Return the number of objective terms."""
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
        Add one or more objective terms to the combined objective.

        Each added callable is converted to an ``ObjectiveFunctor`` if
        needed and appended to the existing term list. The corresponding
        weights are appended in the same order.

        Args:
            obj_funcs: A single objective callable or a sequence of objective
                callables to add.
            weights: Either a single non-negative weight applied to every
                added callable, or a sequence of non-negative weights whose
                length matches the number of added callables.

        Returns:
            The current instance.

        Raises:
            AssertionError: If a sequence of weights is provided with a
                length that does not match the number of added callables, or
                if any provided weight is negative.

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
        Flatten multiple combined objectives into a single instance.

        The objective functions from all input instances are concatenated
        into one flat list. The weights of each input instance are scaled by
        the corresponding entry in ``weights`` before concatenation.

        Warning:
            This method **does not preserve execution policy** from the input
            combined objectives. The resulting instance uses the default
            ``reduction``, ``exception_handler``, and
            ``child_context_configurator`` unless they are explicitly set
            afterward.

        Args:
            combined_objective_functions_list: Combined objective instances to
                flatten.
            weights: Optional non-negative scaling weights, one per input
                combined objective. If ``None``, all scaling weights default
                to ``1.0``.

        Returns:
            A new combined objective containing all flattened terms and
            scaled weights.

        Raises:
            AssertionError: If the number of scaling weights does not match
                the number of combined objectives, or if any scaling weight
                is negative.

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

    def prepare_child_contexts(
        self, ctx: EvaluateContext, child_contexts: Iterable[EvaluateContext]
    ):
        """
        Prepare child contexts for term evaluation.

        This method applies ``child_context_configurator`` to each spawned child context.

        Args:
            ctx: Parent evaluation context
            child_contexts: Child contexts that will be prepared

        Side effects:
            - Arbitrary changes in the child contexts

        """

        if self.child_context_configurator is not None:
            for idx_cur_ctx, child_ctx in enumerate(child_contexts):
                self.child_context_configurator(
                    idx_child_ctx=idx_cur_ctx,
                    child_ctx=child_ctx,
                    num_children=self.n_terms(),
                    parent_ctx=ctx,
                )

    def evaluate_term(
        self,
        parameters: dict[str, Any],
        idx: int,
        ctx: EvaluateContext,
    ) -> float | None:
        """
        Evaluate a single weighted objective term.

        The selected objective function is evaluated with the provided
        parameters and child context, then multiplied by its corresponding
        weight. If evaluation raises an exception, the configured
        ``exception_handler`` is called.

        Args:
            parameters: Parameter dictionary for the current evaluation.
            idx: Absolute index of the objective term to evaluate.
            ctx: Child evaluation context for this term.

        Returns:
            The weighted term value, or ``None`` if the exception handler
            chooses to skip the term.

        """

        try:
            return self.objective_functions[idx](parameters, ctx) * self.weights[idx]
        except Exception as e:
            return self.exception_handler(e, ctx, idx)

    def evaluate_terms(
        self, parameters: dict[str, Any], ctx: EvaluateContext
    ) -> list[float]:
        """
        Evaluate the objective terms.

        This method prepares child contexts, evaluates each selected term in
        its own context, and drops any terms for which ``evaluate_term()``
        returns ``None``.

        Args:
            parameters: Parameter dictionary for the current evaluation.
            ctx: Parent evaluation context.

        Returns:
            List of weighted term values that were successfully evaluated and
            not skipped by the exception handler.

        """

        ctx.meta.update({"n_terms": self.n_terms()})

        with ctx.child_contexts(n_children=self.n_terms()) as child_ctxs:
            self.prepare_child_contexts(ctx, child_contexts=child_ctxs)

            terms = []

            for idx, ctx_term in enumerate(child_ctxs):
                terms.append(self.evaluate_term(parameters, idx, ctx_term))

            return [t for t in terms if t is not None]

    def __call__(
        self,
        parameters: dict[str, Any],
        ctx: EvaluateContext | None = None,
    ) -> float:
        """
        Evaluate the combined objective.

        Each objective term is evaluated in its own child context using the
        same parameter dictionary. The weighted term values are combined
        using ``self.reduction``. After evaluation, child metadata is
        collected into the parent context and the reduced loss is stored in
        ``ctx.loss``.

        Args:
            parameters: Parameter dictionary for the evaluation.
            ctx: Optional parent evaluation context. If ``None``, a new
                ``EvaluateContext`` is created.

        Returns:
            The reduced scalar loss computed from the evaluated terms.

        Side Effects:
            - Populates ``ctx.parameters``.
            - Spawns child contexts in ``ctx``.
            - Collects child metadata into ``ctx.meta["children"]``.
            - Stores the final reduced loss in ``ctx.loss``.

        """

        if ctx is None:
            ctx = EvaluateContext()

        ctx.parameters = parameters

        terms = self.evaluate_terms(parameters=parameters, ctx=ctx)

        ctx.loss = self.reduction(terms)
        return ctx.loss
