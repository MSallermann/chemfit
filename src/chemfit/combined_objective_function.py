from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic, Protocol, TypeVar

from typing_extensions import Self

from chemfit.abstract_objective_function import (
    ChildContextConfigurator,
    EvaluateContext,
    ObjectiveFunctor,
)
from chemfit.scheduling import (
    NodeOutcome,
    SchedulableCompositeObjective,
)
from chemfit.wrap_funcs import WrappedObjectiveFunctor

# Deliberately invariant: CombinedObjectiveFunction.add() mutates the stored
# objective list.  Treating one instance as accepting a wider or narrower
# parameter type could then allow an incompatible objective to be appended.
ParametersT = TypeVar("ParametersT", bound=Mapping[str, object])
ObjectiveLike = Callable[[ParametersT], float] | ObjectiveFunctor[ParametersT]


def transform_generic_callables(
    list_of_callables: Sequence[ObjectiveLike[ParametersT]],
) -> list[ObjectiveFunctor[ParametersT]]:
    res: list[ObjectiveFunctor[ParametersT]] = []
    for func in list_of_callables:
        if isinstance(func, ObjectiveFunctor):
            res.append(func)
        else:
            res.append(WrappedObjectiveFunctor(func))
    return res


class Reducer(Protocol):
    def __call__(self, terms: list[float], /) -> float: ...


class Aggregator(Protocol):
    def __call__(
        self,
        terms: list[float],
        quantities: list[dict[str, Any] | None],
        ctx: EvaluateContext,
        /,
    ) -> float: ...


def sum_reducer(terms: list[float]) -> float:
    return sum(terms)


def mean_reducer(terms: list[float]) -> float:
    return sum(terms) / len(terms)


def root_mean_reducer(terms: list[float]) -> float:
    return math.sqrt(mean_reducer(terms))


class ExceptionHandler(Protocol):
    def __call__(
        self, exception: Exception, ctx: EvaluateContext, idx: int, /
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


class WrappedReducer(Aggregator):
    def __init__(self, reducer: Reducer) -> None:
        """A reducer that is wrapped in order to be used like an Aggregator."""
        self.reducer = reducer

    def __call__(
        self,
        terms: Sequence[float],
        quantities: Sequence[dict[str, Any] | None],  # noqa: ARG002
        ctx: EvaluateContext,  # noqa: ARG002
    ) -> float:
        return self.reducer(list(terms))

    def to_reducer(self) -> Reducer:
        return self.reducer


class CombinedObjectiveFunction(
    ObjectiveFunctor[ParametersT],
    SchedulableCompositeObjective[ParametersT],
    Generic[ParametersT],
):
    def __init__(
        self,
        objective_functions: Sequence[ObjectiveLike[ParametersT]],
        weights: Sequence[float] | None = None,
        child_context_configurator: ChildContextConfigurator | None = None,
        reduction: Reducer | None = None,
        exception_handler: ExceptionHandler = raising_exception_handler,
        aggregator: Aggregator | None = None,
    ) -> None:
        """
        Initialize a combined objective from multiple weighted terms.

        Each objective term is evaluated independently in its own child
        context. The resulting term values are multiplied by their
        corresponding weights, optionally filtered through
        ``exception_handler`` if evaluation fails, and then combined by the
        configured reducer or aggregator.

        Generic callables are automatically wrapped as ``ObjectiveFunctor``
        instances.

        Args:
            objective_functions: Sequence of objective functors or compatible
                callables.
            weights: Optional non-negative weight for each objective term. If
                ``None``, all weights default to ``1.0``.
            child_context_configurator: Optional callable used to configure
                each spawned child context before term evaluation.
            reduction: Callable that reduces the list of weighted term values
                to a scalar. Defaults to :func:`sum_reducer`. Mutually exclusive
                with ``aggregator``.
            exception_handler: Callable used to handle exceptions raised
                during term evaluation. It may return a replacement value or
                ``None`` to skip the term entirely.
            aggregator: Context-aware callable that reduces weighted term values
                using child quantities and the parent context. Mutually exclusive
                with ``reduction``.

        Raises:
            AssertionError: If the number of weights does not match the
                number of objective functions, or if any weight is negative.
            ValueError: If both ``reduction`` and ``aggregator`` are supplied.

        """

        super().__init__()

        # Convert to list internally for mutability
        self.objective_functions = transform_generic_callables(objective_functions)

        self.child_context_configurator = child_context_configurator

        if reduction is not None and aggregator is not None:
            msg = "Specify either `reduction` or `aggregator`, not both."
            raise ValueError(msg)

        if aggregator is None:
            reducer = sum_reducer if reduction is None else reduction
            self.reduction: Aggregator = WrappedReducer(reducer)
        else:
            self.reduction = aggregator

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

    def child_objectives(self) -> tuple[ObjectiveFunctor[ParametersT], ...]:
        """Return the immediate objective terms in evaluation order."""

        return tuple(self.objective_functions)

    def _child_objectives(self) -> tuple[ObjectiveFunctor[ParametersT], ...]:
        """Return child objectives for recursive objective operations."""

        return self.child_objectives()

    def begin_composite_evaluation(
        self,
        parameters: ParametersT,
        ctx: EvaluateContext,
    ) -> Sequence[EvaluateContext]:
        """Begin the lifecycle and create configured child contexts."""

        try:
            self._begin_evaluation(parameters, ctx)
            child_contexts = ctx.spawn_children(
                self.n_terms(),
                self.child_context_configurator,
            )
        except Exception as exception:
            self._end_evaluation(ctx, exception)

            # A nested failed setup is later serialized by its parent. Match
            # direct serial evaluation by materializing any contexts that the
            # failed configurator managed to create. Root failures deliberately
            # leave their partial child batch unmaterialized.
            if getattr(ctx.temp, "_is_composite_child", False):
                ctx.collect_child_meta_data(recursive=True)
            raise

        for child_ctx in child_contexts:
            child_ctx.temp._is_composite_child = True  # noqa: SLF001

        return child_contexts

    def finish_composite_evaluation(
        self,
        child_outcomes: Sequence[NodeOutcome],
        ctx: EvaluateContext,
    ) -> float:
        """Interpret child outcomes, reduce them, and end the lifecycle."""

        try:
            terms: list[float | None] = []
            try:
                for idx, (outcome, child_ctx) in enumerate(
                    zip(child_outcomes, ctx._children, strict=True)  # noqa: SLF001
                ):
                    if isinstance(outcome, Exception):
                        term = self.exception_handler(outcome, child_ctx, idx)
                    else:
                        try:
                            term = outcome * self.weights[idx]
                        except Exception as exception:
                            term = self.exception_handler(exception, child_ctx, idx)
                    terms.append(term)
            finally:
                ctx.collect_child_meta_data(recursive=False)

            value = self._reduce_terms(terms, ctx)
            ctx.loss = value
        except BaseException as exception:
            self._end_evaluation(ctx, exception)
            raise
        else:
            self._end_evaluation(ctx, None)

        return value

    def n_terms(self) -> int:
        """Return the number of objective terms."""
        return len(self.weights)

    def add(
        self,
        obj_funcs: Sequence[ObjectiveLike[ParametersT]] | ObjectiveLike[ParametersT],
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

    def filter_terms(
        self, terms: list[float | None], ctx: EvaluateContext
    ) -> list[float]:
        """
        Filter out terms that are 'None', while recording the skipped terms in ctx.meta['skipped_indices'].

        Side effects:
            - Writes to ctx.meta['skipped_indices']
        """
        skipped_indices = []
        filtered_terms = []
        for i, t in enumerate(terms):
            if t is None:
                skipped_indices.append(i)
            else:
                filtered_terms.append(t)
        ctx.meta["skipped_indices"] = skipped_indices
        return filtered_terms

    def apply_reduction(self, terms: Sequence[float], ctx: EvaluateContext) -> float:
        child_quantities: list[dict[str, Any] | None] = []
        for idx, child in enumerate(ctx.meta["children"]):
            if idx not in ctx.meta["skipped_indices"]:
                child_quantities.append(child["quantities"])
        return self.reduction(list(terms), child_quantities, ctx)

    def _reduce_terms(
        self,
        terms: Sequence[float | None],
        ctx: EvaluateContext,
    ) -> float:
        """Compute this COB's result from precomputed immediate term results."""

        if len(terms) != self.n_terms():
            msg = f"Expected {self.n_terms()} terms, got {len(terms)}."
            raise ValueError(msg)

        ctx.meta["n_terms"] = self.n_terms()

        return self.apply_reduction(
            self.filter_terms(list(terms), ctx),
            ctx,
        )

    def evaluate_weighted_term(
        self, parameters: ParametersT, idx: int, ctx: EvaluateContext
    ):
        try:
            return self.objective_functions[idx](parameters, ctx) * self.weights[idx]
        except Exception as e:
            return self.exception_handler(e, ctx, idx)

    def _evaluate(
        self,
        parameters: ParametersT,
        ctx: EvaluateContext,
    ) -> float:
        with (
            ctx.child_contexts(
                n_children=self.n_terms(),
                configurator=self.child_context_configurator,
                recursive=False,  # <- in general nested COBs should manage child ctx retrieval
            ) as child_ctxs
        ):
            terms = [
                self.evaluate_weighted_term(
                    parameters,
                    idx,
                    child_ctx,
                )
                for idx, (objective, weight, child_ctx) in enumerate(
                    zip(
                        self.objective_functions,
                        self.weights,
                        child_ctxs,
                        strict=True,
                    )
                )
            ]

        return self._reduce_terms(terms, ctx)
