from __future__ import annotations

import inspect
import math
import numbers
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, TypeVar

from typing_extensions import Self

from chemfit.abstract_objective_function import (
    EvaluateContext,
    JoinedComponent,
    NoOutput,
    ObjectiveComponent,
    WrappedComponent,
)

ParametersT = TypeVar("ParametersT", bound=Mapping[str, object])
QuantitiesT_co = TypeVar("QuantitiesT_co", bound=Mapping[str, object])
ObjectiveLike = Callable[[ParametersT], float] | ObjectiveComponent[ParametersT, float]


def transform_generic_callables(
    list_of_callables: Sequence[ObjectiveLike[ParametersT]],
) -> list[ObjectiveComponent[ParametersT, float]]:
    res: list[ObjectiveComponent[ParametersT, float]] = []

    for func in list_of_callables:
        if isinstance(func, ObjectiveComponent):
            res.append(func)
        else:
            res.append(WrappedComponent(func))
    return res


def _validate_weights(weights: Sequence[float], n_objectives: int) -> None:
    if len(weights) != n_objectives:
        msg = (
            "`weights` must contain one value per objective function "
            f"(expected {n_objectives}, got {len(weights)})."
        )
        raise ValueError(msg)

    if any(weight < 0 for weight in weights):
        msg = "`weights` must be non-negative."
        raise ValueError(msg)

    if any(
        math.isnan(weight) or weight in {-math.inf, +math.inf} for weight in weights
    ):
        msg = "`weights` must not be NaN and finite."
        raise ValueError(msg)


class Reducer(Protocol):
    def __call__(self, terms: list[float], *, ctx: EvaluateContext) -> float: ...


class SimpleReducer(Protocol):
    def __call__(self, terms: list[float]) -> float: ...


def sum_reducer(terms: list[float]) -> float:
    return sum(terms)


def mean_reducer(terms: list[float]) -> float:
    return sum(terms) / len(terms)


def root_mean_reducer(terms: list[float]) -> float:
    return math.sqrt(mean_reducer(terms))


def accepts_ctx(reducer: Callable) -> bool:
    sig = inspect.signature(reducer)

    param = sig.parameters.get("ctx")
    if param is not None:
        return param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )

    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )


class WrappedReducer:
    def __init__(self, reducer: SimpleReducer | Reducer):
        self.reducer: SimpleReducer | Reducer = reducer
        self.pass_ctx: bool = accepts_ctx(reducer)

    def __call__(self, terms: list[float], *, ctx: EvaluateContext) -> float:
        if self.pass_ctx:
            return self.reducer(terms, ctx=ctx)  # type: ignore[call-arg]
        return self.reducer(terms)  # type: ignore[call-arg]


def wrap_reducer(reducer: SimpleReducer | Reducer) -> Reducer:
    return WrappedReducer(reducer)


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


class CombinedObjectiveFunction(JoinedComponent[ParametersT, float]):
    def __init__(
        self,
        objective_functions: Sequence[ObjectiveLike[ParametersT]],
        weights: Sequence[float] | None = None,
        reduction: Reducer | SimpleReducer | None = None,
        exception_handler: ExceptionHandler = raising_exception_handler,
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
            reduction: Callable that reduces the list of weighted term values
                to a scalar. Defaults to :func:`sum_reducer`. Mutually exclusive
                with ``aggregator``.
            exception_handler: Callable used to handle exceptions raised
                during term evaluation. It may return a replacement value or
                ``None`` to skip the term entirely.

        Raises:
            ValueError: If no objective functions are supplied, both ``reduction``
                and ``aggregator`` are supplied, the number of weights does not
                match the number of objective functions, or any weight is negative.

        """

        super().__init__(
            components=tuple(transform_generic_callables(objective_functions))
        )

        self.reduction: Reducer = wrap_reducer(
            sum_reducer if reduction is None else reduction
        )

        self.exception_handler = exception_handler
        if weights is None:
            configured_weights = [1.0] * self.n_components()
        else:
            configured_weights = list(weights)

        _validate_weights(configured_weights, self.n_components())
        self._weights = tuple(configured_weights)

    def with_weights(self, weights: Sequence[float], /) -> Self:
        """
        Return a copy with replacement weights for the current terms.

        Args:
            weights: One non-negative weight for each current objective term.
                The supplied sequence is copied into an internal list.

        Returns:
            A configured copy; the source objective is unchanged.

        Raises:
            ValueError: If the number of weights does not match the number of
                terms or any weight is negative.

        Examples:
            Configure two terms without changing the original objective::

                objective = chemfit.combine(term_a, term_b).with_weights(
                    [1.0, 0.2]
                )

        """

        new_weights = list(weights)
        _validate_weights(new_weights, self.n_components())
        new = self.copy()
        new._weights = tuple(new_weights)  # noqa: SLF001
        return new

    def with_reduction(
        self,
        reduction: SimpleReducer | Reducer,
    ) -> Self:
        """
        Return a copy using a simple reducer for weighted term values.

        Args:
            reduction: Callable that reduces the weighted values to one float.

        Returns:
            A configured copy; the source objective is unchanged.

        Notes:
            This replaces any reducer or aggregator configured earlier in the
            fluent chain. A later :meth:`with_aggregator` call replaces it in
            turn.

        Examples:
            Use the mean of the weighted terms::

                objective = chemfit.combine(terms).with_reduction(
                    chemfit.mean_reducer
                )

        """

        new = self.copy()
        new.reduction = wrap_reducer(reduction)
        return new

    def with_exception_handler(
        self,
        exception_handler: ExceptionHandler,
        /,
    ) -> Self:
        """
        Return a copy using a replacement term-exception handler.

        Args:
            exception_handler: Callable that re-raises, replaces, or skips a
                failed term.

        Returns:
            A configured copy; the source objective is unchanged.

        Notes:
            Repeated calls replace the previous handler, so the last call in a
            fluent chain determines how failed terms contribute.

        Examples:
            Convert failed terms to NaN before applying a mean reduction::

                objective = (
                    chemfit.combine(terms)
                    .with_exception_handler(chemfit.nan_exception_handler)
                    .with_reduction(chemfit.mean_reducer)
                )

        """

        new = self.copy()
        new.exception_handler = exception_handler
        return new

    def combine_outcomes(
        self, outcomes: Sequence[float | Exception | NoOutput], ctx: EvaluateContext
    ) -> float:
        """
        Combine outcomes.

        Applies exception handler and filters out terms that are 'None'
        while recording the skipped terms in ctx.meta['skipped_indices'].

        Side effects:
            - Writes to ctx.meta['skipped_indices']

        """

        outcomes_filtered: list[float | None | Exception | NoOutput] = list(outcomes)

        # apply exception handlers to outcomes
        for i in range(self.n_components()):
            t = outcomes[i]
            if isinstance(t, Exception):
                outcomes_filtered[i] = self.exception_handler(t, ctx.incoming[i], i)

        skipped_indices = []
        filtered_terms = []
        for i, (t, w) in enumerate(zip(outcomes_filtered, self._weights, strict=True)):
            if t is None:
                skipped_indices.append(i)
            elif isinstance(t, numbers.Real):
                filtered_terms.append(t * w)
            else:
                msg = f"Invalid term encountered: index {i}, term {t}"
                raise ValueError(msg)

        ctx.meta["skipped_indices"] = skipped_indices

        return self.reduction(list(filtered_terms), ctx=ctx)
