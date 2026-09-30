from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic, Protocol, TypeVar, cast

from typing_extensions import Self

from chemfit.abstract_objective_function import (
    ChildContextConfigurator,
    EvaluateContext,
    ObjectiveFunctor,
)
from chemfit.scheduling import (
    PreparedSchedule,
    PreparedScheduleBase,
    Scheduler,
    SchedulingProfile,
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


def evaluate_weighted_term(
    objective: ObjectiveFunctor[ParametersT],
    weight: float,
    exception_handler: ExceptionHandler,
    parameters: ParametersT,
    idx: int,
    ctx: EvaluateContext,
) -> float | None:
    """
    Evaluate one weighted term without retaining its combined objective.

    Keeping this operation independent of the combined objective lets process
    executors serialize term work without also serializing the execution
    policy (and, potentially, the executor itself).
    """
    try:
        return objective(parameters, ctx) * weight
    except Exception as e:
        return exception_handler(e, ctx, idx)


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


class SerialSchedule(PreparedScheduleBase[ParametersT], Generic[ParametersT]):
    def __init__(self, cob: CombinedObjectiveFunction[ParametersT]) -> None:
        """Initialize the serial schedule."""
        super().__init__()
        self.cob = cob

    def evaluate_terms(
        self,
        parameters: ParametersT,
        ctx: EvaluateContext,
        /,
    ) -> list[float | None]:
        """Evaluate all terms in index order."""

        with ctx.child_contexts(
            n_children=self.cob.n_terms(),
            configurator=self.cob.child_context_configurator,
        ) as child_ctxs:
            terms: list[float | None] = []

            for idx, ctx_term in enumerate(child_ctxs):
                terms.append(
                    evaluate_weighted_term(
                        self.cob.objective_functions[idx],
                        self.cob.weights[idx],
                        self.cob.exception_handler,
                        parameters,
                        idx,
                        ctx_term,
                    )
                )

            return terms


class SerialScheduler(Scheduler[ParametersT], Generic[ParametersT]):
    def prepare(
        self,
        objective: CombinedObjectiveFunction[ParametersT],
        /,
        *,
        profile: Mapping[tuple[int, ...], float] | None = None,  # noqa: ARG002
    ) -> SerialSchedule[ParametersT]:
        for term in objective.objective_functions:
            if isinstance(term, CombinedObjectiveFunction):
                term.prepare()

        return SerialSchedule(objective)


class CombinedObjectiveFunction(ObjectiveFunctor[ParametersT], Generic[ParametersT]):
    def __init__(
        self,
        objective_functions: Sequence[ObjectiveLike[ParametersT]],
        weights: Sequence[float] | None = None,
        child_context_configurator: ChildContextConfigurator | None = None,
        reduction: Reducer | Aggregator = sum_reducer,
        exception_handler: ExceptionHandler = raising_exception_handler,
        scheduler: Scheduler[ParametersT] | None = None,
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
                values to a single scalar loss. Can be either a simple reducer,
                or the more advanced Aggregator, which can make use of the full context
                and the quantities.
            exception_handler: Callable used to handle exceptions raised
                during term evaluation. It may return a replacement value or
                ``None`` to skip the term entirely.
            execution_policy: Strategy used to schedule term evaluations. If
                omitted, terms are evaluated serially in index order.

        Raises:
            AssertionError: If the number of weights does not match the
                number of objective functions, or if any weight is negative.

        """

        super().__init__()

        # Convert to list internally for mutability
        self.objective_functions = transform_generic_callables(objective_functions)

        self.child_context_configurator = child_context_configurator

        # TODO(MS): Replace signature-length inspection with an explicit,  # noqa: TD003
        # reliable way to distinguish reducers from aggregators.
        if len(inspect.signature(reduction).parameters) == 1:
            reduction = cast("Reducer", reduction)
            self.reduction: Aggregator = WrappedReducer(reduction)
        else:
            reduction = cast("Aggregator", reduction)
            self.reduction: Aggregator = reduction

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

        self._scheduler: Scheduler = (
            SerialScheduler() if scheduler is None else scheduler
        )
        self._schedule: PreparedSchedule | None = None

    def prepare(self, profile: SchedulingProfile | None = None):
        """Prepare the schedule."""
        if self._schedule is not None:
            self._schedule.close()

        self._schedule = self._scheduler.prepare(self, profile=profile)

    def set_scheduler(self, scheduler: Scheduler) -> None:
        """Set scheduler. Invalidates the current scheduler."""
        self._scheduler = scheduler
        self._schedule = None

    def _child_objectives(self) -> tuple[ObjectiveFunctor[ParametersT], ...]:
        """Return the objective terms evaluated in child contexts."""
        return tuple(self.objective_functions)

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

        # Mutation needs to invalidated the schedule
        self._schedule = None

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

    def _evaluate(
        self,
        parameters: ParametersT,
        ctx: EvaluateContext,
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
            ctx: Parent evaluation context.

        Returns:
            The reduced scalar loss computed from the evaluated terms.

        """

        if self._schedule is None:
            msg = "No `_schedule` found! Call the `.prepare` function, before invoking the objective."
            raise Exception(msg)

        terms = self._schedule.evaluate_terms(parameters, ctx)
        return self._reduce_terms(terms, ctx)
