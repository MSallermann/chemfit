"""
Public interfaces for preparing and executing objective evaluations.

A Scheduler turns an objective functor into an objective-specific
PreparedSchedule. Prepared schedules are callable for synchronous singleton
evaluation. They also accept batches of EvaluationRequest objects through
``evaluate_many`` and yield EvaluationResult objects as individual evaluations
finish. Results carry their input index, so a backend may emit them in
completion order instead of request order.

The interfaces describe capabilities rather than a particular execution
model. Implementations may run locally, use an executor, or coordinate
persistent distributed workers while exposing the same lifecycle and result
protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from typing_extensions import Self

from chemfit.abstract_objective_function import (
    EvaluateContext,
    InputT_contra,
    OutputT_co,
)

if TYPE_CHECKING:
    from chemfit.abstract_objective_function import ObjectiveComponent


@dataclass(frozen=True)
class EvaluationRequest(Generic[InputT_contra]):
    """
    Input for one evaluation in a prepared-schedule batch.

    Args:
        parameters: Parameter mapping passed to the prepared objective.
        ctx: Evaluation context that receives result state, metadata, and hook
            effects. A separate context should be supplied for every request
            that may execute concurrently.

    """

    parameters: InputT_contra
    ctx: EvaluateContext


@dataclass(frozen=True)
class EvaluationResult(Generic[OutputT_co]):
    """
    Outcome of one request submitted through a prepared schedule.

    Args:
        index: Zero-based position of the corresponding request in the sequence
            passed to PreparedSchedule.evaluate_many. Consumers should use
            this index to restore request order because schedules may yield
            results in completion order.
        value: Computed objective value, or the exception representing a
            failure of this individual evaluation.

    """

    index: int
    value: OutputT_co | Exception

    @property
    def success(self) -> bool:
        """Whether the evaluation completed with a numerical value."""

        return not isinstance(self.value, Exception)


@runtime_checkable
class PreparedSchedule(Protocol[InputT_contra, OutputT_co]):
    """
    Reusable, objective-specific execution plan produced by a scheduler.

    A prepared schedule owns state derived from a particular objective, such
    as a compiled call tree, worker placement, or distributed object
    registrations. It may be reused for multiple evaluations until it is
    closed or the objective structure changes. Call the schedule directly to
    evaluate one parameter mapping synchronously, or use ``evaluate_many`` for
    a batch.

    Implementations are structural and need not inherit from
    PreparedScheduleBase as long as they provide this interface.
    """

    def evaluate_many(
        self,
        requests: Sequence[EvaluationRequest[InputT_contra]],
        /,
    ) -> Iterator[EvaluationResult]:
        """
        Evaluate a batch and yield each request's outcome.

        Args:
            requests: Evaluations to execute. Each request must have its own context
                when requests can overlap in time.

        Yields:
            One result per request. Results may be yielded in completion
            order; EvaluationResult.index identifies the corresponding
            position in requests.

        Notes:
            An exception stored in a result represents failure of that
            individual evaluation. Exceptions raised directly by the iterator
            indicate that the batch itself could not continue, for example
            because of a backend or communication failure.

        """

        ...

    def evaluate(
        self, parameters: InputT_contra, ctx: EvaluateContext, /
    ) -> OutputT_co:
        """
        Evaluate one parameter mapping synchronously with an explicit context.

        Calling the prepared schedule is the convenient singleton interface;
        this method provides the underlying operation when a context is always
        supplied.

        Args:
            parameters: Parameter mapping passed to the prepared objective.
            ctx: Context updated with this evaluation's state and metadata.

        Returns:
            Computed objective value.

        Raises:
            Exception: The exception stored for the individual evaluation if
                it fails.

        """

        ...

    def __call__(
        self, parameters: InputT_contra, ctx: EvaluateContext | None = None
    ) -> OutputT_co:
        """
        Evaluate one parameter mapping synchronously.

        Args:
            parameters: Parameter mapping passed to the prepared objective.
            ctx: Optional context updated with this evaluation's state and metadata.

        Returns:
            Computed objective value.

        Raises:
            Exception: The exception stored for the individual evaluation if
                it fails.

        """
        ...

    def close(self) -> None:
        """Release resources owned by this prepared schedule."""

        ...

    @property
    def closed(self) -> bool:
        """Whether the schedule has been closed and can no longer be used."""

        ...

    def __enter__(self) -> Self:
        """Enter the context and return the concrete prepared schedule."""

        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        """Close the prepared schedule when leaving its context."""

        ...


class PreparedScheduleBase(ABC, Generic[InputT_contra, OutputT_co]):
    """
    Convenience base class for prepared schedules.

    The structural :class:`PreparedSchedule` protocol is the public contract.
    Subclassing this class is optional.

    This base class provides callable singleton evaluation, closed-state
    tracking, idempotent closing, and context-manager support.
    """

    def __init__(self) -> None:
        """Initialize an open prepared schedule."""

        self._closed = False

    @abstractmethod
    def evaluate_many(
        self,
        requests: Sequence[EvaluationRequest[InputT_contra]],
        /,
    ) -> Iterator[EvaluationResult]:
        """
        Evaluate a batch and yield one indexed outcome per request.

        Subclasses must implement backend-specific batch execution. They may
        yield results in any order as long as every result contains the correct
        request index.
        """

        raise NotImplementedError

    def evaluate(
        self,
        parameters: InputT_contra,
        ctx: EvaluateContext,
        /,
    ) -> OutputT_co:
        """Evaluate one request with an explicit context."""

        (result,) = self.evaluate_many(
            (EvaluationRequest[InputT_contra](parameters=parameters, ctx=ctx),)
        )
        if isinstance(result.value, Exception):
            raise result.value
        return result.value

    def __call__(
        self, parameters: InputT_contra, ctx: EvaluateContext | None = None
    ) -> OutputT_co:
        """
        Evaluate one parameter mapping synchronously.

        Args:
            parameters: Parameter mapping passed to the prepared objective.
            ctx: Optional context updated with this evaluation's state and metadata.

        Returns:
            Computed objective value.

        Raises:
            Exception: The exception stored for the individual evaluation if
                it fails.

        """

        if ctx is None:
            ctx = EvaluateContext()

        return self.evaluate(parameters, ctx)

    def close(self) -> None:
        """
        Mark the schedule as closed.

        Subclasses that own backend resources should release them before or
        after calling this implementation. Repeated calls are safe.
        """

        self._closed = True

    @property
    def closed(self) -> bool:
        """Whether close has been called."""

        return self._closed

    def _close_after_catastrophic_failure(self, exception: BaseException) -> None:
        """Poison this schedule without replacing the original failure."""

        try:
            self.close()
        except BaseException as cleanup_exception:
            if hasattr(exception, "add_note"):  # add_note was only introduced in 3.11
                exception.add_note(f"Schedule cleanup failed: {cleanup_exception!r}")
        finally:
            self._closed = True

    def __enter__(self) -> Self:
        """Return this schedule, rejecting attempts to reuse a closed one."""

        if self._closed:
            msg = "Prepared schedule is closed."
            raise RuntimeError(msg)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        """Close the schedule when its context-manager scope exits."""

        self.close()


PreparedScheduleT_co = TypeVar(
    "PreparedScheduleT_co",
    bound=PreparedSchedule[Any, Any],
    covariant=True,
)


@runtime_checkable
class Scheduler(Protocol[PreparedScheduleT_co]):
    """
    Factory for prepared objective execution plans.

    A scheduler describes reusable backend configuration or scheduling policy.
    Calling prepare binds that configuration to one objective and
    returns a PreparedSchedule containing objective-specific state.

    The protocol is structural. Implementations need not inherit from a
    ChemFit base class, and backends may differ substantially in how they
    execute requests.
    """

    def prepare(
        self,
        objective: ObjectiveComponent[Any, Any],
        /,
    ) -> PreparedScheduleT_co:
        """
        Prepare an objective-specific plan for repeated evaluations.

        Args:
            objective: Root objective to prepare. For a combined objective,
                preparation covers its complete nested call tree.

        Returns:
            Objective-specific execution plan reusable for repeated parameter
            evaluations.

        Notes:
            Preparation is the appropriate place for work that depends on the
            objective but should not occur in the hot evaluation path.
            Examples include tree traversal, rank assignment, scattering
            immutable data, and installing persistent worker-side objective
            state.

            The returned schedule is valid only for the objective structure
            that was prepared. Structural mutation of the objective must
            invalidate and close the schedule before further evaluation.

            Implementations should avoid mutating the scientific definition of
            the supplied objective. Backend-specific compiled representations
            belong in the returned schedule.

        """
        ...


class SerialSchedule(
    PreparedScheduleBase[InputT_contra, OutputT_co], Generic[InputT_contra, OutputT_co]
):
    def __init__(self, ob: ObjectiveComponent[InputT_contra, OutputT_co]) -> None:
        """Initialize a reference serial schedule."""
        super().__init__()
        self.ob = ob

    def evaluate_many(
        self,
        requests: Sequence[EvaluationRequest[InputT_contra]],
    ) -> Iterator[EvaluationResult[OutputT_co]]:
        if self.closed:
            msg = "Prepared schedule is closed."
            raise RuntimeError(msg)

        try:
            for idx, req in enumerate(requests):
                try:
                    value = self.ob(req.parameters, req.ctx)
                except Exception as e:  # noqa: PERF203
                    yield EvaluationResult[OutputT_co](index=idx, value=e)
                else:
                    yield EvaluationResult[OutputT_co](index=idx, value=value)

        except BaseException as e:
            self._close_after_catastrophic_failure(e)
            raise


class SerialScheduler(
    Scheduler[SerialSchedule[InputT_contra, OutputT_co]],
    Generic[InputT_contra, OutputT_co],
):
    def prepare(
        self,
        objective: ObjectiveComponent[InputT_contra, OutputT_co],
        /,
    ) -> SerialSchedule[InputT_contra, OutputT_co]:
        return SerialSchedule(objective)
