"""
Public interfaces for preparing and executing component evaluations.

A Scheduler turns a component into a component-specific PreparedSchedule.
Prepared schedules are callable for synchronous singleton
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

import logging
from abc import ABC, abstractmethod
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
    OutputT_co,
)

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT")

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from chemfit.abstract_objective_function import Component


@dataclass(frozen=True)
class EvaluationRequest(Generic[InputT]):
    """
    Input for one evaluation in a prepared-schedule batch.

    Args:
        input: Value passed to the prepared component.
        ctx: Evaluation context that receives result state, metadata, and hook
            effects. A separate context should be supplied for every request
            that may execute concurrently.

    """

    input: InputT
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
        value: Computed output, or the exception representing a
            failure of this individual evaluation.

    """

    index: int
    value: OutputT_co | Exception

    @property
    def success(self) -> bool:
        """Whether the evaluation completed with a non-exception output."""

        return not isinstance(self.value, Exception)


@runtime_checkable
class PreparedSchedule(Protocol[InputT, OutputT_co]):
    """
    Reusable, component-specific execution plan produced by a scheduler.

    A prepared schedule owns state derived from a particular component, such
    as a compiled call graph, worker placement, or distributed object
    registrations. It may be reused for multiple evaluations until it is
    closed or the component structure changes. Call the schedule directly to
    evaluate one input synchronously, or use ``evaluate_many`` for
    a batch.

    Implementations are structural and need not inherit from
    PreparedScheduleBase as long as they provide this interface.
    """

    def evaluate_many(
        self,
        requests: Sequence[EvaluationRequest[InputT]],
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
        self,
        input: InputT,  # noqa: A002
        ctx: EvaluateContext,
        /,
    ) -> OutputT_co:
        """
        Evaluate one input synchronously with an explicit context.

        Calling the prepared schedule is the convenient singleton interface;
        this method provides the underlying operation when a context is always
        supplied.

        Args:
            input: Value passed to the prepared component.
            ctx: Context updated with this evaluation's state and metadata.

        Returns:
            Computed output.

        Raises:
            Exception: The exception stored for the individual evaluation if
                it fails.

        """

        ...

    def __call__(
        self,
        input: InputT,  # noqa: A002
        ctx: EvaluateContext | None = None,
    ) -> OutputT_co:
        """
        Evaluate one input synchronously.

        Args:
            input: Value passed to the prepared component.
            ctx: Optional context updated with this evaluation's state and metadata.

        Returns:
            Computed output.

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


class PreparedScheduleBase(ABC, Generic[InputT, OutputT_co]):
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
        requests: Sequence[EvaluationRequest[InputT]],
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
        input: InputT,  # noqa: A002
        ctx: EvaluateContext,
        /,
    ) -> OutputT_co:
        """Evaluate one request with an explicit context."""

        (result,) = self.evaluate_many(
            (EvaluationRequest[InputT](input=input, ctx=ctx),)
        )
        if isinstance(result.value, Exception):
            raise result.value
        return result.value

    def __call__(
        self,
        input: InputT,  # noqa: A002
        ctx: EvaluateContext | None = None,
    ) -> OutputT_co:
        """
        Evaluate one input synchronously.

        Args:
            input: Value passed to the prepared component.
            ctx: Optional context updated with this evaluation's state and metadata.

        Returns:
            Computed output.

        Raises:
            Exception: The exception stored for the individual evaluation if
                it fails.

        """

        if ctx is None:
            ctx = self._create_context()

        return self.evaluate(input, ctx)

    def _create_context(self) -> EvaluateContext:
        """Create the default context for a singleton evaluation."""
        return EvaluateContext()

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
            self._report_cleanup_failure(exception, cleanup_exception)
        finally:
            self._closed = True

    @staticmethod
    def _report_cleanup_failure(
        exception: BaseException,
        cleanup_exception: BaseException,
    ) -> None:
        """Record a cleanup failure without replacing the primary exception."""
        # Cleanup is secondary to the failure that triggered it. Raising the
        # cleanup error here would hide the scientifically relevant traceback.
        if hasattr(exception, "add_note"):  # add_note was only introduced in 3.11
            exception.add_note(f"Schedule cleanup failed: {cleanup_exception!r}")
        logger.warning(
            "Schedule cleanup failed while handling %s: %r",
            type(exception).__name__,
            cleanup_exception,
            exc_info=(
                type(cleanup_exception),
                cleanup_exception,
                cleanup_exception.__traceback__,
            ),
        )

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

        try:
            self.close()
        except BaseException as cleanup_exception:
            self._closed = True
            if exc_value is None:
                raise

            # Context-manager cleanup must not mask an exception raised by the
            # body; retain that exception and report the close failure on it.
            self._report_cleanup_failure(exc_value, cleanup_exception)


PreparedScheduleT_co = TypeVar(
    "PreparedScheduleT_co",
    bound=PreparedSchedule[Any, Any],
    covariant=True,
)


@runtime_checkable
class Scheduler(Protocol[PreparedScheduleT_co]):
    """
    Factory for prepared component execution plans.

    A scheduler describes reusable backend configuration or scheduling policy.
    Calling prepare binds that configuration to one component and
    returns a PreparedSchedule containing component-specific state.

    The protocol is structural. Implementations need not inherit from a
    ChemFit base class, and backends may differ substantially in how they
    execute requests.
    """

    def prepare(
        self,
        component: Component[Any, Any],
        /,
    ) -> PreparedScheduleT_co:
        """
        Prepare a component-specific plan for repeated evaluations.

        Args:
            component: Root component to prepare. For a structural component,
                preparation covers its complete nested component graph.

        Returns:
            Component-specific execution plan reusable for repeated inputs.

        Notes:
            Preparation is the appropriate place for work that depends on the
            component but should not occur in the hot evaluation path.
            Examples include tree traversal, rank assignment, scattering
            immutable data, and installing persistent worker-side component
            state.

            The returned schedule is valid only for the component structure
            that was prepared. Structural mutation of the component must
            invalidate and close the schedule before further evaluation.

            Implementations should avoid mutating the scientific definition of
            the supplied component. Backend-specific compiled representations
            belong in the returned schedule.

        """
        ...


class DirectSchedule(
    PreparedScheduleBase[InputT, OutputT_co], Generic[InputT, OutputT_co]
):
    def __init__(self, ob: Component[InputT, OutputT_co]) -> None:
        """Initialize a schedule that invokes its component directly."""
        super().__init__()
        self.ob = ob

    def _create_context(self) -> EvaluateContext:
        """Delegate default context creation to the prepared component."""
        return self.ob._create_context()  # noqa: SLF001

    def evaluate_many(
        self,
        requests: Sequence[EvaluationRequest[InputT]],
    ) -> Iterator[EvaluationResult[OutputT_co]]:
        if self.closed:
            msg = "Prepared schedule is closed."
            raise RuntimeError(msg)

        try:
            for idx, req in enumerate(requests):
                try:
                    value = self.ob(req.input, req.ctx)
                except Exception as e:  # noqa: PERF203
                    yield EvaluationResult[OutputT_co](index=idx, value=e)
                else:
                    yield EvaluationResult[OutputT_co](index=idx, value=value)

        except BaseException as e:
            self._close_after_catastrophic_failure(e)
            raise


class DirectScheduler(
    Scheduler[DirectSchedule[InputT, OutputT_co]],
    Generic[InputT, OutputT_co],
):
    """Prepare schedules that invoke components without graph compilation."""

    def prepare(
        self,
        component: Component[InputT, OutputT_co],
        /,
    ) -> DirectSchedule[InputT, OutputT_co]:
        return DirectSchedule(component)
