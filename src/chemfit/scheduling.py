"""
Public interfaces for preparing and executing objective evaluations.

A Scheduler turns a combined objective into an objective-specific
PreparedSchedule. Prepared schedules accept batches of EvaluationRequest
objects and yield EvaluationResult objects as individual evaluations finish.
Results carry their input index, so a backend may emit them in completion
order instead of request order.

The interfaces describe capabilities rather than a particular execution
model. Implementations may run locally, use an executor, or coordinate
persistent distributed workers while exposing the same lifecycle and result
protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from chemfit.abstract_objective_function import EvaluateContext

if TYPE_CHECKING:
    from chemfit.combined_objective_function import CombinedObjectiveFunction

ParametersT_co = TypeVar("ParametersT_co", bound=Mapping[str, object], covariant=True)
ParametersT_contra = TypeVar(
    "ParametersT_contra", bound=Mapping[str, object], contravariant=True
)
ParametersT = TypeVar("ParametersT", bound=Mapping[str, object])
ResultT_co = TypeVar("ResultT_co", covariant=True)


@dataclass(frozen=True)
class EvaluationRequest(Generic[ParametersT_co]):
    """
    Input for one evaluation in a prepared-schedule batch.

    Args:
        parameters: Parameter mapping passed to the prepared objective.
        ctx: Evaluation context that receives result state, metadata, and hook
            effects. A separate context should be supplied for every request
            that may execute concurrently.

    """

    parameters: ParametersT_co
    ctx: EvaluateContext


@dataclass(frozen=True)
class EvaluationResult:
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
    value: float | Exception

    @property
    def success(self) -> bool:
        """Whether the evaluation completed with a numerical value."""

        return not isinstance(self.value, Exception)


#: Stable path to an objective within a nested combined-objective tree.
#:
#: ``()`` identifies the root combined objective. ``(1,)`` identifies its
#: second immediate term. ``(2, 1, 0)`` identifies child 2, then child 1,
#: then child 0.
ObjectivePath: TypeAlias = tuple[int, ...]

#: Relative or absolute evaluation cost estimates keyed by objective-tree path.
#:
#: Values are deliberately unitless from the scheduler contract's point of
#: view.  They may be seconds, normalized costs, or another positive scalar
#: suitable for static placement.
SchedulingProfile: TypeAlias = Mapping[ObjectivePath, float]

#: Extensible named resource request.
#:
#: This alias is provided for optional scheduler capabilities and future
#: resource-aware integrations.  Core scheduling does not require schedulers to
#: interpret these values.  Example keys might include ``"cpus"``, ``"gpus"``,
#: ``"memory_gb"``, or backend-specific resources.
ResourceRequest: TypeAlias = Mapping[str, float]


@runtime_checkable
class PreparedSchedule(Protocol[ParametersT_contra]):
    """
    Reusable, objective-specific execution plan produced by a scheduler.

    A prepared schedule owns state derived from a particular combined
    objective, such as a compiled call tree, worker placement, or distributed
    object registrations. It may be reused for multiple evaluations until it
    is closed or the objective structure changes.

    Implementations are structural and need not inherit from
    PreparedScheduleBase as long as they provide this interface.
    """

    def evaluate_many(
        self,
        requests: Sequence[EvaluationRequest[ParametersT_contra]],
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
        self, parameters: ParametersT_contra, ctx: EvaluateContext, /
    ) -> float:
        """
        Evaluate one parameter mapping synchronously.

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


class PreparedScheduleBase(ABC, Generic[ParametersT_contra]):
    """
    Convenience base class for prepared schedules.

    The structural :class:`PreparedSchedule` protocol is the public contract.
    Subclassing this class is optional.

    This base class provides singleton evaluation, closed-state tracking,
    idempotent closing, and context-manager support.
    """

    def __init__(self) -> None:
        """Initialize an open prepared schedule."""

        self._closed = False

    @abstractmethod
    def evaluate_many(
        self,
        requests: Sequence[EvaluationRequest[ParametersT_contra]],
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
        parameters: ParametersT_contra,
        ctx: EvaluateContext,
        /,
    ) -> float:
        """Evaluate one request and raise its stored exception on failure."""

        (result,) = self.evaluate_many(
            (EvaluationRequest(parameters=parameters, ctx=ctx),)
        )
        if isinstance(result.value, Exception):
            raise result.value
        return result.value

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
    bound=PreparedSchedule[Any],
    covariant=True,
)


@runtime_checkable
class Scheduler(Protocol[PreparedScheduleT_co]):
    """
    Factory for prepared combined-objective execution plans.

    A scheduler describes reusable backend configuration or scheduling policy.
    Calling prepare binds that configuration to one combined objective and
    returns a PreparedSchedule containing objective-specific state.

    The protocol is structural. Implementations need not inherit from a
    ChemFit base class, and backends may differ substantially in how they
    execute requests.
    """

    def prepare(
        self,
        objective: CombinedObjectiveFunction[ParametersT],
        /,
        *,
        profile: SchedulingProfile | None = None,
    ) -> PreparedScheduleT_co:
        """
        Prepare an objective-specific plan for repeated evaluations.

        Args:
            objective: Root combined objective to prepare. Preparation covers
                its complete nested call tree. Scheduler settings attached to
                nested combined objectives do not apply while they are
                executed through this root plan.
            profile: Optional measured or user-supplied cost profile keyed by
                nested objective path. Implementations may ignore the profile
                if they do not perform cost-aware placement.

        Returns:
            Objective-specific execution plan reusable for repeated parameter
            evaluations.

        Notes:
            Preparation is the appropriate place for work that depends on the
            objective but should not occur in the hot evaluation path.
            Examples include tree traversal, static load balancing, rank
            assignment, scattering immutable data, installing persistent
            worker-side objective state, and validating resource requirements.

            The returned schedule is valid only for the objective structure
            that was prepared. Structural mutation of the objective must
            invalidate and close the schedule before further evaluation.

            Implementations should avoid mutating the scientific definition of
            the supplied objective. Backend-specific compiled representations
            belong in the returned schedule.

        """
        ...
