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
    parameters: ParametersT_co
    ctx: EvaluateContext


@dataclass(frozen=True)
class EvaluationResult:
    index: int
    value: float


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
    def evaluate_many(
        self,
        requests: Sequence[EvaluationRequest[ParametersT_contra]],
        /,
    ) -> Iterator[EvaluationResult]: ...

    def evaluate(
        self, parameters: ParametersT_contra, ctx: EvaluateContext, /
    ) -> float: ...

    def close(self) -> None: ...

    @property
    def closed(self) -> bool: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None: ...


class PreparedScheduleBase(ABC, Generic[ParametersT_contra]):
    """
    Convenience base class for prepared schedules.

    The structural :class:`PreparedSchedule` protocol is the public contract.
    Subclassing this class is optional.

    This base class provides singleton evaluation, closed-state tracking,
    idempotent closing, and context-manager support.
    """

    def __init__(self) -> None:
        """Placeholder."""

        self._closed = False

    @abstractmethod
    def evaluate_many(
        self,
        requests: Sequence[EvaluationRequest[ParametersT_contra]],
        /,
    ) -> Iterator[EvaluationResult]:
        raise NotImplementedError

    def evaluate(
        self,
        parameters: ParametersT_contra,
        ctx: EvaluateContext,
        /,
    ) -> float:
        (result,) = self.evaluate_many(
            (EvaluationRequest(parameters=parameters, ctx=ctx),)
        )
        return result.value

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> Self:
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

    ``Scheduler`` is intentionally a small protocol.  The only mandatory
    operation is :meth:`prepare`.

    The scheduler represents an execution facility or scheduling policy that
    may be reused across several combined objectives.  Objective-specific state
    belongs in the returned :class:`PreparedSchedule`.

    This distinction allows schedulers to have radically different internal
    behavior without forcing a common executor-like model:

    * a serial scheduler can return a trivial schedule;
    * an executor scheduler can prepare normalized term calls around a shared
      executor;
    * an MPI scheduler can perform static placement and distribute objective
      state once;
    * a Dask scheduler can scatter static state or create task metadata.

    Implementations do not need to inherit from a ChemFit base class.
    """

    def prepare(
        self,
        objective: CombinedObjectiveFunction[ParametersT],
        /,
        *,
        profile: SchedulingProfile | None = None,
    ) -> PreparedScheduleT_co:
        """
        Prepare repeated execution of ``objective``.

        Scheduler.prepare(root) prepares the complete CombinedObjectiveFunction
        call tree rooted at root.
        The scheduler is authoritative for that preparation.
        Scheduler configurations attached to nested combined objectives are ignored;
        they apply only when those objectives are independently prepared as roots.
        Implementations may install one or more PreparedSchedule objects throughout the tree, but all
        such schedules belong to the same root preparation and scheduling backend


        Parameters
        ----------
        objective
            Combined objective whose term evaluations will be scheduled.
            Implementations may inspect nested combined objectives, weights,
            exception handling, resource annotations, and other static
            structure needed to construct a plan.
        profile
            Optional measured or user-supplied cost profile keyed by nested
            objective path.  Implementations may ignore the profile if they do
            not perform cost-aware placement.

        Returns
        -------
        PreparedSchedule
            Objective-specific execution plan reusable for repeated parameter
            evaluations.

        Notes
        -----
        ``prepare()`` is the appropriate place for work that depends on the
        objective but should not occur in the hot evaluation path.  Examples
        include tree traversal, static load balancing, rank assignment,
        scattering immutable data, installing persistent worker-side objective
        state, and validating resource requirements.

        The returned schedule is valid only for the objective structure that
        was prepared.  Structural mutation of the objective must invalidate
        and close the schedule before further evaluation.

        Implementations should avoid mutating the scientific definition of the
        supplied objective.  Backend-specific compiled representations belong
        in the returned schedule.

        """
        ...
