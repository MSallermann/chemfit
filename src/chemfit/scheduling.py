"""
Scheduling interfaces for combined-objective evaluation.

This module defines the extension contract for scheduling the independent terms
of a :class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`.
It intentionally contains no concrete scheduler implementation.

The design separates two concepts:

``Scheduler``
    Describes an execution facility and prepares an objective for repeated
    evaluation.

``PreparedSchedule``
    Represents the scheduler-specific execution plan for one particular
    combined objective.  A prepared schedule may hold static placement
    information, persistent remote state, cached task metadata, or nothing at
    all.

This separation is important for ChemFit's main execution modes.  A serial
scheduler may prepare a trivial plan.  An executor-backed scheduler may retain
references to an executor and normalized term calls.  An MPI scheduler may
distribute static objective state once during ``prepare()`` and thereafter
communicate mainly parameter mappings and result state.  A Dask-backed
scheduler may use ``prepare()`` to scatter immutable state or create scheduling
metadata.

The public contract is deliberately structural: implementations do not have to
inherit from a ChemFit base class.  ``PreparedScheduleBase`` is provided only as
a convenience for implementations that want a default no-op ``close()`` and
context-manager support.

Lifecycle
---------

A combined objective is expected to use a scheduler approximately as follows::

    schedule = scheduler.prepare(combined_objective)

    try:
        value_1 = combined_objective(parameters_1, ctx_1)
        value_2 = combined_objective(parameters_2, ctx_2)
        ...
    finally:
        schedule.close()

In practice the :class:`CombinedObjectiveFunction` should own and cache the
prepared schedule.  Its evaluation path is conceptually::

    terms = prepared_schedule.evaluate_terms(parameters, ctx)
    terms = combined_objective.filter_terms(terms, ctx)
    return combined_objective.apply_reduction(terms, ctx)

A prepared schedule is therefore responsible for *term evaluation*, while the
combined objective remains responsible for filtering skipped terms and applying
its reducer or aggregator.

The combined objective should invalidate and close its prepared schedule when
its structural definition changes, for example when terms or weights are
added.  A schedule prepared for one objective structure must not silently be
reused after that structure has changed.

Term-evaluation contract
------------------------

``PreparedSchedule.evaluate_terms()`` receives the parent
:class:`~chemfit.abstract_objective_function.EvaluateContext` and must return
one value for every *immediate* term of the prepared combined objective, in term
index order.  Returned values are the weighted term values; ``None`` denotes a
term skipped by the objective's exception handler.

The prepared schedule owns the mechanics of creating/evaluating child contexts
because execution backends may need different context-transport strategies.
Nevertheless, the observable semantics must match ordinary ChemFit evaluation:

* each immediate term is evaluated in its own child context;
* the combined objective's ``child_context_configurator`` is honored;
* objective evaluation goes through the normal ``ObjectiveFunctor`` lifecycle,
  including pre- and post-evaluation hooks;
* worker-side context result state is propagated back to the caller-visible
  context;
* on return, the parent context contains the child metadata required by
  ``CombinedObjectiveFunction.apply_reduction()``;
* result ordering is the original term ordering, independent of physical
  execution order.

A scheduler may execute terms serially, concurrently, remotely, or according to
a precomputed static placement.  Those choices must not change the scientific
semantics of the combined objective.

Nested combined objectives
--------------------------

Nested :class:`CombinedObjectiveFunction` instances are ordinary objective
terms and must remain semantically nested.  In particular, their own weights,
reducers/aggregators, exception handling, hooks, and context hierarchy must not
be flattened away merely because physical execution is flattened.

A scheduler implementation has several valid choices:

1. Treat nested combined objectives as opaque terms and schedule only the
   immediate children of the prepared objective.
2. During ``prepare()``, inspect nested combined objectives and build a static
   execution plan containing subtrees or leaves.
3. Use a hybrid plan in which some nested subtrees are kept local while others
   are distributed.

An advanced scheduler that compiles across nested combined objectives is
responsible for preserving the same observable evaluation lifecycle and context
tree that normal nested objective calls would produce.

For MPI, recursive use of the same communicator from nested combined objectives
should generally be avoided.  A typical MPI scheduler should instead perform a
single preparation pass for the objective tree, assign persistent work to
ranks, and then reuse that placement for repeated parameter evaluations.

Profiling and static load balancing
-----------------------------------

``Scheduler.prepare()`` accepts an optional :class:`SchedulingProfile`.
Profiles are intentionally small and backend-independent.  They associate
stable objective-tree paths with relative or absolute cost estimates.

ChemFit evaluation hooks can be used to measure objective runtimes during one
or more real evaluations.  A caller may then close the initial schedule and
prepare a new one using the measured profile.  This supports static
load-balancing without requiring a dynamic work-stealing scheduler::

    schedule = scheduler.prepare(objective)
    # Run one or more profiling evaluations.
    schedule.close()

    profile = build_profile_from_recorded_hook_data(...)
    schedule = scheduler.prepare(objective, profile=profile)

The profile values need not be wall-clock seconds.  Only relative cost is
required by many static partitioning strategies.  If several measurements are
available, implementations or profile-building utilities may use a robust
summary such as the median.

``ObjectivePath`` identifies an objective by its path through the nested
combined-objective tree.  For example, ``(2, 1, 0)`` means child 2 of the root,
then child 1, then child 0.  A profile is valid only for the objective structure
from which those paths were derived.

Resources
---------

Resource annotations are intentionally *not* part of the mandatory scheduler
method signatures.  Simple schedulers should be free to ignore resources.
Resource requirements belong to objectives/computers (or to metadata derived
from them), and resource-aware schedulers may inspect that information during
``prepare()``.

This keeps serial and ordinary-executor schedulers simple while allowing future
MPI or Dask schedulers to account for CPUs, memory, GPUs, MPI ranks, licenses,
node features, or other named resources.

Optional submission capability
------------------------------

Some schedulers naturally expose an executor-like ``submit()`` operation.
Others do not.  In particular, an efficient MPI scheduler may rely on
persistent worker-side objective state rather than serializing arbitrary
callables for each submission.

For that reason ``submit()`` is not part of :class:`Scheduler`.
Implementations that support arbitrary task submission may additionally satisfy
the independent :class:`TaskSubmitter` protocol.  ChemFit code that requires
that capability should type against ``TaskSubmitter`` explicitly instead of
assuming every scheduler supports it.

Ownership
---------

A :class:`CombinedObjectiveFunction` should own its prepared
:class:`PreparedSchedule` and close it when the plan is invalidated or the
objective is explicitly closed.

The scheduler itself represents the execution facility/configuration and may be
shared by multiple objectives.  Closing one prepared schedule must therefore
not implicitly close an externally owned executor, Dask client, MPI
communicator, or similar scheduler resource unless the concrete scheduler
explicitly documents such ownership.

Implementing a scheduler
------------------------

A minimal scheduler requires two objects::

    class MyScheduler:
        def prepare(
            self,
            objective: CombinedObjectiveFunction[MyParameters],
            /,
            *,
            profile: SchedulingProfile | None = None,
        ) -> PreparedSchedule[MyParameters]:
            ...

    class MyPreparedSchedule(PreparedScheduleBase[MyParameters]):
        def evaluate_terms(
            self,
            parameters: MyParameters,
            ctx: EvaluateContext,
            /,
        ) -> list[float | None]:
            ...

``prepare()`` is the correct place for work that depends on the objective but
should not be repeated for every parameter evaluation, such as:

* traversing the objective tree;
* computing static placement;
* applying a measured scheduling profile;
* distributing static objective state;
* scattering immutable data;
* establishing worker-side registries;
* validating resource requirements.

``evaluate_terms()`` should be the hot path.  It should reuse state produced by
``prepare()`` and avoid retransmitting static objective data where the backend
allows that optimization.

Using a scheduler
-----------------
o
Once ``CombinedObjectiveFunction`` is integrated with this module, user code
should look like::

    scheduler = MyScheduler(...)

    objective = CombinedObjectiveFunction(
        [term_a, term_b, term_c],
        scheduler=scheduler,
    )

    loss = objective(parameters)

The eventual high-level API may expose the same concept through ``combine()``::

    objective = chemfit.combine(
        term_a,
        term_b,
        term_c,
        scheduler=scheduler,
    )

The scheduler should be selected according to execution needs; reducers,
aggregators, weights, exception handling, and scientific objective structure
remain properties of ``CombinedObjectiveFunction``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from chemfit.abstract_objective_function import EvaluateContext, FutureLike

if TYPE_CHECKING:
    from chemfit.combined_objective_function import CombinedObjectiveFunction


ParametersT_contra = TypeVar(
    "ParametersT_contra", bound=Mapping[str, object], contravariant=True
)
ResultT_co = TypeVar("ResultT_co", covariant=True)

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
    Prepared execution plan for one combined objective.

    A prepared schedule is created by :meth:`Scheduler.prepare` and is expected
    to be reused for many evaluations of the same objective structure.

    Implementations may contain persistent state such as worker placement,
    remote object registries, scattered data handles, executor configuration,
    or cached traversal information.  They must not store per-evaluation
    scientific results on the combined objective itself; per-evaluation state
    belongs in :class:`EvaluateContext`.

    A prepared schedule becomes invalid when the structural definition of the
    combined objective changes.  The owner is responsible for closing the old
    plan and preparing a replacement.

    Notes
    -----
    The protocol is structural.  Implementations do not need to inherit from
    :class:`PreparedScheduleBase`.

    """

    def evaluate_terms(
        self,
        parameters: ParametersT_contra,
        ctx: EvaluateContext,
        /,
    ) -> list[float | None]:
        """
        Evaluate the immediate weighted terms of the prepared objective.

        Parameters
        ----------
        parameters
            Parameter mapping for this evaluation.
        ctx
            Parent evaluation context belonging to the prepared
            ``CombinedObjectiveFunction``.

        Returns
        -------
        list[float | None]
            One entry per immediate objective term, in original term-index
            order.  Values are the weighted term results.  ``None`` denotes a
            term skipped by the combined objective's exception handler.

        Contract
        --------
        The implementation must preserve ChemFit's normal term-evaluation
        semantics even when physical execution is reordered or remote:

        * create/use one child evaluation context per immediate term;
        * honor the combined objective's child-context configurator;
        * invoke objective terms through their normal evaluation lifecycle so
          evaluation hooks run correctly;
        * apply the combined objective's weights and exception handling;
        * propagate worker-side context result state back to caller-visible
          contexts;
        * ensure parent child metadata is materialized before returning;
        * return results in original term order.

        Filtering skipped terms and applying the reducer/aggregator are *not*
        responsibilities of the schedule.  Those remain responsibilities of
        ``CombinedObjectiveFunction``.

        Nested combined objectives may be treated as opaque terms or compiled
        into a deeper execution plan.  If an implementation compiles across
        nesting boundaries, it must preserve nested reductions, hooks,
        exception behavior, and context hierarchy exactly as ordinary nested
        evaluation would.

        """
        ...

    def close(self) -> None:
        """
        Release state owned specifically by this prepared schedule.

        ``close()`` must be safe to call after the last evaluation and should
        release per-objective remote registrations, cached/scattered handles,
        temporary worker state, or similar resources.

        Closing a schedule must not automatically close externally owned
        scheduler infrastructure (for example a user-provided executor, Dask
        client, or MPI communicator) unless the concrete implementation
        explicitly documents that ownership.
        """
        ...


class PreparedScheduleBase(ABC, Generic[ParametersT_contra]):
    """
    Optional convenience base class for prepared schedules.

    The structural :class:`PreparedSchedule` protocol is the actual contract.
    Subclassing this class is not required.

    This base class only provides a default no-op :meth:`close` and context
    manager support.  It deliberately does not prescribe execution strategy,
    state representation, resource handling, or task submission semantics.
    """

    @abstractmethod
    def evaluate_terms(
        self,
        parameters: ParametersT_contra,
        ctx: EvaluateContext,
        /,
    ) -> list[float | None]:
        """Evaluate terms according to :class:`PreparedSchedule`."""
        raise NotImplementedError

    def close(self) -> None:
        """Release schedule-local state.  Default implementation is a no-op."""

    def __enter__(self) -> PreparedScheduleBase[ParametersT_contra]:
        """Return this prepared schedule."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        """Close this prepared schedule when leaving a context-manager scope."""
        self.close()


@runtime_checkable
class Scheduler(Protocol[ParametersT_contra]):
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
        objective: CombinedObjectiveFunction[ParametersT_contra],
        /,
        *,
        profile: SchedulingProfile | None = None,
    ) -> PreparedSchedule[ParametersT_contra]:
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


@runtime_checkable
class TaskSubmitter(Protocol):
    """
    Optional capability for scheduler-like objects that can submit arbitrary work.

    ``TaskSubmitter`` is deliberately independent of :class:`Scheduler`.
    Arbitrary callable submission is natural for executors and Dask, but it is
    not required for an efficient scheduler.  In particular, an MPI scheduler
    may instead rely on persistent worker-side objective state and communicate
    only changing parameters during repeated evaluation.

    ChemFit components that genuinely require arbitrary task submission should
    request this protocol explicitly rather than assuming every scheduler
    supports ``submit()``.
    """

    def submit(
        self,
        fn: Callable[..., ResultT_co],
        /,
        *args: Any,
        resources: ResourceRequest | None = None,
        **kwargs: Any,
    ) -> FutureLike[ResultT_co]:
        """
        Submit an arbitrary callable for execution.

        Parameters
        ----------
        fn
            Callable to execute.
        *args
            Positional arguments forwarded to ``fn``.
        resources
            Optional named resource requirements.  The meaning of resource
            names is backend-specific.  An implementation may reject
            unsupported resource requests.
        **kwargs
            Keyword arguments forwarded to ``fn``.

        Returns
        -------
        FutureLike
            Future representing the submitted computation.

        Notes
        -----
        This method is an optional capability and is not used by the mandatory
        ``Scheduler.prepare()`` / ``PreparedSchedule.evaluate_terms()`` contract.

        """
        ...
