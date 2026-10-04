from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

from ase import Atoms

from chemfit.abstract_objective_function import EvaluateContext, ObjectiveFunctor
from chemfit.ase_objective_function import ASEComputer, PathAtomsFactory
from chemfit.combined_objective_function import (
    Aggregator,
    ChildContextConfigurator,
    CombinedObjectiveFunction,
    ExceptionHandler,
    Reducer,
    raising_exception_handler,
)
from chemfit.executor_scheduler import ExecutorTreeScheduler
from chemfit.external_computer import ExternalQuantityComputer
from chemfit.fitter import CallbackT, Fitter, FitterEvaluateContext
from chemfit.scheduling import (
    EvaluationRequest,
    Scheduler,
    SerialScheduler,
)
from chemfit.wrap_funcs import WrappedObjectiveFunctor

ParamsT = TypeVar("ParamsT", bound=Mapping[str, Any])
ObjectiveLike = Callable[[ParamsT], float] | ObjectiveFunctor[ParamsT]


def combine(
    *args: ObjectiveLike[ParamsT],
    weights: Sequence[float] | None = None,
    reduction: Reducer | None = None,
    aggregator: Aggregator | None = None,
    exception_handler: ExceptionHandler = raising_exception_handler,
    child_context_configurator: ChildContextConfigurator | None = None,
) -> CombinedObjectiveFunction[ParamsT]:
    return CombinedObjectiveFunction(
        objective_functions=args,
        weights=weights,
        reduction=reduction,
        aggregator=aggregator,
        exception_handler=exception_handler,
        child_context_configurator=child_context_configurator,
    )


def external_quantity(
    workdir: Path | str,
) -> ExternalQuantityComputer[Any, dict[str, Any]]:
    """Create an external quantity computer rooted at ``workdir``."""
    return ExternalQuantityComputer(base_working_directory=workdir)


AtomsSource = Atoms | str | Path | Callable[[], Atoms]


def ase_quantity(atoms: AtomsSource, *, index: int | None = None) -> ASEComputer:
    if not isinstance(atoms, (Path, str)) and index is not None:
        msg = "`index` is only valid for path-based atoms sources."
        raise ValueError(msg)

    if isinstance(atoms, Atoms):

        def atoms_factory() -> Atoms:
            return atoms.copy()

    elif isinstance(atoms, (Path, str)):
        atoms_factory = PathAtomsFactory(Path(atoms), index)

    elif callable(atoms):
        atoms_factory = atoms

    else:
        msg = "`atoms` must be an Atoms object, path, or callable returning Atoms."
        raise TypeError(msg)

    return ASEComputer(atoms_factory=atoms_factory)


def evaluate_many(
    objective: ObjectiveLike[ParamsT],
    parameters: Sequence[ParamsT],
    *,
    executor: Executor | None = None,
    scheduler: Scheduler | None = None,
) -> list[EvaluateContext]:
    if executor is None and scheduler is None:
        msg = "Specify either `executor` or `scheduler`"
        raise ValueError(msg)

    if executor is not None and scheduler is not None:
        msg = "Specify only one of `executor` or `scheduler`"
        raise ValueError(msg)

    if scheduler is None:
        assert executor is not None
        scheduler = ExecutorTreeScheduler(executor=executor)

    if not isinstance(objective, ObjectiveFunctor):
        ob = WrappedObjectiveFunctor(objective)
    else:
        ob = objective

    reqs = [EvaluationRequest(p, EvaluateContext()) for p in parameters]

    with scheduler.prepare(ob) as schedule:
        results = list(schedule.evaluate_many(reqs))

    # `schedule.evaluate_many` returns results in completion order
    # we sort results so they are in the same order as the requests
    results.sort(key=lambda res: res.index)

    for res in results:
        if isinstance(res.value, Exception):
            raise res.value

    return [req.ctx for req in reqs]


@dataclass(frozen=True)
class FitResult(Generic[ParamsT]):
    recommendation: ParamsT
    contexts: tuple[FitterEvaluateContext, ...]

    @property
    def best_context(self) -> FitterEvaluateContext:
        evaluated = [ctx for ctx in self.contexts if ctx.opt_loss is not None]

        if not evaluated:
            msg = "No successful evaluations were recorded."
            raise RuntimeError(msg)

        return min(
            evaluated,
            key=lambda ctx: cast("float", ctx.opt_loss),
        )

    @property
    def best_loss(self) -> float:
        loss = self.best_context.opt_loss
        assert loss is not None
        return loss

    @property
    def best_parameters(self) -> Mapping[str, Any]:
        parameters = self.best_context.opt_params
        assert parameters is not None
        return parameters


def fit(
    objective: ObjectiveLike[ParamsT],
    initial: ParamsT,
    *,
    budget: int,
    workers: int = 1,
    execution_workers: int | None = None,
    bounds: Mapping[str, Any] | None = None,
    optimizer: str = "NgIohTuned",
    executor: Executor | None = None,
    scheduler: Scheduler | None = None,
    callbacks: Sequence[tuple[CallbackT, int]] | None = None,
    parametrization: Mapping[str, object] | None = None,
) -> FitResult[ParamsT]:
    """
    Fit ``objective`` with Nevergrad.

    ``workers`` controls the number of candidates Nevergrad asks for and the
    maximum candidate batch size. ``execution_workers`` controls the maximum
    number of objective leaf tasks run concurrently by the built-in thread
    scheduler and defaults to ``workers``. When ``executor`` or ``scheduler``
    is supplied, that object controls execution concurrency and
    ``execution_workers`` must be omitted.
    """

    if budget < 1:
        msg = "`budget` must be at least 1."
        raise ValueError(msg)

    if workers < 1:
        msg = "`workers` must be at least 1."
        raise ValueError(msg)

    if execution_workers is not None and execution_workers < 1:
        msg = "`execution_workers` must be at least 1."
        raise ValueError(msg)

    if executor is not None and scheduler is not None:
        msg = "Specify only one of `executor` or `scheduler`"
        raise ValueError(msg)

    if execution_workers is not None and (
        executor is not None or scheduler is not None
    ):
        msg = "`execution_workers` cannot be combined with `executor` or `scheduler`."
        raise ValueError(msg)

    if scheduler is None and executor is not None:
        scheduler = ExecutorTreeScheduler(executor=executor)

    if scheduler is None:
        effective_execution_workers = (
            workers if execution_workers is None else execution_workers
        )
        if effective_execution_workers == 1:
            scheduler = SerialScheduler()
        else:
            scheduler = ExecutorTreeScheduler(
                executor_factory=lambda: ThreadPoolExecutor(
                    max_workers=effective_execution_workers
                )
            )

    fitter = Fitter[ParamsT](
        objective_function=objective,
        initial_params=initial,
        bounds=bounds,
        scheduler=scheduler,
    )

    if callbacks is not None:
        for func, nsteps in callbacks:
            fitter.register_callback(func, nsteps)

    recommendation = fitter.fit_nevergrad(
        budget=budget,
        optimizer_str=optimizer,
        num_workers=workers,
        parametrization=parametrization,
    )

    return FitResult(recommendation=recommendation, contexts=tuple(fitter.contexts))
