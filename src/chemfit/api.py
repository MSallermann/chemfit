from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar, cast, overload

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


@overload
def combine(
    args: Iterable[ObjectiveLike[ParamsT]],
    /,
    *,
    weights: Sequence[float] | None = None,
    reduction: Reducer | None = None,
    aggregator: Aggregator | None = None,
    exception_handler: ExceptionHandler = raising_exception_handler,
    child_context_configurator: ChildContextConfigurator | None = None,
) -> CombinedObjectiveFunction[ParamsT]: ...


@overload
def combine(
    *args: ObjectiveLike[ParamsT],
    weights: Sequence[float] | None = None,
    reduction: Reducer | None = None,
    aggregator: Aggregator | None = None,
    exception_handler: ExceptionHandler = raising_exception_handler,
    child_context_configurator: ChildContextConfigurator | None = None,
) -> CombinedObjectiveFunction[ParamsT]: ...


def combine(
    *args: ObjectiveLike[ParamsT] | Iterable[ObjectiveLike[ParamsT]],
    weights: Sequence[float] | None = None,
    reduction: Reducer | None = None,
    aggregator: Aggregator | None = None,
    exception_handler: ExceptionHandler = raising_exception_handler,
    child_context_configurator: ChildContextConfigurator | None = None,
) -> CombinedObjectiveFunction[ParamsT]:
    """
    Combine independent objective terms into one objective function.

    Terms may be supplied positionally or as one iterable, including a
    generator. Each term is an independent objective leaf with its own child
    evaluation context, so a concurrent scheduler may execute terms in
    parallel. Term values are multiplied by ``weights`` before they are
    combined.

    Args:
        *args: Objective functors or callables supplied as positional terms,
            or one iterable of terms.
        weights: Non-negative weight for each term. All weights default to
            ``1.0``.
        reduction: Callable that reduces the weighted term values to one
            float. The default is :func:`chemfit.sum_reducer`.
        aggregator: Context-aware alternative to ``reduction``. It receives
            the weighted values, child quantities, and parent context.
        exception_handler: Policy for an exception raised by a term. The
            default re-raises it.
        child_context_configurator: Optional callback that configures each
            term's child context before evaluation.

    Returns:
        A :class:`CombinedObjectiveFunction` containing the supplied terms.

    Raises:
        ValueError: If both ``reduction`` and ``aggregator`` are supplied, or
            if ``weights`` are invalid.

    Examples:
        Combine two preconfigured terms with different weights::

            objective = chemfit.combine(
                energy_term,
                force_term,
                weights=[1.0, 0.1],
            )

        An iterable or generator can be passed as the single positional
        argument, and the reduction can be replaced::

            objective = chemfit.combine(
                terms,
                reduction=chemfit.mean_reducer,
            )

    Notes:
        ``reduction`` consumes only the weighted scalar values. Use
        ``aggregator`` when combining terms also requires child quantities or
        access to the parent :class:`EvaluateContext`.

    """

    if len(args) == 1 and not callable(args[0]):
        objective_functions = tuple(args[0])
    else:
        objective_functions = cast("tuple[ObjectiveLike[ParamsT], ...]", args)

    return CombinedObjectiveFunction(
        objective_functions=objective_functions,
        weights=weights,
        reduction=reduction,
        aggregator=aggregator,
        exception_handler=exception_handler,
        child_context_configurator=child_context_configurator,
    )


def external_quantity(
    workdir: Path | str,
) -> ExternalQuantityComputer[Any, dict[str, Any]]:
    """
    Create a quantity computer for an external program.

    Every evaluation runs in a fresh isolated directory below ``workdir``.
    Commands and Python hooks execute as an ordered pipeline, after which
    configured output files are parsed into quantities. Parser inputs and
    completion paths are relative to that evaluation directory.

    Args:
        workdir: Parent directory in which isolated evaluation directories
            are created.

    Returns:
        A new :class:`ExternalQuantityComputer` for fluent configuration.

    Examples:
        Configure an external energy term::

            term = (
                chemfit.external_quantity("runs")
                .with_cmd(run_simulation)
                .with_parser(parse_energy, "energy.dat")
                .wait_for("done")
                .with_loss(loss, target=-10.0)
            )

        Static metadata can be attached at any point in the fluent chain::

            term = (
                chemfit.external_quantity("runs")
                .with_meta(dataset="training")
                .with_cmd(run_simulation)
                .with_parser(parse_energy, "energy.dat")
                .with_loss(loss, target=-10.0)
                .with_meta(observable="energy")
            )

    Notes:
        Static quantity-computer metadata is merged into ``ctx.meta`` before
        objective metadata. Consequently, metadata added after ``with_loss``
        wins when both levels use the same key.

    """
    return ExternalQuantityComputer(base_working_directory=workdir)


AtomsSource = Atoms | str | Path | Callable[[], Atoms]


def ase_quantity(atoms: AtomsSource, *, index: int | None = None) -> ASEComputer:
    """
    Create an ASE-backed quantity computer from atoms or an atoms source.

    ``atoms`` may be an :class:`ase.Atoms` object, a path readable by ASE, or
    a zero-argument callable that creates an ``Atoms`` object. An input
    ``Atoms`` object is copied for each evaluation. Path inputs are read
    lazily, and ``index`` selects one image from such a path.

    Args:
        atoms: Atoms object, structure-file path, or zero-argument atoms
            factory.
        index: ASE image index for a path input. It is invalid for an
            ``Atoms`` object or callable source.

    Returns:
        An :class:`ASEComputer` for fluent calculator, evaluator, processor,
        metadata, and loss configuration.

    Raises:
        ValueError: If ``index`` is supplied for a non-path source.
        TypeError: If ``atoms`` is not an accepted source.

    Examples:
        Configure a calculator and convert its energy into a loss term::

            term = (
                chemfit.ase_quantity("structure.xyz")
                .with_calculator(make_calculator)
                .with_loss(energy_loss, target=-12.4)
            )

        Metadata from the quantity and objective stages shares ``ctx.meta``::

            term = (
                chemfit.ase_quantity(atoms)
                .with_meta(dataset="liquid", temperature=298)
                .with_calculator(make_calculator)
                .with_loss(loss, target=0.997)
                .with_meta(observable="density")
            )

    Notes:
        During evaluation, the computer copies its cached base atoms, creates
        a calculator, runs the configured evaluator, and then extracts
        quantities. Static metadata precedence is existing context metadata,
        then quantity-computer metadata, then objective metadata; later
        stages win on key collisions.

    """
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
    scheduler: Scheduler[Any] | None = None,
) -> list[EvaluateContext]:
    """
    Evaluate one objective for several parameter mappings.

    A fresh :class:`EvaluateContext` is created for every parameter mapping.
    Evaluations may finish out of order, but the returned contexts are
    restored to the input order. Ordinary evaluation exceptions are
    re-raised.

    Args:
        objective: Objective functor or compatible one-argument callable.
        parameters: Parameter mappings to evaluate, in desired result order.
        executor: Executor used through an :class:`ExecutorTreeScheduler`.
            The caller retains ownership and is responsible for shutting it
            down.
        scheduler: Scheduler used to prepare and execute the objective.

    Returns:
        Fresh evaluation contexts in the same order as ``parameters``.

    Raises:
        ValueError: If neither or both of ``executor`` and ``scheduler`` are
            supplied.
        Exception: Re-raises an ordinary exception from an evaluation.

    Examples:
        Evaluate candidates concurrently and collect their losses::

            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=4) as executor:
                contexts = chemfit.evaluate_many(
                    objective,
                    parameter_sets,
                    executor=executor,
                )

            losses = [ctx.loss for ctx in contexts]

    Notes:
        Exactly one execution backend is required. A supplied executor remains
        caller-owned; ChemFit closes only the prepared schedule that borrows
        it.

    """
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
    """
    Results from the high-level Nevergrad fitting interface.

    Returned by :func:`fit_nevergrad` after the optimization finishes.
    ``recommendation`` is Nevergrad's final recommendation. In contrast,
    ``best_parameters`` and ``best_loss`` describe the best successful,
    optimizer-visible evaluation recorded by ChemFit. The recommendation and
    the best evaluated candidate need not be identical.

    ``contexts`` contains one context per candidate slot, including current and
    best evaluation state. Use :attr:`best_context` to select the context with
    the lowest recorded optimizer loss.

    Examples:
        Inspect both forms of result::

            print(result.recommendation)
            print(result.best_parameters)
            print(result.best_loss)
            print(result.best_context.opt_meta)

    """

    recommendation: ParamsT
    contexts: tuple[FitterEvaluateContext, ...]

    @property
    def best_context(self) -> FitterEvaluateContext:
        """
        Return the context containing the best successful evaluation.

        Returns:
            The candidate-slot context with the smallest non-``None``
            ``opt_loss``.

        Raises:
            RuntimeError: If no successful optimizer-visible evaluation was
                recorded.

        """
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
        """
        Return the lowest successful optimizer-visible loss.

        Returns:
            The ``opt_loss`` stored by :attr:`best_context`.

        Raises:
            RuntimeError: If no successful evaluation was recorded.

        """
        loss = self.best_context.opt_loss
        assert loss is not None
        return loss

    @property
    def best_parameters(self) -> Mapping[str, Any]:
        """
        Return the parameters from the best successful evaluation.

        These parameters correspond to :attr:`best_loss`, not necessarily to
        Nevergrad's final :attr:`recommendation`.

        Returns:
            The ``opt_params`` snapshot stored by :attr:`best_context`.

        Raises:
            RuntimeError: If no successful evaluation was recorded.

        """
        parameters = self.best_context.opt_params
        assert parameters is not None
        return parameters


def fit_nevergrad(
    objective: ObjectiveLike[ParamsT],
    initial: ParamsT,
    *,
    budget: int,
    workers: int = 1,
    execution_workers: int | None = None,
    bounds: Mapping[str, Any] | None = None,
    optimizer: str = "NgIohTuned",
    executor: Executor | None = None,
    scheduler: Scheduler[Any] | None = None,
    callbacks: Sequence[tuple[CallbackT, int]] | None = None,
    parametrization: Mapping[str, object] | None = None,
    initial_observations: Iterable[tuple[ParamsT, float | None]] | None = None,
) -> FitResult[ParamsT]:
    """
    Fit an objective with Nevergrad through ChemFit's recommended interface.

    The keys and nested structure of ``initial`` define the parameter mapping
    passed to the objective. Real-valued leaves receive inferred Nevergrad
    scalar parameters; ``bounds`` may constrain any of them. Use
    ``parametrization`` to replace selected leaves with explicit Nevergrad
    parameters, including choices, arrays, logarithmic scalars, or constants.

    Args:
        objective: Objective functor or compatible callable to minimize.
        initial: Initial nested parameter mapping. Its structure is preserved
            in evaluations and returned parameter mappings.
        budget: Total number of live objective evaluations.
        workers: Nevergrad candidate batch size and optimizer concurrency.
        execution_workers: Maximum number of objective leaf tasks executed
            concurrently by the built-in scheduler. Defaults to ``workers``.
        bounds: Optional nested mapping from parameter leaves to
            ``(lower, upper)`` pairs.
        optimizer: Name registered in ``nevergrad.optimizers.registry``.
        executor: Caller-owned executor used as the execution backend. It
            replaces the built-in backend and is not shut down by ChemFit.
        scheduler: Scheduler used as the execution backend instead of the
            built-in scheduler.
        callbacks: ``(callback, n_steps)`` pairs. Each callback receives the
            completed optimizer step and candidate-slot contexts at that
            interval and once for a final partial interval.
        parametrization: Optional nested mapping of explicit Nevergrad
            parameter objects. It may replace selected leaves of ``initial``.
        initial_observations: Previously evaluated ``(parameters, loss)``
            pairs used to seed Nevergrad before live optimization. Entries
            outside ``bounds`` are skipped. Replayed observations do not
            consume ``budget`` or trigger callbacks.

    Returns:
        A :class:`FitResult` containing Nevergrad's recommendation and the
        evaluation contexts recorded by ChemFit.

    Raises:
        ValueError: If budgets or worker counts are invalid, execution backend
            options conflict, or bounds or parametrization are incompatible.
        KeyError: If ``optimizer`` is not registered with Nevergrad.
        TypeError: If a non-real leaf has no explicit Nevergrad
            parametrization.

    Examples:
        Run an ordinary bounded fit::

            result = chemfit.fit_nevergrad(
                objective,
                initial={"epsilon": 0.8, "sigma": 1.1},
                bounds={
                    "epsilon": (0.1, 2.0),
                    "sigma": (0.5, 1.5),
                },
                budget=200,
                workers=4,
            )

            print(result.best_parameters)
            print(result.best_loss)

        Optimizer and objective execution concurrency can be tuned
        independently::

            result = chemfit.fit_nevergrad(
                objective,
                initial=initial,
                budget=200,
                workers=4,
                execution_workers=8,
            )

        Seed a new run with results from earlier evaluations::

            result = chemfit.fit_nevergrad(
                objective,
                initial=initial,
                budget=200,
                initial_observations=[
                    ({"epsilon": 0.7, "sigma": 1.0}, 0.42),
                    ({"epsilon": 0.9, "sigma": 1.2}, 0.31),
                ],
            )

    Notes:
        ``workers`` controls how many candidates Nevergrad asks for in a batch;
        it does not impose a leaf-execution limit on a custom backend.
        ``execution_workers`` defaults to ``workers`` and applies only to the
        built-in scheduler. Supplying ``executor`` or ``scheduler`` replaces
        that backend, so ``execution_workers`` cannot be supplied alongside
        either one.

        The returned ``recommendation`` is Nevergrad's final recommendation.
        ``best_parameters`` and ``best_loss`` instead identify the best
        successful optimizer-visible evaluation recorded by ChemFit.

        Initial observations provide an approximate warm start; they do not
        restore Nevergrad's internal optimizer state.

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
        initial_observations=initial_observations,
    )

    return FitResult(recommendation=recommendation, contexts=tuple(fitter.contexts))
