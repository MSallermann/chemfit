from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Callable, Mapping
from numbers import Real
from typing import TYPE_CHECKING, Any, Generic, cast

import nevergrad as ng
import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, minimize
from typing_extensions import TypeVar

from chemfit.abstract_objective_function import EvaluateContext, ObjectiveFunctor
from chemfit.scheduling import EvaluationRequest
from chemfit.tree_schedule import SerialTreeScheduler
from chemfit.utils import check_params_near_bounds
from chemfit.wrap_funcs import WrappedObjectiveFunctor
from pydictnest import flatten_dict, unflatten_dict

if TYPE_CHECKING:
    from collections.abc import Iterable

    from chemfit.scheduling import PreparedSchedule, Scheduler

logger = logging.getLogger(__name__)

# Fitter both consumes candidates (evaluate/fit inputs) and returns parameters
# (finish/fit outputs), so its parameter type must remain invariant.  The
# default keeps concise unannotated lambdas and heterogeneous nested dicts
# usable; an annotated objective still determines a more precise type.
ParametersT = TypeVar("ParametersT", bound=Mapping[str, Any], default=dict[str, Any])


class FitterEvaluateContext(EvaluateContext):
    def __init__(self):
        """
        Initialize fitter-specific evaluation state.

        This context extends ``EvaluateContext`` with optimization-specific
        tracking fields that record the number of evaluations performed and
        the best loss, parameters, and metadata observed so far during a fit.
        """

        super().__init__()
        self.n_evals: int = 0
        self.opt_loss: float | None = None
        self.opt_params: dict[str, Any] | None = None
        self.opt_meta: dict[str, Any] | None = None
        self.opt_quantities: dict[str, Any] | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["n_evals"] = self.n_evals
        state["opt_loss"] = self.opt_loss
        state["opt_params"] = self.opt_params
        state["opt_meta"] = self.opt_meta
        state["opt_quantities"] = self.opt_quantities
        return state

    def __setstate__(self, state: dict[str, Any]):
        super().__setstate__(state)
        self.n_evals = state["n_evals"]
        self.opt_loss = state["opt_loss"]
        self.opt_params = state["opt_params"]
        self.opt_meta = state["opt_meta"]
        self.opt_quantities = state["opt_quantities"]

    def to_result_state(self) -> dict[str, Any]:
        state = super().to_result_state()
        state.update(
            {
                "n_evals": self.n_evals,
                "opt_loss": self.opt_loss,
                "opt_params": self.opt_params,
                "opt_meta": self.opt_meta,
                "opt_quantities": self.opt_quantities,
            }
        )
        return state

    def apply_result_state(self, state: dict[str, Any]):
        super().apply_result_state(state)
        self.n_evals = state["n_evals"]
        self.opt_loss = state["opt_loss"]
        self.opt_params = state["opt_params"]
        self.opt_meta = state["opt_meta"]
        self.opt_quantities = state["opt_quantities"]


CallbackT = Callable[[int, list[FitterEvaluateContext]], None]


class FitterEvaluationHook:
    """Normalize successful root losses before they leave objective lifecycle."""

    @staticmethod
    def post_eval(ctx: EvaluateContext) -> None:
        if not isinstance(ctx, FitterEvaluateContext):
            return

        value_bad_params = getattr(
            ctx.config,
            "_fitter_value_bad_params",
            None,
        )
        if value_bad_params is None:
            return

        # Failed evaluations are interpreted by Fitter after evaluate_many()
        # returns the root Exception outcome.
        if getattr(ctx.temp, "exception", None) is not None:
            return

        ctx.loss = _sanitize_loss(
            ctx.loss,
            value_bad_params=float(value_bad_params),
        )


_FITTER_EVALUATION_HOOK = FitterEvaluationHook()


def _sanitize_loss(value: object, value_bad_params: float) -> float:
    """Convert an optimizer-facing objective value to a finite float."""

    if not isinstance(value, Real):
        logger.debug(
            "Objective function did not return a single float, but returned "
            f"`{value}` with type {type(value)}. "
            f"Clipping loss to {value_bad_params}"
        )
        return float(value_bad_params)

    loss = float(value)
    if math.isnan(loss):
        logger.debug(
            f"Objective function returned NaN. Clipping loss to {value_bad_params}"
        )
        return float(value_bad_params)

    return loss


class Fitter(Generic[ParametersT]):
    def __init__(
        self,
        objective_function: (
            Callable[[ParametersT], float] | ObjectiveFunctor[ParametersT]
        ),
        initial_params: Mapping[str, Any],
        bounds: Mapping[str, object] | None = None,
        near_bound_tol: float | None = None,
        value_bad_params: float = 1e5,
        swallow_exceptions: bool = False,
        log_exceptions: bool = True,
        scheduler: Scheduler[Any] | None = None,
    ) -> None:
        """
        Driver class for parameter optimization.

        A `Fitter` evaluates an objective (either a plain callable or an
        `ObjectiveFunctor`) through a prepared scheduler and exposes
        convenience methods for running optimizations with nevergrad and
        SciPy.

        Args:
            objective_function (Callable | ObjectiveFunctor): Objective to
                be minimized. If a plain callable is provided, it is
                converted to an `ObjectiveFunctor` using
                `objective`.
            initial_params: Nested mapping of concrete initial parameter
                values passed to the objective.
            bounds (Mapping[str, object] | None, optional): Bounds for each
                parameter. The structure must mirror ``initial_params``,
                but may omit bounds for parameters.
                Defaults to None.
            near_bound_tol (float | None, optional): If provided, parameters
                whose optimized values lie within this relative distance of
                their bounds will trigger a warning in `hook_post_fit`.
                Defaults to None.
            value_bad_params (float, optional): Penalty used for invalid,
                non-scalar, NaN, or swallowed-exception objective results.
                Defaults to 1e5.
            scheduler: Scheduler used to evaluate the objective. Defaults
                to ``SerialTreeScheduler``.

        """

        if not isinstance(initial_params, Mapping):
            msg = "initial_params must be a mapping"
            raise TypeError(msg)

        self.initial_parameters = cast(
            "ParametersT", copy.deepcopy(dict(initial_params))
        )
        self.bounds: Mapping[str, object] = {} if bounds is None else bounds

        # Make sure that we have an ObjectiveFunctor instance
        if not isinstance(objective_function, ObjectiveFunctor):
            objective_function = WrappedObjectiveFunctor(
                func=objective_function, pass_ctx=False
            )

        self.objective_function: ObjectiveFunctor[ParametersT] = objective_function

        # Register one stateless fitter hook on the root objective. Per-fit
        # configuration lives on FitterEvaluateContext, so sharing an objective
        # between fitters does not put fitter-specific state on the hook itself.
        if (
            _FITTER_EVALUATION_HOOK.post_eval
            not in self.objective_function.post_eval_hooks
        ):
            self.objective_function.register_eval_hook(hook=_FITTER_EVALUATION_HOOK)

        self.value_bad_params: float = value_bad_params
        self.swallow_exceptions = swallow_exceptions
        self.log_exceptions = log_exceptions
        self._scheduler = SerialTreeScheduler() if scheduler is None else scheduler
        self._schedule: PreparedSchedule[ParametersT] | None = None

        self.near_bound_tol = near_bound_tol

        self.contexts: list[FitterEvaluateContext] = []

        self.callbacks: list[
            tuple[Callable[[int, list[FitterEvaluateContext]], None], int]
        ] = []

    def _record_loss(
        self,
        parameters: ParametersT,
        value: object,
        ctx: FitterEvaluateContext,
    ) -> float:
        """Record one optimizer-visible evaluation result."""

        loss = _sanitize_loss(value, self.value_bad_params)
        ctx.loss = loss
        ctx.n_evals += 1

        if ctx.opt_loss is None or loss < ctx.opt_loss:
            ctx.opt_loss = loss
            # Some supported leaves (for example NumPy arrays) are mutable.
            # Keep the incumbent as a true snapshot of the evaluated values.

            ctx.opt_params = copy.deepcopy(dict(parameters))
            ctx.opt_meta = copy.deepcopy(ctx.meta)
            ctx.opt_quantities = copy.deepcopy(ctx.quantities)

        return loss

    def _process_evaluation_outcome(
        self,
        parameters: ParametersT,
        outcome: float | Exception,
        ctx: FitterEvaluateContext,
    ) -> float:
        """Convert a completed scheduler root outcome into an optimizer loss."""

        if isinstance(outcome, Exception):
            if self.log_exceptions:
                logger.error(
                    "Caught exception while evaluating objective function.",
                    exc_info=(type(outcome), outcome, outcome.__traceback__),
                )

            if not self.swallow_exceptions:
                raise outcome

            return self._record_loss(
                parameters=parameters,
                value=float("nan"),
                ctx=ctx,
            )

        # Root post-evaluation hooks may intentionally modify ctx.loss. Treat
        # that context value as authoritative after a successful lifecycle.
        value: object = ctx.loss if ctx.loss is not None else outcome
        return self._record_loss(
            parameters=parameters,
            value=value,
            ctx=ctx,
        )

    def _close_schedule(self) -> None:
        """Close the currently prepared schedule, if any."""

        if self._schedule is not None:
            self._schedule.close()
            self._schedule = None

    def register_callback(self, func: CallbackT, n_steps: int) -> None:
        """
        Register a callback to be executed during optimization.

        The callback is invoked after every ``n_steps`` completed optimizer
        steps and once at the end of the fit if the final step does not fall
        exactly on the requested interval. The callback receives the number
        of completed optimizer steps and the list of `FitterEvaluateContext`
        instances used by the fitter.

        Args:
            func (Callable[[int, list[FitterEvaluateContext]], None]):
                Callback function of the form ``func(step, contexts)``.
            n_steps (int): Number of completed optimizer steps between callback
                invocations.

        """
        if n_steps < 1:
            msg = "n_steps must be at least 1"
            raise ValueError(msg)

        self.callbacks.append((func, n_steps))

    def _dispatch_callbacks(self, final: bool = False) -> None:
        """Dispatch callbacks due at the current completed optimizer step."""

        if self._session_step == 0:
            return

        for callback, n_steps in self.callbacks:
            due = self._session_step % n_steps == 0
            if (not final and due) or (final and not due):
                callback(self._session_step, self.contexts)

    def _hook_pre_fit(self):
        """Run bookkeeping steps before starting an optimization."""

        logger.info("Start fitting")
        self.time_fit_start = time.time()

    def _hook_post_fit(self, opt_params: ParametersT):
        """
        Run bookkeeping steps after an optimization.

        This method records the fit end time, logs completion, and optionally
        warns if any optimized parameters lie near or outside their bounds.

        Args:
            opt_params: Optimized parameter dictionary returned by the
                optimizer.

        """

        self.time_fit_end = time.time()
        logger.info("End fitting")

        if self.near_bound_tol is not None:
            self.problematic_params = check_params_near_bounds(
                opt_params, self.bounds, self.near_bound_tol
            )

            if len(self.problematic_params) > 0:
                logger.warning(
                    f"The following parameters are near or outside the bounds (tolerance {self.near_bound_tol * 100:.1f}%):"
                )
                for kp, vp, lower, upper in self.problematic_params:
                    logger.warning(
                        f"    parameter = {kp}, lower = {lower}, value = {vp}, upper = {upper}"
                    )

    def init(
        self,
        num_workers: int = 1,
        contexts: list[FitterEvaluateContext] | None = None,
    ) -> None:
        """
        Initialize a user-driven optimization session.

        After initialization the user owns the optimization loop: obtain
        candidates from an optimizer, pass them to :meth:`evaluate`, feed the
        returned losses back to the optimizer, and call :meth:`step` once per
        optimizer step. Call :meth:`finish` with the optimizer's final
        recommendation when the loop is complete.

        ``num_workers`` controls the maximum candidate batch size. Actual
        execution is delegated entirely to the configured scheduler.
        """

        if num_workers < 1:
            msg = "num_workers must be at least 1"
            raise ValueError(msg)
        if contexts is not None and len(contexts) != num_workers:
            msg = "contexts must contain one context per worker"
            raise ValueError(msg)

        self._close_schedule()

        self._hook_pre_fit()
        self._session_num_workers = num_workers
        self._session_step = 0
        self.contexts = (
            [FitterEvaluateContext() for _ in range(num_workers)]
            if contexts is None
            else contexts
        )

        for ctx in self.contexts:
            ctx.config._fitter_value_bad_params = self.value_bad_params  # noqa: SLF001

        self._schedule = self._scheduler.prepare(self.objective_function)

    def evaluate(
        self,
        parameters: ParametersT | list[ParametersT],
        context_index: int = 0,
    ) -> float | list[float]:
        """
        Evaluate one candidate or a candidate batch through the prepared schedule.

        A mapping produces one loss. A list produces a list of losses in
        input order and may contain at most ``num_workers`` candidates.
        Scheduler results may complete out of order; ``EvaluationResult.index``
        is used to restore the original request order before returning.
        """

        if not hasattr(self, "_session_num_workers") or self._schedule is None:
            msg = "call fitter.init() before fitter.evaluate()"
            raise RuntimeError(msg)

        is_single = isinstance(parameters, Mapping)

        if is_single:
            batch = [cast("ParametersT", dict(parameters))]
            contexts = [self.contexts[context_index]]
        else:
            batch = parameters

            if len(batch) > self._session_num_workers:
                msg = "a batch cannot contain more candidates than workers"
                raise ValueError(msg)
            if len(batch) == 0:
                return []

            contexts = self.contexts[: len(batch)]

        requests = [
            EvaluationRequest(
                parameters=params,
                ctx=ctx,
            )
            for params, ctx in zip(batch, contexts, strict=True)
        ]

        missing = object()
        outcomes: list[float | Exception | object] = [missing] * len(requests)

        # Fully drain the batch before applying fitter exception policy. An
        # ordinary objective Exception is a normal scheduler outcome, and the
        # scheduler may still have other candidate work in flight.
        for result in self._schedule.evaluate_many(requests):
            outcomes[result.index] = result.value

        if any(outcome is missing for outcome in outcomes):
            msg = "prepared schedule did not produce a result for every request"
            raise RuntimeError(msg)

        losses: list[float] = []
        for params, ctx, outcome in zip(batch, contexts, outcomes, strict=True):
            assert outcome is not missing
            losses.append(
                self._process_evaluation_outcome(
                    parameters=params,
                    outcome=cast("float | Exception", outcome),
                    ctx=ctx,
                )
            )

        if is_single:
            return losses[0]

        return losses

    def step(self, step: int | None = None) -> None:
        """
        Notify ChemFit that the user completed an optimizer step.

        This advances the completed-step counter and dispatches registered
        fitter callbacks. If ``step`` is supplied, it is interpreted as the
        number of completed optimizer steps.
        """

        if not hasattr(self, "_session_step"):
            msg = "call fitter.init() before fitter.step()"
            raise RuntimeError(msg)

        if step is None:
            self._session_step += 1
        else:
            self._session_step = step

        self._dispatch_callbacks()

    def finish(self, opt_params: ParametersT | None = None) -> ParametersT:
        """
        Finalize a user-driven session and return its chosen parameters.

        When no recommendation is supplied, the best candidate evaluated by
        ChemFit is used.
        """

        if opt_params is None:
            evaluated = [ctx for ctx in self.contexts if ctx.opt_loss is not None]
            if not evaluated:
                msg = "cannot finish before evaluating a candidate"
                raise RuntimeError(msg)
            best_context = min(evaluated, key=lambda ctx: cast("float", ctx.opt_loss))
            assert best_context.opt_params is not None
            opt_params = cast("ParametersT", copy.deepcopy(best_context.opt_params))

        try:
            self._hook_post_fit(opt_params)
            self._dispatch_callbacks(final=True)
            return opt_params
        finally:
            self._close_schedule()

    def _make_nevergrad_parameterization(
        self, parametrization: Mapping[str, object] | None
    ) -> ng.p.Instrumentation:
        """Build Nevergrad's representation from concrete parameter values."""
        flat_initial_params = flatten_dict(self.initial_parameters)
        flat_bounds = flatten_dict(self.bounds)

        if parametrization is None:
            flat_parametrization = {}
        elif isinstance(parametrization, Mapping):
            flat_parametrization = flatten_dict(parametrization)
        else:
            msg = "parametrization must be a mapping"
            raise TypeError(msg)

        unknown_keys = set(flat_parametrization) - set(flat_initial_params)
        if unknown_keys:
            names = ", ".join(repr(key) for key in sorted(unknown_keys))
            msg = f"parametrization contains unknown parameter leaves: {names}"
            raise ValueError(msg)

        ng_params = ng.p.Dict()
        for key, value in flat_initial_params.items():
            if key in flat_parametrization:
                parameter = flat_parametrization[key]
                if not isinstance(parameter, ng.p.Parameter):
                    msg = (
                        "parametrization leaves must be Nevergrad parameters, "
                        f"got {key!r}"
                    )
                    raise TypeError(msg)
                ng_params[key] = parameter.copy()
            elif isinstance(value, Real):
                lower, upper = flat_bounds.get(key, (None, None))
                ng_params[key] = ng.p.Scalar(
                    init=float(value), lower=lower, upper=upper
                )
            else:
                msg = (
                    f"cannot infer a Nevergrad parameter for leaf {key!r} "
                    f"with type {type(value).__name__}; provide one in "
                    "parametrization (use ng.p.Constant(...) to keep the "
                    "value fixed)"
                )
                raise TypeError(msg)

        return ng.p.Instrumentation(ng_params)

    def fit_nevergrad(
        self,
        budget: int,
        optimizer_str: str = "NgIohTuned",
        num_workers: int = 1,
        contexts: list[FitterEvaluateContext] | None = None,
        parametrization: Mapping[str, object] | None = None,
        initial_observations: Iterable[tuple[ParametersT, float | None]] | None = None,
    ) -> ParametersT:
        """
        Optimize parameters using a nevergrad optimizer.

        This method drives nevergrad's ask/tell interface and evaluates each
        candidate batch through the configured scheduler. One
        ``FitterEvaluateContext`` is used per candidate slot so that
        evaluation-side state can be tracked independently.

        Args:
            budget: Total number of objective evaluations to allow.
            optimizer_str: Name of the nevergrad optimizer to use. Must be a
                key in ``ng.optimizers.registry``.
            num_workers: Maximum number of candidates requested from Nevergrad
                in one ask/tell step. The configured scheduler determines
                how those candidates are executed.
            contexts: Optional list of per-candidate-slot fitter contexts. If
                provided, its length must equal ``num_workers``.
            parametrization: Optional nested mapping of Nevergrad parameter
                leaves. It may override any leaf in ``initial_params``; other
                real-valued scalar leaves use ``Scalar``. All other leaf types
                must be specified explicitly.
            initial_observations:
                Optional iterable of previously evaluated ``(parameters, loss)``
                pairs used to seed the optimizer.
                These observations are replayed into the optimizer before the main
                optimization loop begins. This allows approximate continuation of a
                previous run or warm-starting a new optimization.
                If any parameter set violates the bounds, it is skipped.
                These observations do not consume evaluations from the main budget
                and do not trigger callbacks.
                This does not restore the exact internal state of the optimizer.
                Only the provided observations are injected.

        Returns:
            Dictionary of optimized parameter values.

        Raises:
            KeyError: If ``optimizer_str`` is not found in the nevergrad
                optimizer registry.
            ValueError: If ``contexts`` is provided and its length does
                not equal ``num_workers``.

        Side Effects:
            - Initializes fitter bookkeeping via ``_hook_pre_fit()``.
            - Populates ``self.contexts`` with one context per worker.
            - Invokes registered callbacks during optimization.
            - Runs post-fit checks via ``_hook_post_fit()``.

        """

        flat_initial_params = flatten_dict(self.initial_parameters)
        flat_bounds = flatten_dict(self.bounds)
        instru = self._make_nevergrad_parameterization(parametrization)

        try:
            optimizer_cls = ng.optimizers.registry[optimizer_str]
        except KeyError as exc:
            available_solvers = list(ng.optimizers.registry.keys())
            msg = (
                f"Unknown nevergrad optimizer {optimizer_str!r}. "
                f"Available solvers: {available_solvers}"
            )
            raise KeyError(msg) from exc

        optimizer = optimizer_cls(
            parametrization=instru, budget=budget, num_workers=num_workers
        )

        self.init(num_workers=num_workers, contexts=contexts)

        if initial_observations is not None:
            for restart_params, restart_loss_value in initial_observations:
                skip = False
                flat_params = flatten_dict(restart_params)

                for key, (lower, upper) in flat_bounds.items():
                    restart_value = flat_params.get(key)
                    if restart_value is not None and (
                        restart_value < lower or restart_value > upper
                    ):
                        skip = True

                if skip:
                    continue

                optimizer.suggest(flat_params)
                asked_params = optimizer.ask()

                # Replay the recorded value through the same fitter-side
                # normalization and incumbent bookkeeping used for live results.
                post_processed_loss_value = self._record_loss(
                    parameters=restart_params,
                    value=restart_loss_value,
                    ctx=self.contexts[0],
                )
                optimizer.tell(asked_params, post_processed_loss_value)

        for step, batch_start in enumerate(range(0, budget, num_workers)):
            batch_size = min(num_workers, budget - batch_start)

            if step == 0:
                optimizer.suggest(flat_initial_params)

            asked_params = [optimizer.ask() for _ in range(batch_size)]
            flat_params = [candidate.value[0][0] for candidate in asked_params]
            nested_params = [
                cast(
                    "ParametersT",
                    unflatten_dict(parameters, dict_factory=dict[str, Any]),
                )
                for parameters in flat_params
            ]
            asked_losses = self.evaluate(nested_params)
            assert isinstance(asked_losses, list)

            for params, loss in zip(asked_params, asked_losses, strict=True):
                optimizer.tell(params, loss)

            self.step()

        recommendation = optimizer.provide_recommendation()
        args, _ = recommendation.value
        flat_opt_params = args[0]
        opt_params = cast(
            "ParametersT",
            unflatten_dict(flat_opt_params, dict_factory=dict[str, Any]),
        )

        return self.finish(opt_params)

    def fit_scipy(
        self,
        method: str = "L-BFGS-B",
        ctx: FitterEvaluateContext | None = None,
        **kwargs,
    ) -> ParametersT:
        """
        Optimize parameters using ``scipy.optimize.minimize``.

        The parameter dictionary is flattened into a vector representation for
        SciPy and reconstructed on each objective evaluation. Because SciPy's
        ``minimize`` interface is synchronous, a single
        ``FitterEvaluateContext`` is used for the full optimization run.

        Args:
            method: Optimization method passed to
                ``scipy.optimize.minimize``.
            ctx: Optional fitter evaluation context to reuse during the fit.
                If ``None``, a new one is created.
            **kwargs: Additional keyword arguments forwarded to
                ``scipy.optimize.minimize``.

        Returns:
            Dictionary of optimized parameter values.

        Warning:
            If the optimizer does not converge, a warning is logged.

        Side Effects:
            - Initializes fitter bookkeeping via ``_hook_pre_fit()``.
            - Populates ``self.contexts`` with a single context.
            - Invokes registered callbacks during optimization.
            - Runs post-fit checks via ``_hook_post_fit()``.

        """

        flat_params = flatten_dict(self.initial_parameters)
        non_scalar_leaves = [
            key for key, value in flat_params.items() if not isinstance(value, Real)
        ]
        if non_scalar_leaves:
            names = ", ".join(repr(key) for key in non_scalar_leaves)
            msg = f"fit_scipy requires real-valued scalar leaves, got {names}"
            raise TypeError(msg)

        flat_bounds = flatten_dict(self.bounds)
        self._keys = flat_params.keys()
        x0 = np.array([flat_params[key] for key in self._keys])

        if len(flat_bounds) == 0:
            bounds = None
        else:
            bounds = np.array(
                [flat_bounds.get(key, (None, None)) for key in self._keys]
            )

        # Since we know that scipy.optimize works synchronously, we create a single context, which we'll keep alive.
        self.init(contexts=None if ctx is None else [ctx])

        def f_scipy(x: npt.NDArray) -> float:
            parameters = cast(
                "ParametersT",
                unflatten_dict(
                    dict(zip(self._keys, x, strict=False)), dict_factory=dict[str, Any]
                ),
            )
            loss = self.evaluate(parameters)
            assert isinstance(loss, float)
            return loss

        def callback_scipy(_intermediate_result: OptimizeResult):
            self.step()

        res = minimize(
            f_scipy, x0, method=method, bounds=bounds, **kwargs, callback=callback_scipy
        )

        if not res.success:
            logger.warning(f"Fit did not converge: {res.message}")

        opt_params = cast(
            "ParametersT", unflatten_dict(dict(zip(self._keys, res.x, strict=False)))
        )

        return self.finish(opt_params)
