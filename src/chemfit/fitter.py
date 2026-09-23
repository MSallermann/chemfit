from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable, Generic, cast

import nevergrad as ng
import numpy as np
import numpy.typing as npt
from pydictnest import flatten_dict, unflatten_dict
from scipy.optimize import OptimizeResult, minimize
from typing_extensions import TypeVar

from chemfit.abstract_objective_function import (
    EvaluateContext,
    ExecutorLike,
    ObjectiveFunctor,
)
from chemfit.executor_utils import map_with_context
from chemfit.utils import check_params_near_bounds
from chemfit.wrap_funcs import WrappedObjectiveFunctor

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Fitter both consumes candidates (ask/fit inputs) and returns parameters
# (finish/fit outputs), so its parameter type must remain invariant.  The
# default keeps concise unannotated lambdas and heterogeneous nested dicts
# usable; an annotated objective still determines a more precise type.
ParametersT = TypeVar("ParametersT", bound=dict[str, Any], default=dict[str, Any])


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


class FitterObjectiveFunctor(ObjectiveFunctor[ParametersT], Generic[ParametersT]):
    def __init__(
        self,
        wrap_me: ObjectiveFunctor[ParametersT],
        swallow_exceptions: bool = False,
        log_exceptions: bool = True,
        value_bad_params: float = 1e5,
    ):
        """
        Initialize a fitter-specific objective wrapper.

        This wrapper sits between a raw objective and an optimizer. It adds
        basic robustness and tracking behavior on top of the wrapped
        objective:

        - exceptions may be logged and optionally swallowed
        - non-scalar or NaN return values are replaced by a large penalty
        - the attached ``FitterEvaluateContext`` is updated with the number of
          evaluations and the best loss/parameters seen so far

        Args:
            wrap_me: Underlying objective functor to evaluate.
            swallow_exceptions: If ``True``, exceptions raised by the wrapped
                objective are converted into a penalized objective value
                instead of being re-raised.
            log_exceptions: If ``True``, exceptions raised by the wrapped
                objective are logged.
            value_bad_params (float, optional): Threshold used to represent invalid or numerically
                unstable parameter regions. Defaults to 1e5.

        """

        self.wrap_me = wrap_me
        self.value_bad_params = value_bad_params
        self.swallow_exceptions: bool = swallow_exceptions
        self.log_exceptions: bool = log_exceptions

    def post_process_return_value(
        self,
        parameters: ParametersT,
        value: float | None,
        ctx: FitterEvaluateContext,
    ) -> float:
        ctx.n_evals += 1

        # then we make sure that the value is a float
        if not isinstance(value, Real):
            logger.debug(
                f"Objective function did not return a single float, but returned `{value}` with type {type(value)}. Clipping loss to {self.value_bad_params}"
            )

            value = float(self.value_bad_params)

        if math.isnan(value):
            logger.debug(
                f"Objective function returned NaN. Clipping loss to {self.value_bad_params}"
            )
            value = self.value_bad_params

        loss = float(value)

        if ctx.opt_loss is None or loss < ctx.opt_loss:
            ctx.opt_loss = loss
            # Some supported leaves (for example NumPy arrays) are mutable.
            # Keep the incumbent as a true snapshot of the evaluated values.
            ctx.opt_params = copy.deepcopy(dict(parameters))
            ctx.opt_meta = dict(ctx.meta)
            ctx.opt_quantities = ctx.quantities

        return loss

    def __call__(
        self, parameters: ParametersT, ctx: EvaluateContext | None = None
    ) -> float:
        if ctx is None:
            ctx = FitterEvaluateContext()
        elif not isinstance(ctx, FitterEvaluateContext):
            msg = "FitterObjectiveFunctor requires a FitterEvaluateContext"
            raise TypeError(msg)

        # first we try if we can get a value at all
        try:
            value = self.wrap_me(parameters, ctx)
        except Exception as e:
            if self.log_exceptions:
                logger.exception(
                    "Caught exception while evaluating objective function."
                )

            if not self.swallow_exceptions:
                raise e

            value = float("nan")

        return self.post_process_return_value(
            parameters=parameters, value=value, ctx=ctx
        )


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
    ) -> None:
        """
        Driver class for parameter optimization.

        A `Fitter` wraps an objective (either a plain callable or an
        `ObjectiveFunctor`) in a `FitterObjectiveFunctor` and exposes
        convenience methods for running optimizations with nevergrad and
        SciPy.

        Args:
            objective_function (Callable | ObjectiveFunctor): Objective to
                be minimized. If a plain callable is provided, it is
                converted to an `ObjectiveFunctor` using
                `to_objective_functor`.
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
            value_bad_params (float, optional): Threshold used by some
                objective wrappers to represent invalid or numerically
                unstable parameter regions. Defaults to 1e5.

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

        self.objective_function = FitterObjectiveFunctor(
            objective_function,
            swallow_exceptions=swallow_exceptions,
            log_exceptions=log_exceptions,
            value_bad_params=value_bad_params,
        )

        self.value_bad_params: float = value_bad_params

        self.near_bound_tol = near_bound_tol

        self.contexts: list[FitterEvaluateContext] = []

        self.callbacks: list[
            tuple[Callable[[int, list[FitterEvaluateContext]], None], int]
        ] = []

    def register_callback(
        self, func: Callable[[int, list[FitterEvaluateContext]], None], n_steps: int
    ):
        """
        Register a callback to be executed during optimization.

        The callback is invoked every ``n_steps`` iterations (or
        nevergrad/SciPy "steps", depending on the backend), and receives
        the current step index and the list of `FitterEvaluateContext`
        instances used by the fitter.

        Args:
            func (Callable[[int, list[FitterEvaluateContext]], None]):
                Callback function of the form ``func(step, contexts)``.
            n_steps (int): Interval (in steps) at which the callback is
                invoked.

        """
        self.callbacks.append((func, n_steps))

    def _unify_callbacks(
        self,
    ) -> (
        tuple[Callable[[int, list[FitterEvaluateContext]], None], int]
        | tuple[None, int]
    ):
        """Generate a single callback from the list of callbacks."""

        if len(self.callbacks) == 0:
            return None, 0

        min_n_steps = min([n_steps for (_, n_steps) in self.callbacks])

        def callback(step: int, ctxs: list[FitterEvaluateContext]):
            for cb, n_steps in self.callbacks:
                if step % n_steps == 0:
                    cb(step, ctxs)

        return callback, min_n_steps

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
        executor: ExecutorLike | None = None,
    ) -> None:
        """
        Initialize a user-driven optimization session.

        After initialization the user owns the optimization loop: obtain
        candidates from an optimizer, pass them to :meth:`ask`, feed the
        returned losses back to the optimizer, and call :meth:`tell` once per
        optimizer step. Call :meth:`finish` with the optimizer's final
        recommendation when the loop is complete.
        """

        if num_workers < 1:
            msg = "num_workers must be at least 1"
            raise ValueError(msg)
        if contexts is not None and len(contexts) != num_workers:
            msg = "contexts must contain one context per worker"
            raise ValueError(msg)
        self._owns_executor = num_workers != 1 and executor is None
        if self._owns_executor:
            executor = ThreadPoolExecutor(num_workers)

        self._hook_pre_fit()
        self._session_executor = executor
        self._session_num_workers = num_workers
        self._session_step = 0
        self.contexts = (
            [FitterEvaluateContext() for _ in range(num_workers)]
            if contexts is None
            else contexts
        )

    def ask(
        self,
        parameters: ParametersT | list[ParametersT],
        context_index: int = 0,
    ) -> float | list[float]:
        """
        Evaluate one candidate or a parallel batch proposed by the user.

        A mapping produces one loss. A list produces a list of losses in
        input order and may contain at most ``num_workers`` candidates.
        """

        if not hasattr(self, "_session_num_workers"):
            msg = "call fitter.init() before fitter.ask()"
            raise RuntimeError(msg)

        if isinstance(parameters, Mapping):
            return self.objective_function(
                cast("ParametersT", dict(parameters)), self.contexts[context_index]
            )

        if len(parameters) > self._session_num_workers:
            msg = "a batch cannot contain more candidates than workers"
            raise ValueError(msg)
        if len(parameters) == 0:
            return []
        if len(parameters) == 1:
            return [self.objective_function(parameters[0], self.contexts[0])]

        if self._session_executor is None:
            msg = "parallel evaluation requires an executor"
            raise RuntimeError(msg)

        return map_with_context(
            self._session_executor,
            self.objective_function,
            parameters,
            ctxs=self.contexts[: len(parameters)],
        )

    def tell(self, step: int | None = None) -> None:
        """
        Notify ChemFit that the user completed an optimizer step.

        This dispatches registered fitter callbacks. If ``step`` is omitted,
        an internal zero-based step counter is used and advanced automatically.
        """

        if not hasattr(self, "_session_step"):
            msg = "call fitter.init() before fitter.tell()"
            raise RuntimeError(msg)

        current_step = self._session_step if step is None else step
        callback, n_steps = self._unify_callbacks()

        if callback is not None and current_step % n_steps == 0:
            callback(current_step, self.contexts)

        self._session_step = current_step + 1

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

        self._hook_post_fit(opt_params)

        if self._owns_executor:
            cast("ThreadPoolExecutor", self._session_executor).shutdown()

        return opt_params

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
        executor: ExecutorLike | None = None,
        parametrization: Mapping[str, object] | None = None,
        initial_observations: (
            Iterable[tuple[ParametersT, float | None]] | None
        ) = None,
    ) -> ParametersT:
        """
        Optimize parameters using a nevergrad optimizer.

        This method drives nevergrad's ask/tell interface and can evaluate
        multiple candidate points in parallel through an ``ExecutorLike``
        instance. One ``FitterEvaluateContext`` is used per worker so that
        evaluation-side state can be tracked independently.

        Args:
            budget: Total number of objective evaluations to allow.
            optimizer_str: Name of the nevergrad optimizer to use. Must be a
                key in ``ng.optimizers.registry``.
            num_workers: Number of points to evaluate in parallel per ask/tell
                step.
            contexts: Optional list of per-worker fitter contexts. If
                provided, its length must equal ``num_workers``.
            executor: Optional executor used for parallel evaluation when
                ``num_workers > 1``. If ``None``, a ``ThreadPoolExecutor`` is
                created.
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

        self.init(num_workers=num_workers, contexts=contexts, executor=executor)

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

                # The recorded loss value may be changed by our wrapper
                # Also we record the side effects on the context this way
                post_processed_loss_value = (
                    self.objective_function.post_process_return_value(
                        parameters=restart_params,
                        value=restart_loss_value,
                        ctx=self.contexts[0],
                    )
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
            asked_losses = self.ask(nested_params)
            assert isinstance(asked_losses, list)

            for params, loss in zip(asked_params, asked_losses, strict=True):
                optimizer.tell(params, loss)

            self.tell(step)

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
                unflatten_dict(dict(zip(self._keys, x)), dict_factory=dict[str, Any]),
            )
            loss = self.ask(parameters)
            assert isinstance(loss, float)
            return loss

        def callback_scipy(intermediate_result: OptimizeResult):
            if "nit" in intermediate_result:
                step = intermediate_result.nit
            else:
                step = self.contexts[0].n_evals

            self.tell(step)

        res = minimize(
            f_scipy, x0, method=method, bounds=bounds, **kwargs, callback=callback_scipy
        )

        if not res.success:
            logger.warning(f"Fit did not converge: {res.message}")

        opt_params = cast("ParametersT", unflatten_dict(dict(zip(self._keys, res.x))))

        return self.finish(opt_params)
