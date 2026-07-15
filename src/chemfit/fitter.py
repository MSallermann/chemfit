from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable, cast

import nevergrad as ng
import numpy as np
import numpy.typing as npt
from pydictnest import (
    flatten_dict,
    unflatten_dict,
)
from scipy.optimize import OptimizeResult, minimize

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

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["n_evals"] = self.n_evals
        state["opt_loss"] = self.opt_loss
        state["opt_params"] = self.opt_params
        state["opt_meta"] = self.opt_meta
        return state

    def __setstate__(self, state: dict[str, Any]):
        super().__setstate__(state)
        self.n_evals = state["n_evals"]
        self.opt_loss = state["opt_loss"]
        self.opt_params = state["opt_params"]
        self.opt_meta = state["opt_meta"]


class FitterObjectiveFunctor(ObjectiveFunctor):
    def __init__(
        self,
        wrap_me: ObjectiveFunctor,
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
        parameters: dict[str, Any],
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
            ctx.opt_params = dict(parameters)
            ctx.opt_meta = dict(ctx.meta)

        return loss

    def __call__(  # type: ignore
        self, parameters: dict[str, Any], ctx: FitterEvaluateContext | None = None
    ) -> float:
        if ctx is None:
            ctx = FitterEvaluateContext()

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


class Fitter:
    def __init__(
        self,
        objective_function: Callable[[dict[str, Any]], float] | ObjectiveFunctor,
        initial_params: dict[str, Any],
        bounds: dict[str, Any] | None = None,
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
            initial_params (dict[str, Any]): Initial parameter values.
            bounds (dict[str, Any] | None, optional): Bounds for each
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

        self.initial_parameters = initial_params

        if bounds is None:
            self.bounds = {}
        else:
            self.bounds = bounds

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

    def _hook_post_fit(self, opt_params: dict[str, Any]):
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

        self._user_owns_executor = num_workers != 1 and executor is None
        if self._user_owns_executor:
            executor = ThreadPoolExecutor(num_workers)

        self._hook_pre_fit()
        self._user_executor = executor
        self._user_num_workers = num_workers
        self._user_step = 0
        self.contexts = (
            [FitterEvaluateContext() for _ in range(num_workers)]
            if contexts is None
            else contexts
        )

    def ask(
        self,
        parameters: dict[str, Any] | list[dict[str, Any]],
        context_index: int = 0,
    ) -> float | list[float]:
        """
        Evaluate one candidate or a parallel batch proposed by the user.

        A dictionary produces one loss. A list produces a list of losses in
        input order and may contain at most ``num_workers`` candidates.
        """

        if not hasattr(self, "_user_num_workers"):
            msg = "call fitter.init() before fitter.ask()"
            raise RuntimeError(msg)

        if isinstance(parameters, dict):
            return self.objective_function(parameters, self.contexts[context_index])

        if len(parameters) > self._user_num_workers:
            msg = "a batch cannot contain more candidates than workers"
            raise ValueError(msg)
        if len(parameters) == 0:
            return []
        if len(parameters) == 1:
            return [self.objective_function(parameters[0], self.contexts[0])]

        if self._user_executor is None:
            msg = "parallel evaluation requires an executor"
            raise RuntimeError(msg)
        return map_with_context(
            self._user_executor,
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

        if not hasattr(self, "_user_step"):
            msg = "call fitter.init() before fitter.tell()"
            raise RuntimeError(msg)
        current_step = self._user_step if step is None else step
        callback, n_steps = self._unify_callbacks()
        if callback is not None and current_step % n_steps == 0:
            callback(current_step, self.contexts)
        self._user_step = current_step + 1

    def finish(self, opt_params: dict[str, Any] | None = None) -> dict[str, Any]:
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
            opt_params = dict(best_context.opt_params)
        self._hook_post_fit(opt_params)
        if self._user_owns_executor:
            cast("ThreadPoolExecutor", self._user_executor).shutdown()
        return opt_params

    def fit_nevergrad(
        self,
        budget: int,
        optimizer_str: str = "NgIohTuned",
        num_workers: int = 1,
        contexts: list[FitterEvaluateContext] | None = None,
        executor: ExecutorLike | None = None,
        initial_observations: Iterable[tuple[dict[str, Any], float | None]]
        | None = None,
    ) -> dict[str, Any]:
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

        flat_bounds = flatten_dict(self.bounds)
        flat_initial_params = flatten_dict(self.initial_parameters)

        ng_params = ng.p.Dict()
        for k, v in flat_initial_params.items():
            # If `k` is in bounds, fetch the lower and upper bound
            # It `k` is not in bounds just put lower=None and upper=None
            lower, upper = flat_bounds.get(k, (None, None))
            ng_params[k] = ng.p.Scalar(init=v, lower=lower, upper=upper)
        instru = ng.p.Instrumentation(ng_params)

        try:
            OptimizerCls = ng.optimizers.registry[optimizer_str]
        except KeyError as e:
            available_solvers = list(ng.optimizers.registry.keys())
            msg = (
                f"Unknown nevergrad optimizer {optimizer_str!r}. "
                f"Available solvers: {available_solvers}"
            )
            raise KeyError(msg) from e

        optimizer = OptimizerCls(
            parametrization=instru, budget=budget, num_workers=num_workers
        )

        self.init(num_workers=num_workers, contexts=contexts, executor=executor)

        # This applies restart parameters, **if** they are within the bounds
        if initial_observations is not None:
            for restart_params, restart_loss_value in initial_observations:
                skip = False

                flat_params = flatten_dict(restart_params)

                for k, (lower, upper) in flat_bounds.items():
                    rp = flat_params.get(k, None)
                    if rp is not None and (rp < lower or rp > upper):
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

            # On the first evaluation we ensure that the optimizer suggests the initial params
            if step == 0:
                optimizer.suggest(flat_initial_params)

            # The final batch may be smaller than num_workers when the budget
            # is not evenly divisible by the worker count.
            asked_params = [optimizer.ask() for _ in range(batch_size)]
            flat_params = [p.value[0][0] for p in asked_params]
            nested_params = [
                unflatten_dict(params, dict_factory=dict[str, Any])
                for params in flat_params
            ]
            asked_losses = self.ask(nested_params)
            assert isinstance(asked_losses, list)
            losses = asked_losses

            [
                optimizer.tell(params, loss)
                for params, loss in zip(asked_params, losses, strict=True)
            ]

            self.tell(step)

        recommendation = optimizer.provide_recommendation()

        args, _ = recommendation.value
        # Our optimal params are the first positional argument
        flat_opt_params = args[0]

        opt_params = unflatten_dict(flat_opt_params, dict_factory=dict[str, Any])

        return self.finish(opt_params)

    def fit_scipy(
        self,
        method: str = "L-BFGS-B",
        ctx: FitterEvaluateContext | None = None,
        **kwargs,
    ) -> dict[str, Any]:
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

        # Scipy expects a function with n real-valued parameters f(x)
        # but our objective function takes a dictionary of parameters.
        # Moreover, the dictionary might not be flat but nested.
        # Therefore, as a first step, we flatten the bounds and
        # initial parameter dicts
        flat_params = flatten_dict(self.initial_parameters)
        flat_bounds = flatten_dict(self.bounds)

        # We then capture the order of keys in the flattened dictionary
        self._keys = flat_params.keys()

        # The initial value of x and of the bounds are derived from that order
        x0 = np.array([flat_params[k] for k in self._keys])

        if len(flat_bounds) == 0:
            bounds = None
        else:
            bounds = np.array([flat_bounds.get(k, (None, None)) for k in self._keys])

        # Since we know that scipy.optimize works synchronously, we create a single context, which we'll keep alive.
        self.init(contexts=None if ctx is None else [ctx])

        # The local objective function first creates a flat dictionary from the `x` array
        # by zipping it with the captured flattened keys and then unflattens the dictionary
        # to pass it to the objective functions
        def f_scipy(x: npt.NDArray) -> float:
            p = unflatten_dict(dict(zip(self._keys, x)), dict_factory=dict[str, Any])
            cast("dict[str, Any]", p)
            loss = self.ask(p)
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

        opt_params = dict(zip(self._keys, res.x))

        opt_params = unflatten_dict(opt_params)

        return self.finish(opt_params)
