from __future__ import annotations

import asyncio
import logging
import math
import time
from numbers import Real
from typing import Any, Callable, cast

import nevergrad as ng
import numpy as np
import numpy.typing as npt
from pydictnest import (
    flatten_dict,
    unflatten_dict,
)
from scipy.optimize import OptimizeResult, minimize

from chemfit.abstract_objective_function import EvaluateContext, ObjectiveFunctor
from chemfit.async_helpers import async_eval_many
from chemfit.utils import check_params_near_bounds
from chemfit.wrap_funcs import to_objective_functor

logger = logging.getLogger(__name__)


class FitterEvaluateContext(EvaluateContext):
    def __init__(self):
        """
        Extended evaluation context used by the fitter.

        This context tracks additional information useful during
        optimization.
        """

        super().__init__()
        self.n_evals: int = 0
        self.opt_loss: float | None = None
        self.opt_params: dict[str, Any] | None = None


class FitterObjectiveFunctor(ObjectiveFunctor):
    def __init__(
        self,
        wrap_me: ObjectiveFunctor,
        swallow_exceptions: bool = False,
        log_exceptions: bool = True,
    ):
        """
        ObjectiveFunctor wrapper with robust error handling and tracking.

        This wrapper sits between the raw objective and the optimizer. It:

        * Catches exceptions from the wrapped objective.
        * Optionally logs and/or re-raises them.
        * Clips non-float and NaN results to a large "bad" value.
        * Updates the attached `FitterEvaluateContext` with evaluation
          counts and the best loss/parameters seen so far.

        Args:
            wrap_me (ObjectiveFunctor): The underlying objective functor
                to be evaluated.
            swallow_exceptions (bool, optional): If True, exceptions
                raised by the wrapped objective are converted to NaN and
                then clipped to ``value_bad_params`` instead of being
                re-raised. Defaults to False.
            log_exceptions (bool, optional): If True, exceptions are
                logged using the module logger. Defaults to True.

        """
        self.wrap_me = wrap_me
        self.value_bad_params = 1e5
        self.swallow_exceptions: bool = swallow_exceptions
        self.log_exceptions: bool = log_exceptions

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
            ctx.opt_params = parameters

        return loss


class Fitter:
    def __init__(
        self,
        objective_function: Callable[[dict[str, Any]], float] | ObjectiveFunctor,
        initial_params: dict[str, Any],
        bounds: dict[str, Any] | None = None,
        near_bound_tol: float | None = None,
        value_bad_params: float = 1e5,
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
            objective_function = to_objective_functor(objective_function)

        self.objective_function = FitterObjectiveFunctor(objective_function)

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
        """A hook, which is invoked before optimizing."""
        logger.info("Start fitting")
        self.time_fit_start = time.time()

    def _hook_post_fit(self, opt_params: dict[str, Any]):
        """A hook, which is invoked after optimizing."""
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

    def fit_nevergrad(
        self,
        budget: int,
        optimizer_str: str = "NgIohTuned",
        num_workers: int = 1,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Optimize parameters using a nevergrad optimizer.

        This method drives nevergrad's ask/tell interface and optionally
        evaluates multiple points in parallel using `async_eval_many`.

        Args:
            budget (int): Total number of objective evaluations to allow.
            optimizer_str (str, optional): Name of the nevergrad optimizer
                to use (key in ``ng.optimizers.registry``). Defaults to
                ``"NgIohTuned"``.
            num_workers (int, optional): Number of points to evaluate in
                parallel per step. If greater than 1, evaluations are
                performed via `asyncio.run(async_eval_many(...))`. Defaults
                to 1.
            **kwargs: Additional keyword arguments forwarded to the
                nevergrad optimizer constructor.

        Returns:
            dict[str, Any]: Dictionary of optimized parameter values.

        Warning:
            When ``num_workers > 1``, this method uses ``asyncio.run`` to
            perform parallel evaluations. It therefore cannot be safely
            called from within an already running event loop (e.g. inside
            ``asyncio`` tasks or some notebook environments). In such
            cases, you should either run the fit in a separate process or
            implement your own async driver around `async_eval_many`.

        """

        self._hook_pre_fit()

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
            e.add_note(f"Available solvers: {list(ng.optimizers.registry.keys())}")
            raise e

        optimizer = OptimizerCls(parametrization=instru, budget=budget)

        def f_ng(parameters: dict[str, Any], ctx: FitterEvaluateContext) -> float:
            params = unflatten_dict(parameters, dict_factory=dict)
            return self.objective_function(params, ctx)

        callback, n_steps = self._unify_callbacks()

        # We need one context per worker
        self.contexts = [FitterEvaluateContext() for _ in range(num_workers)]

        for step in range(budget // num_workers):
            # On the first evaluation we ensure that the optimizer suggests the initial params
            if step == 0:
                optimizer.suggest(flat_initial_params)

            # Ask for num_workers parameters to evaluate in parallel
            asked_params = [optimizer.ask() for _ in range(num_workers)]
            flat_params = [p.value[0][0] for p in asked_params]

            if num_workers == 1:
                losses = [f_ng(flat_params[0], self.contexts[0])]
            else:
                losses = asyncio.run(async_eval_many(f_ng, flat_params, self.contexts))  # pyright: ignore[reportArgumentType]

            [
                optimizer.tell(params, loss)
                for params, loss in zip(asked_params, losses, strict=True)
            ]

            if callback is not None and step % n_steps == 0:
                callback(budget, self.contexts)

        recommendation = optimizer.provide_recommendation()
        args, kwargs = recommendation.value

        # Our optimal params are the first positional argument
        flat_opt_params = args[0]

        opt_params = unflatten_dict(flat_opt_params, dict_factory=dict[str, Any])

        self._hook_post_fit(opt_params)

        return opt_params

    def fit_scipy(self, method: str = "L-BFGS-B", **kwargs) -> dict[str, Any]:
        """
        Optimize parameters using SciPy's ``minimize`` function.

        Args:
            method (str, optional): Optimization method passed to
                ``scipy.optimize.minimize``. Defaults to ``"L-BFGS-B"``.
            **kwargs: Additional keyword arguments forwarded to
                ``scipy.optimize.minimize``.

        Returns:
            dict[str, Any]: Dictionary of optimized parameter values
            (unflattened).

        Warning:
            If the optimizer does not converge, a warning is logged.

        """

        self._hook_pre_fit()

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
        self.contexts = [FitterEvaluateContext()]

        # The local objective function first creates a flat dictionary from the `x` array
        # by zipping it with the captured flattened keys and then unflattens the dictionary
        # to pass it to the objective functions
        def f_scipy(x: npt.NDArray) -> float:
            p = unflatten_dict(dict(zip(self._keys, x)), dict_factory=dict[str, Any])
            cast("dict[str, Any]", p)
            assert self.contexts is not None
            return self.objective_function(p, ctx=self.contexts[0])

        # First concatenate the list of callbacks into a single function
        callback, n_steps = self._unify_callbacks()

        def callback_scipy(intermediate_result: OptimizeResult):
            if "nit" in intermediate_result:
                step = intermediate_result.nit
            else:
                step = self.contexts[0].n_evals

            if callback is not None and step % n_steps == 0:
                callback(step, self.contexts)

        res = minimize(
            f_scipy, x0, method=method, bounds=bounds, **kwargs, callback=callback_scipy
        )

        if not res.success:
            logger.warning(f"Fit did not converge: {res.message}")

        opt_params = dict(zip(self._keys, res.x))

        opt_params = unflatten_dict(opt_params)

        self._hook_post_fit(opt_params)

        return opt_params
