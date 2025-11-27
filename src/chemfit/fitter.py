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
        """Initialize the FitterEvaluateContext."""
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
        """Initialize the FitterObjectiveFunctor."""
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
        Initialize a Fitter.

        Args:
            objective_function (Callable[[dict], float]):
                The objective function to be minimized.
            initial_params (dict):
                Initial values of the parameters.
            bound (Optional[dict]):
                Dictionary specifying bounds for each parameter.
            near_bound_tol (Optional[float]):
                If specified, checks whether any parameters are too close to their bounds and logs a warning if so.
            value_bad_params (float):
                Threshold value beyond which the objective function is considered to be in a poor or invalid region.

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
        Register a callback which is executed after every `n_steps` of the optimization.

        Multiple callbacks may be registered. They are executed in the order of registration.
        The callback must be a callable with the following signature:

            func(arg: CallbackInfo)

        The `CallbackInfo` is a dataclass with the following attributes:
            - `opt_params`: The optimal parameters at the time the callback is invoked.
            - `opt_loss`: The loss value corresponding to the optimal parameters.
            - `cur_params`: The parameters tested most recently when the callback is invoked.
            - `cur_loss`: The loss value associated with the most recently tested parameters.
            - `step`: The number of optimization steps performed so far
                    (generally not equal to the number of loss function evaluations).
            - `info`: The current `FitInfo` instance of the fitter at the time the callback is invoked.
        """
        self.callbacks.append((func, n_steps))

    def unify_callbacks(
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

    def hook_pre_fit(self):
        """A hook, which is invoked before optimizing."""
        logger.info("Start fitting")
        self.time_fit_start = time.time()

    def hook_post_fit(self, opt_params: dict[str, Any]):
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
        self.hook_pre_fit()

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

        callback, n_steps = self.unify_callbacks()

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

        self.hook_post_fit(opt_params)

        return opt_params

    def fit_scipy(self, method: str = "L-BFGS-B", **kwargs) -> dict[str, Any]:
        """
        Optimize parameters using SciPy's minimize function.

        Parameters
        ----------
        initial_parameters : dict
            Initial guess for each parameter, as a mapping from name to value.
        **kwargs
            Additional keyword arguments passed directly to scipy.optimize.minimize.

        Returns
        -------
        dict
            Dictionary of optimized parameter values.

        Warnings
        --------
        If the optimizer does not converge, a warning is logged.

        Example
        -------
        >>> def objective_function(idx: int, params: dict):
        ...     return 2.0 * (params["x"] - 2) ** 2 + 3.0 * (params["y"] + 1) ** 2
        >>> fitter = Fitter(objective_function=objective_function)
        >>> initial_params = dict(x=0.0, y=0.0)
        >>> optimal_params = fitter.fit_scipy(initial_parameters=initial_params)
        >>> print(optimal_params)
        {'x': 2.0, 'y': -1.0}

        """

        self.hook_pre_fit()

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
        callback, n_steps = self.unify_callbacks()

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

        self.hook_post_fit(opt_params)

        return opt_params
