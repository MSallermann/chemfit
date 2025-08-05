import logging
import numpy as np
from typing import Optional, Callable, Any
import time

from numbers import Real
from functools import wraps

from chemfit.exceptions import FactoryException

from dataclasses import dataclass

import math

from pydictnest import (
    flatten_dict,
    unflatten_dict,
)

logger = logging.getLogger(__name__)


@dataclass
class FitInfo:
    initial_value: Optional[float] = None
    final_value: Optional[float] = None
    time_taken: Optional[float] = None
    n_evals: int = 0


@dataclass
class CallbackInfo:
    opt_params: dict
    opt_loss: float
    cur_params: dict
    cur_loss: float
    step: int
    info: FitInfo


class Fitter:
    """
    Fits parameters by minimizing an objective function.
    """

    def __init__(
        self,
        objective_function: Callable[[dict], float],
        initial_params: dict,
        bounds: Optional[dict] = None,
        near_bound_tol: Optional[float] = None,
        value_bad_params: float = 1e5,
    ):
        """
        Args:
           objective_function (Callable[[dict], float]):
               The objective function to be minimized.
            initial_params (dict):
                Initial values of the parameters
            bound (Optional[dict]):
                Dictionary of parameter bounds
            near_bound_tol(Optional[float]):
                If specified, performs a check to see if any of the parameters
                is too close to the bounds and logs a warning if so
            value_bad_params (float):
                A value beyond which the objective function is considered to be in a bad region
        """

        self.objective_function = self.ob_func_wrapper(objective_function)

        self.initial_parameters = initial_params

        if bounds is None:
            self.bounds = {}
        else:
            self.bounds = bounds

        self.value_bad_params = value_bad_params

        self.near_bound_tol = near_bound_tol

        self.info = FitInfo()

        self.callbacks: list[tuple[Callable[[dict, float, int, FitInfo]], int]] = []

    def register_callback(self, func: Callable[[CallbackInfo], None], n_steps: int):
        """
        Register a callback which is run after every `n_steps` of the optimization.

        Multiple callbacks may be registered. They are executed in order of registration.

        The callback needs to be callable with
            func( arg : CallbackInfo )

        The `CallbackInfo` is a dataclass with the following members

            - `opt_params` are the optimal parameters at the time the callback is invoked
            - `opt_loss` is the loss with those optimal parameters
            - `cur_params` are the parameters which have been tested lasts at the time the callback is invoked.
            - `cur_loss` are the parameters which have been tested lasts at the time the callback is invoked.
            - `step` is the number of optimization steps performed thus far
                    (in general not equal to the number of evaluations of the loss function)
            - `info` is the current `FitInfo` of the fitter at the time the callback is invoked
        """
        self.callbacks.append((func, n_steps))

    def ob_func_wrapper(self, ob_func: Any) -> float:
        """Wraps the objective function and applies some checks plus logging"""

        @wraps(ob_func)
        def wrapped_ob_func(params: dict):
            # first we try if we can get a value at all
            try:
                value = ob_func(params)
                self.info.n_evals += 1
            except FactoryException as e:
                # If we catch a factory exception we should just crash the code, therefore we re-raise
                logger.exception(
                    "Caught factory exception while evaluating objective function.",
                    stack_info=True,
                    stacklevel=2,
                )
                raise e
            except Exception:
                # On a general exception we continue execution, since it might just be a bad parameter region
                logger.debug(
                    f"Caught exception with {params = }. Clipping loss to {self.value_bad_params}",
                    exc_info=True,
                    stack_info=True,
                    stacklevel=2,
                )
                value = self.value_bad_params

            # then we make sure that the value is a float
            if not isinstance(value, Real):
                logger.debug(
                    f"Objective function did not return a single float, but returned `{value}` with type {type(value)}. Clipping loss to {self.value_bad_params}"
                )
                value = self.value_bad_params

            if math.isnan(value):
                logger.debug(
                    f"Objective function returned NaN. Clipping loss to {self.value_bad_params}"
                )
                value = self.value_bad_params

            return value

        return wrapped_ob_func

    def _produce_callback(self) -> tuple[Optional[Callable[[CallbackInfo], None]], int]:
        """Generate a single callback from the list of callbacks"""

        if len(self.callbacks) == 0:
            return None, float("inf")

        min_n_steps = min([n_steps for (_, n_steps) in self.callbacks])

        def callback(callback_args: CallbackInfo):
            for cb, n_steps in self.callbacks:
                if callback_args.step % n_steps == 0:
                    cb(callback_args)

        return callback, min_n_steps

    def hook_pre_fit(self):
        """A hook, which is invoked before optimizing"""

        # Overwrite with a fresh FitInfo object
        self.info = FitInfo()

        logger.info("Start fitting")

        self.info.initial_value = self.objective_function(self.initial_parameters)
        logger.info(f"    Initial obj func: {self.info.initial_value}")

        if self.info.initial_value == self.value_bad_params:
            logger.warning(
                f"Starting optimization in a `bad` region. Objective function could not be evaluated properly. Loss has been set to {self.value_bad_params = }"
            )
        elif self.info.initial_value > self.value_bad_params:
            new_value_bad_params = 1.1 * self.info.initial_value
            logger.warning(
                f"Starting optimization in a high loss region. Loss is {self.info.initial_value}, which is greater than {self.value_bad_params = }. Adjusting to {new_value_bad_params = }."
            )
            self.value_bad_params = new_value_bad_params

        self.info.n_evals = 0
        self.time_fit_start = time.time()

    def hook_post_fit(self, opt_params: dict):
        """A hook, which is invoked after optimizing"""

        self.time_fit_end = time.time()
        self.info.time_taken = self.time_fit_end - self.time_fit_start

        if self.info.final_value >= self.value_bad_params:
            logger.warning(
                f"Ending optimization in a `bad` region. Loss is greater or equal to {self.value_bad_params = }"
            )

        logger.info("End fitting")
        logger.info(f"    Info {self.info}")

        if self.near_bound_tol is not None:
            self.problematic_params = self.check_params_near_bounds(
                opt_params, self.near_bound_tol
            )

            if len(self.problematic_params) > 0:
                logger.warning(
                    f"The following parameters are within {self.near_bound_tol * 100:.1f}% of the bounds."
                    "You *may* have to loosen the bounds for an optimal result."
                )
                for kp, vp, lower, upper in self.problematic_params:
                    logger.warning(
                        f"    parameter = {kp}, lower = {lower}, upper = {upper}, value = {vp}"
                    )

    def check_params_near_bounds(self, params, relative_tol: float) -> list:
        """Check if any of the parameters are near the bounds"""

        flat_params = flatten_dict(params)
        flat_bounds = flatten_dict(self.bounds)

        problematic_params = []

        for (kp, vp), (kb, (lower, upper)) in zip(
            flat_params.items(), flat_bounds.items()
        ):
            abs_tol = relative_tol * np.abs(vp)

            if vp - lower < abs_tol or upper - vp < abs_tol:
                problematic_params.append([kp, vp, lower, upper])

        return problematic_params

    def fit_nevergrad(
        self, budget: int, optimizer_str: str = "NgIohTuned", **kwargs
    ) -> dict:
        import nevergrad as ng

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

        def f_ng(p):
            params = unflatten_dict(p)
            return self.objective_function(params)

        callback, n_steps = self._produce_callback()

        opt_loss = self.info.initial_value

        for i in range(budget):
            if i == 0:
                flat_params = flat_initial_params
                cur_loss = self.info.initial_value
                p = optimizer.parametrization.spawn_child()
                p.value = (
                    (flat_params,),
                    {},
                )
                optimizer.tell(p, self.info.initial_value)
            else:
                p = optimizer.ask()
                args, kwargs = p.value

                flat_params = args[0]
                cur_loss = f_ng(flat_params)

                optimizer.tell(p, cur_loss)

            if cur_loss < opt_loss:
                opt_loss = cur_loss

            if callback is not None and i % n_steps == 0:
                recommendation = optimizer.provide_recommendation()
                args, kwargs = recommendation.value
                flat_opt_params = args[0]

                opt_params = unflatten_dict(flat_opt_params)
                cur_params = unflatten_dict(flat_params)

                callback(
                    CallbackInfo(
                        opt_params=opt_params,
                        opt_loss=opt_loss,
                        cur_params=cur_params,
                        cur_loss=cur_loss,
                        step=i,
                        info=self.info,
                    )
                )

        recommendation = optimizer.provide_recommendation()
        args, kwargs = recommendation.value

        # Our optimal params are the first positional argument
        flat_opt_params = args[0]

        # loss is an optional field in the recommendation so we have to test if it has been written
        if recommendation.loss is not None:
            self.info.final_value = recommendation.loss
        else:  # otherwise we compute the optimal loss
            self.info.final_value = self.objective_function(flat_opt_params)

        opt_params = unflatten_dict(flat_opt_params)

        self.hook_post_fit(opt_params)

        return opt_params

    def fit_scipy(self, method: str = "L-BFGS-B", **kwargs) -> dict:
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

        from scipy.optimize import minimize, OptimizeResult

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
        bounds = np.array([flat_bounds.get(k, (None, None)) for k in self._keys])

        if len(bounds) == 0:
            bounds = None

        # The local objective function first creates a flat dictionary from the `x` array
        # by zipping it with the captured flattened keys and then unflattens the dictionary
        # to pass it to the objective functions
        def f_scipy(x):
            p = unflatten_dict(dict(zip(self._keys, x)))
            return self.objective_function(p)

        # Then we need to handle some awkwardness:
        #   1. Scipy does not mandate all of the optimizers
        #      to write all the values we need for our callback system.
        #      Therefore, we need to roll our own bookkeeping logic for the
        #      number of steps taken.
        #   2. Scipy mandates a different function signature, so we have to "translate"
        # We do this in the following functor:
        class CallbackScipy:
            def __init__(
                self,
                keys: list[str],
                info: FitInfo,
                callback: Callable[[CallbackInfo], None],
                n_steps: int,
            ):
                self._step = 0
                self._keys = keys
                self._info = info
                self._callback = callback
                self._n_steps = n_steps

            def __call__(self, intermediate_result: OptimizeResult):
                # This callback is executed after *every* iteration

                # We may have to track the step ourselves
                self._step += 1

                # If we are given "nit", we use it instead
                if "nit" in intermediate_result.keys():
                    self._step = intermediate_result.nit

                if self._step % self._n_steps == 0:
                    x = intermediate_result.x

                    cur_params = unflatten_dict(dict(zip(self._keys, x)))
                    cur_loss = intermediate_result.fun

                    # We assume (can be wrong though)
                    opt_params = cur_params
                    opt_loss = cur_loss

                    self._callback(
                        CallbackInfo(
                            opt_params=opt_params,
                            opt_loss=opt_loss,
                            cur_params=cur_params,
                            cur_loss=cur_loss,
                            step=self._step,
                            info=self._info,
                        )
                    )

        # First concatenate the list of callbacks into a single function
        callback, n_steps = self._produce_callback()

        # Then, we wrap it in a way that scipy understands
        if callback is not None:
            callback_scipy = CallbackScipy(
                keys=self._keys, info=self.info, callback=callback, n_steps=n_steps
            )
        else:
            callback_scipy = None

        # ob = partial(self.ob_func_wrapper, ob_func=f_scipy)
        res = minimize(
            f_scipy, x0, method=method, bounds=bounds, **kwargs, callback=callback_scipy
        )

        if not res.success:
            logger.warning(f"Fit did not converge: {res.message}")

        self.info.final_value = res.fun
        opt_params = dict(zip(self._keys, res.x))

        opt_params = unflatten_dict(opt_params)

        self.hook_post_fit(opt_params)

        return opt_params
