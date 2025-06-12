import logging
from typing import Callable
import numpy as np
from typing import Dict, Optional
import time

logger = logging.getLogger(__name__)


class Fitter:
    """
    Fits parameters by minimizing a weighted sum of individual contribution functions.

    The Fitter class allows users to define an objective function callback that computes
    individual contributions to a global objective based on an index and a parameter set.
    It then aggregates these contributions with optional weights, and offers an interface
    to optimize the parameters using SciPy.
    """

    def __init__(
        self,
        objective_function: Callable[[Dict[str, float]], float],
        initial_params: Dict[str, float],
        bounds: Optional[Dict[str, tuple[float, float]]] = None,
    ):
        """
        Initialize a Fitter instance.

        Parameters
        ----------
        objective_function : Callable[[Dict[str, float]], float]
            A callback function that, given a parameter dict,
            returns a float which is the value of the objective function to be minimized.
        """

        self.objective_function = objective_function
        self.initial_parameters = initial_params

        if bounds is None:
            self.bounds = {}
        else:
            self.bounds = bounds

        self._keys: list[str] = initial_params.keys()

    def hook_pre_fit(self):
        self.time_fit_start = time.time()

        logger.info("Start fitting")
        logger.info(f"    Initial parameters: {self.initial_parameters}")
        logger.info(f"    Bounds: {self.bounds}")
        ob_init = self.objective_function(self.initial_parameters)
        logger.info(f"    Initial obj func: {ob_init}")

    def hook_post_fit(self, opt_params: Dict):
        self.time_fit_end = time.time()

        logger.info("End fitting")
        logger.info(
            f"    Final objective function {self.objective_function(opt_params)}"
        )
        logger.info(f"    Optimal parameters {opt_params}")
        logger.info(f"    Time taken {self.time_fit_end - self.time_fit_start} seconds")

    def fit_nevergrad(self, budget: int, **kwargs) -> Dict:
        import nevergrad as ng

        self.hook_pre_fit()

        ng_params = ng.p.Dict()

        for k, v in self.initial_parameters.items():
            # If `k` is in bounds, fetch the lower and upper bound
            # It `k` is not in bounds just put lower=None and upper=None
            lower, upper = self.bounds.get(k, (None, None))
            ng_params[k] = ng.p.Scalar(v, lower=lower, upper=upper)

        instru = ng.p.Instrumentation(ng_params)

        optimizer = ng.optimizers.NgIohTuned(parametrization=instru, budget=budget)

        recommendation = optimizer.minimize(
            self.objective_function, **kwargs
        )  # best value
        args, kwargs = recommendation.value

        # Our optimal params are the first positional argument
        opt_params = args[0]

        self.hook_post_fit(opt_params)

        return opt_params

    def fit_scipy(self, **kwargs) -> Dict:
        """
        Optimize parameters using SciPy's minimize function.

        Parameters
        ----------
        initial_parameters : Dict[str, float]
            Initial guess for each parameter, as a mapping from name to value.
        **kwargs
            Additional keyword arguments passed directly to scipy.optimize.minimize.

        Returns
        -------
        Dict[str, float]
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

        from scipy.optimize import minimize

        self.hook_pre_fit()

        x0 = np.array([self.initial_parameters[k] for k in self._keys])

        bounds = np.array([self.bounds.get(k, (None, None)) for k in self._keys])

        if len(self.bounds) == 0:
            bounds = None

        # Scipy expects a function with n real-valued parameters f(x)
        # but our objective function takes a dictionary of parameters.
        # This is fine, we just define our objective function locally and
        # put a parameters in a dictionary based on the captured keys
        def f_scipy(x):
            p = dict(zip(self._keys, x))
            return self.objective_function(p)

        res = minimize(f_scipy, x0, bounds=bounds, **kwargs)

        if not res.success:
            logger.warning("Fit did not converge: %s", res.message)

        opt_params = dict(zip(self._keys, res.x))

        self.hook_post_fit(opt_params)

        return opt_params
