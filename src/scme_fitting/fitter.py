import logging
from typing import Callable
import numpy as np
from typing import Dict
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
        self._keys: list[str] = []

    def hook_pre_fit(self, initial_parameters: Dict):
        self.time_fit_start = time.time()

        logger.info(f"Start fitting with initial parameters {initial_parameters}")
        logger.info(
            f"Initial objective function {self.objective_function(initial_parameters)}"
        )

    def hook_post_fit(self, opt_params: Dict):
        self.time_fit_end = time.time()

        logger.info(f"Final objective function {self.objective_function(opt_params)}")
        logger.info(f"Optimal parameters {opt_params}")
        logger.info(f"Time taken {self.time_fit_end - self.time_fit_start} seconds")

    def fit_nevergrad(self, initial_parameters: Dict, budget: int, **kwargs) -> Dict:
        """
        Optimize parameters using Nevergrad`s NgIohTuned function.

        Parameters
        ----------
        initial_parameters : Dict[str, float]
            Initial guess for each parameter, as a mapping from name to value.
        budget : int
            The budget (number of function evaluations)
        **kwargs
            Additional keyword arguments passed directly to ` ng.optimizers.NgIohTuned.minimize`.

        Returns
        -------
        Dict[str, float]
            Dictionary of optimized parameter values.

        Example
        -------
        >>> def objective_function(idx: int, params: dict):
        ...     return 2.0 * (params["x"] - 2) ** 2 + 3.0 * (params["y"] + 1) ** 2
        >>> fitter = Fitter(objective_function=objective_function)
        >>> initial_params = dict(x=0.0, y=0.0)
        >>> optimal_params = fitter.fit_nevergrad(initial_parameters=initial_params, budget=100)
        >>> print(optimal_params)
        {'x': 2.0, 'y': -1.0}
        """
        import nevergrad as ng

        self.hook_pre_fit(initial_parameters)

        ng_params = ng.p.Dict(
            **{k: ng.p.Scalar(v) for k, v in initial_parameters.items()}
        )

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

    def fit_scipy(self, initial_parameters: Dict[str, float], **kwargs) -> Dict:
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

        self.hook_pre_fit(initial_parameters)

        # capture key order once
        self._keys = list(initial_parameters.keys())
        x0 = np.array([initial_parameters[k] for k in self._keys])

        # Scipy expects a function with n real-valued parameters f(x)
        # but our objective function takes a dictionary of parameters.
        # This is fine, we just define our objective function locally and
        # put a parameters in a dictionary based on the captured keys
        def f_scipy(x):
            p = dict(zip(self._keys, x))
            return self.objective_function(p)

        res = minimize(f_scipy, x0, **kwargs)

        if not res.success:
            logger.warning("Fit did not converge: %s", res.message)

        opt_params = dict(zip(self._keys, res.x))

        self.hook_post_fit(opt_params)

        return opt_params
