from scme_fitting.fitter import Fitter
from scme_fitting.combined_objective_function import CombinedObjectiveFunction

import numpy as np
import logging

logging.basicConfig(filename="test_fitter.log", level=logging.DEBUG)


def test_with_square_func():
    def cont1(params):
        return 2.0 * (params["x"] - 2) ** 2

    def cont2(params):
        return 3.0 * (params["y"] + 1) ** 2

    obj_func = CombinedObjectiveFunction([cont1, cont2])

    fitter = Fitter(objective_function=obj_func)

    initial_params = dict(x=0.0, y=0.0)
    optimal_params = fitter.fit_scipy(initial_parameters=initial_params)

    print(f"{optimal_params = }")

    assert np.isclose(optimal_params["x"], 2.0)
    assert np.isclose(optimal_params["y"], -1.0)

    optimal_params = fitter.fit_nevergrad(initial_parameters=initial_params, budget=100)
    print(optimal_params)


if __name__ == "__main__":
    test_with_square_func()
