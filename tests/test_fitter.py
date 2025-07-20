from scme_fitting.fitter import Fitter
from scme_fitting.combined_objective_function import CombinedObjectiveFunction

import numpy as np
import logging

from pydictnest import items_nested, has_nested, get_nested

logging.basicConfig(filename="test_fitter.log", level=logging.DEBUG)


def test_with_square_func():
    def cont1(params):
        return 2.0 * (params["x"] - 2) ** 2

    def cont2(params):
        return 3.0 * (params["y"] + 1) ** 2

    obj_func = CombinedObjectiveFunction([cont1, cont2])

    initial_params = dict(x=0.0, y=0.0)
    fitter = Fitter(objective_function=obj_func, initial_params=initial_params)

    optimal_params = fitter.fit_scipy()

    print(f"{optimal_params = }")
    assert np.isclose(optimal_params["x"], 2.0)
    assert np.isclose(optimal_params["y"], -1.0)
    assert np.isclose(obj_func(initial_params), fitter.info.initial_value)
    assert np.isclose(obj_func(optimal_params), fitter.info.final_value)

    optimal_params = fitter.fit_nevergrad(budget=100)

    print(f"{optimal_params = }")
    assert np.isclose(optimal_params["x"], 2.0, atol=1e-2)
    assert np.isclose(optimal_params["y"], -1.0, atol=1e-2)
    assert np.isclose(obj_func(initial_params), fitter.info.initial_value)
    assert np.isclose(obj_func(optimal_params), fitter.info.final_value)

    print(f"{fitter.info = }")


def test_with_square_func_bounds():
    def cont1(params):
        return 2.0 * (params["x"] - 2) ** 2

    def cont2(params):
        return 3.0 * (params["y"] + 1) ** 2

    obj_func = CombinedObjectiveFunction([cont1, cont2])

    initial_params = dict(x=0.0, y=0.0)
    bounds = dict(x=(0.0, 1.5))

    fitter = Fitter(
        objective_function=obj_func, initial_params=initial_params, bounds=bounds
    )

    optimal_params = fitter.fit_scipy()

    print(f"{optimal_params = }")
    assert np.isclose(optimal_params["x"], 1.5)
    assert np.isclose(optimal_params["y"], -1.0)
    assert np.isclose(obj_func(initial_params), fitter.info.initial_value)
    assert np.isclose(obj_func(optimal_params), fitter.info.final_value)

    optimal_params = fitter.fit_nevergrad(budget=100)

    print(optimal_params)
    assert np.isclose(optimal_params["x"], 1.5, atol=1e-2)
    assert np.isclose(optimal_params["y"], -1.0, atol=1e-2)
    assert np.isclose(obj_func(initial_params), fitter.info.initial_value)
    assert np.isclose(obj_func(optimal_params), fitter.info.final_value)

    print(f"{fitter.info = }")


def test_with_nested_dict():
    def cont1(params):
        return 2.0 * (params["params"]["x"] - 2) ** 2

    def cont2(params):
        return 3.0 * (params["y"] + 1) ** 2

    obj_func = CombinedObjectiveFunction([cont1, cont2])

    initial_params = {"params": dict(x=0.0), "y": 0.0}
    bounds = {"params": dict(x=(0.0, 1.5))}

    fitter = Fitter(
        objective_function=obj_func, initial_params=initial_params, bounds=bounds
    )

    optimal_params = fitter.fit_scipy()
    print(f"{optimal_params = }")
    assert np.isclose(optimal_params["params"]["x"], 1.5)
    assert np.isclose(optimal_params["y"], -1.0)
    assert np.isclose(obj_func(initial_params), fitter.info.initial_value)
    assert np.isclose(obj_func(optimal_params), fitter.info.final_value)

    optimal_params = fitter.fit_nevergrad(budget=100)

    print(optimal_params)
    assert np.isclose(optimal_params["params"]["x"], 1.5, atol=1e-2)
    assert np.isclose(optimal_params["y"], -1.0, atol=1e-2)
    assert np.isclose(obj_func(initial_params), fitter.info.initial_value)
    assert np.isclose(obj_func(optimal_params), fitter.info.final_value)

    print(f"{fitter.info = }")


def test_with_complicated_dict():
    def ob(params):
        res = 0
        for k, v in items_nested(params):
            res += v**2
        return res

    initial_params = {
        "electrostatic": {"bla": {"a": 1.0, "b": 1.0, "c": 1.0}, "foo": 1.0},
        "dispersion": 0.4,
        "params": {"a": 1.0, "b": 1.0},
    }

    bounds = {"dispersion": [0.2, 2.0], "electrostatic": {"bla": {"a": [0.5, 1.0]}}}

    # Every non-constrained parameter should be at 0.0
    # and every constrained parameter should be at the lower bound
    def check_solution(opt_params):
        for k, v in items_nested(opt_params):
            if has_nested(bounds, k):
                lower, upper = get_nested(bounds, k)
                print(k, v, lower)
                assert np.isclose(v, lower, atol=1e-2)
            else:
                print(k, v, 0.0)
                assert np.isclose(v, 0.0, atol=1e-2)

    fitter = Fitter(objective_function=ob, initial_params=initial_params, bounds=bounds)

    optimal_params = fitter.fit_scipy()
    print(f"{optimal_params = }")
    check_solution(optimal_params)
    assert np.isclose(ob(initial_params), fitter.info.initial_value)
    assert np.isclose(ob(optimal_params), fitter.info.final_value)

    optimal_params = fitter.fit_nevergrad(budget=500)
    print(f"{optimal_params = }")
    check_solution(optimal_params)
    assert np.isclose(ob(initial_params), fitter.info.initial_value)
    assert np.isclose(ob(optimal_params), fitter.info.final_value)

    print(f"{fitter.info = }")


if __name__ == "__main__":
    # test_with_square_func()
    # test_with_square_func_bounds()
    test_with_complicated_dict()
