try:
    import mpi4py
except ImportError:
    mpi4py = None

import pytest

from chemfit.fitter import Fitter
from chemfit.combined_objective_function import CombinedObjectiveFunction

import numpy as np
import logging

from pydictnest import items_nested, has_nested, get_nested

NG_SOLVERS = ["CMA", "NgIohTuned", "Carola3"]
NG_ATOL = 5e-2


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

    for opt in NG_SOLVERS:
        optimal_params = fitter.fit_nevergrad(budget=2000, optimizer_str=opt)
        print(f"{opt = }")
        print(f"{optimal_params = }")
        print(f"{fitter.info = }")

        assert np.isclose(optimal_params["x"], 2.0, atol=NG_ATOL)
        assert np.isclose(optimal_params["y"], -1.0, atol=NG_ATOL)
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
    print(f"{fitter.info = }")
    assert np.isclose(optimal_params["x"], 1.5)
    assert np.isclose(optimal_params["y"], -1.0)
    assert np.isclose(obj_func(initial_params), fitter.info.initial_value)
    assert np.isclose(obj_func(optimal_params), fitter.info.final_value)

    for opt in NG_SOLVERS:
        optimal_params = fitter.fit_nevergrad(budget=2000, optimizer_str=opt)
        print(f"{opt = }")
        print(f"{optimal_params = }")
        print(f"{fitter.info = }")

        assert np.isclose(optimal_params["x"], 1.5, atol=NG_ATOL)
        assert np.isclose(optimal_params["y"], -1.0, atol=NG_ATOL)
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
    print(f"{fitter.info = }")
    assert np.isclose(optimal_params["params"]["x"], 1.5)
    assert np.isclose(optimal_params["y"], -1.0)
    assert np.isclose(obj_func(initial_params), fitter.info.initial_value)
    assert np.isclose(obj_func(optimal_params), fitter.info.final_value)

    optimal_params = fitter.fit_nevergrad(budget=2000)

    print(f"{optimal_params = }")
    print(f"{fitter.info = }")
    assert np.isclose(optimal_params["params"]["x"], 1.5, atol=NG_ATOL)
    assert np.isclose(optimal_params["y"], -1.0, atol=NG_ATOL)
    assert np.isclose(obj_func(initial_params), fitter.info.initial_value)
    assert np.isclose(obj_func(optimal_params), fitter.info.final_value)


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
                assert np.isclose(v, lower, atol=NG_ATOL)
            else:
                print(k, v, 0.0)
                assert np.isclose(v, 0.0, atol=NG_ATOL)

    fitter = Fitter(objective_function=ob, initial_params=initial_params, bounds=bounds)

    optimal_params = fitter.fit_scipy()
    print(f"{optimal_params = }")
    print(f"{fitter.info = }")
    check_solution(optimal_params)
    assert np.isclose(ob(initial_params), fitter.info.initial_value)
    assert np.isclose(ob(optimal_params), fitter.info.final_value)

    optimal_params = fitter.fit_nevergrad(budget=2000)
    print(f"{optimal_params = }")
    print(f"{fitter.info = }")
    check_solution(optimal_params)
    assert np.isclose(ob(initial_params), fitter.info.initial_value)
    assert np.isclose(ob(optimal_params), fitter.info.final_value)

    print(f"{fitter.info = }")


def test_with_bad_function():

    X_EXPECTED = 2.5

    def ob(params):
        if params["x"] < 1.0:
            return None
        elif params["x"] < 2.0:
            raise Exception("Some random exception")
        elif params["x"] < 3.0:
            return (params["x"] - 2.5) ** 2
        elif params["x"] < 4.0:
            return float("Nan")
        else:
            return "not even a number"

    for x0 in [0.5, 1.5, 2.5, 3.5, 4.5]:
        print(f"{x0 = }")

        fitter = Fitter(
            objective_function=ob, initial_params={"x": x0}, bounds={"x": (0.0, 5.0)}
        )

        # Nevergrad should be able to handle a shitty objective function like this
        optimal_params = fitter.fit_nevergrad(budget=10000, optimizer_str="OnePlusOne")
        print("NEVERGRAD")
        print(f"{optimal_params = }")
        print(f"{fitter.info = }")
        assert np.isclose(optimal_params["x"], X_EXPECTED, atol=NG_ATOL)

        # SCIPY will probably fail, unless starting in the good region
        optimal_params = fitter.fit_scipy()
        print("SCIPY")
        print(f"{optimal_params = }")
        print(fitter.info)

        # only assert if x0 is in the good region
        if x0 >= 2.0 and x0 < 3.0:
            assert np.isclose(optimal_params["x"], X_EXPECTED)


@pytest.mark.skipif(mpi4py is None, reason="Reason mpi4py not installed")
def test_with_bad_function_mpi():
    from chemfit.mpi_wrapper_cob import MPIWrapperCOB

    X_EXPECTED = 2.5

    def f1(params):
        if params["x"] < 1.0:
            return None
        elif params["x"] < 2.0:
            raise Exception("Some random exception")
        elif params["x"] < 3.0:
            return (params["x"] - 2.5) ** 2
        elif params["x"] < 4.0:
            return float("Nan")
        else:
            return "not even a number"

    def f2(params):
        if params["x"] < 1.0:
            return None
        elif params["x"] < 2.0:
            raise Exception("Some random exception")
        elif params["x"] < 3.0:
            return (params["x"] - 2.5) ** 2
        elif params["x"] < 4.0:
            return float("Nan")
        else:
            return "not even a number"

    ob = CombinedObjectiveFunction([f1, f2])

    for x0 in [0.5, 1.5, 2.5, 3.5, 4.5]:
        print(f"{x0 = }")

        with MPIWrapperCOB(cob=ob) as ob_mpi:

            logging.basicConfig(
                filename=f"test_fitter_bad_function_{ob_mpi.rank}.log",
                level=logging.DEBUG,
            )

            if ob_mpi.rank == 0:
                fitter = Fitter(
                    objective_function=ob_mpi,
                    initial_params={"x": x0},
                    bounds={"x": (0.0, 5.0)},
                )

                print(fitter.objective_function({"x": x0}))

                # Nevergrad should be able to handle a shitty objective function like this
                optimal_params = fitter.fit_nevergrad(budget=2000, optimizer_str="CMA")
                print("NEVERGRAD")
                print(f"{optimal_params = }")
                print(f"{fitter.info = }")

                assert np.isclose(optimal_params["x"], X_EXPECTED, atol=NG_ATOL)

                # SCIPY will probably fail, unless starting in the good region
                optimal_params = fitter.fit_scipy()
                print("SCIPY")
                print(f"{optimal_params = }")
                print(f"{fitter.info = }")

                # only assert if x0 is in the good region
                if x0 >= 2.0 and x0 < 3.0:
                    assert np.isclose(optimal_params["x"], X_EXPECTED, atol=1e-1)
            else:
                ob_mpi.worker_loop()


if __name__ == "__main__":
    # test_with_square_func()
    # test_with_square_func_bounds()
    # test_with_complicated_dict()
    test_with_bad_function_mpi()
