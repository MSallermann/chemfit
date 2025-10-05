import numpy as np
from pydictnest import get_nested, has_nested, items_nested

from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.fitter import CallbackInfo, Fitter
from chemfit.utils import check_params_near_bounds

NG_SOLVERS = ["NgIohTuned", "Carola3", "CMA"]
NG_ATOL = 5e-2
NSTEPS_CB = 100
NG_BUDGET = 2000


def collect_progress(
    info: CallbackInfo,
    progress: list,
):
    progress.append(info)


def test_with_square_func():
    def cont1(params: dict):
        return 2.0 * (params["x"] - 2) ** 2

    def cont2(params: dict):
        return 3.0 * (params["y"] + 1) ** 2

    obj_func = CombinedObjectiveFunction([cont1, cont2])

    initial_params = {"x": 0.0, "y": 0.0}
    fitter = Fitter(objective_function=obj_func, initial_params=initial_params)

    progress = []
    fitter.register_callback(
        lambda args: collect_progress(args, progress=progress), n_steps=NSTEPS_CB
    )
    optimal_params = fitter.fit_scipy()

    print(f"{optimal_params = }")
    assert np.isclose(optimal_params["x"], 2.0)
    assert np.isclose(optimal_params["y"], -1.0)

    init_val = obj_func(initial_params)
    opt_val = obj_func(optimal_params)
    assert fitter.info.initial_value is not None
    assert fitter.info.final_value is not None
    assert np.isclose(init_val, fitter.info.initial_value)
    assert np.isclose(opt_val, fitter.info.final_value)

    for opt in NG_SOLVERS:
        progress = []
        optimal_params = fitter.fit_nevergrad(budget=NG_BUDGET, optimizer_str=opt)

        print(f"{opt = }")
        print(f"{optimal_params = }")
        print(f"{fitter.info = }")
        print(f"{len(progress) = }")
        print(f"{NG_BUDGET // NSTEPS_CB = }")

        print(f"{progress[-1].opt_loss = }")
        print(f"{progress[-1].opt_params = }")
        print(f"{obj_func(optimal_params) = }")
        print(f"{fitter.info.final_value = }")

        # This assert is interesting because intuitively we would expect,
        # these to be exactly equal, but this is solver dependent!!
        # The "CMA" solver, for instance, may recommend parameters it has not actually visited yet
        # Therefore, the `opt_loss`, which is only computed from actually visited parameters and the
        # obj_func(optimal_params) value may be very slightly different
        assert np.isclose(progress[-1].opt_loss, obj_func(optimal_params))
        assert np.isclose(optimal_params["x"], 2.0, atol=NG_ATOL)
        assert np.isclose(optimal_params["y"], -1.0, atol=NG_ATOL)
        assert np.isclose(obj_func(initial_params), fitter.info.initial_value)
        assert np.isclose(obj_func(optimal_params), fitter.info.final_value)

    print(f"{fitter.info = }")


def test_with_square_func_bounds():
    def cont1(params: dict):
        return 2.0 * (params["x"] - 2) ** 2

    def cont2(params: dict):
        return 3.0 * (params["y"] + 1) ** 2

    obj_func = CombinedObjectiveFunction([cont1, cont2])

    initial_params = {"x": 0.0, "y": 0.0}
    bounds = {"x": (0.0, 1.5)}

    fitter = Fitter(
        objective_function=obj_func,
        initial_params=initial_params,
        bounds=bounds,
        near_bound_tol=1e-2,
    )

    optimal_params = fitter.fit_scipy()

    print(f"{optimal_params = }")
    print(f"{fitter.info = }")

    assert len(check_params_near_bounds(optimal_params, bounds, 1e-2)) == 1
    assert np.isclose(optimal_params["x"], 1.5)
    assert np.isclose(optimal_params["y"], -1.0)
    assert fitter.info.initial_value is not None
    assert fitter.info.final_value is not None
    assert np.isclose(obj_func(initial_params), fitter.info.initial_value)
    assert np.isclose(obj_func(optimal_params), fitter.info.final_value)

    for opt in NG_SOLVERS:
        optimal_params = fitter.fit_nevergrad(budget=NG_BUDGET, optimizer_str=opt)
        print(f"{opt = }")
        print(f"{optimal_params = }")
        print(f"{fitter.info = }")

        assert np.isclose(optimal_params["x"], 1.5, atol=NG_ATOL)
        assert np.isclose(optimal_params["y"], -1.0, atol=NG_ATOL)
        assert np.isclose(obj_func(initial_params), fitter.info.initial_value)
        assert np.isclose(obj_func(optimal_params), fitter.info.final_value)

    print(f"{fitter.info = }")


def test_with_nested_dict():
    def cont1(params: dict):
        return 2.0 * (params["params"]["x"] - 2) ** 2

    def cont2(params: dict):
        return 3.0 * (params["y"] + 1) ** 2

    obj_func = CombinedObjectiveFunction([cont1, cont2])

    initial_params = {"params": {"x": 0.0}, "y": 0.0}
    bounds = {"params": {"x": (0.0, 1.5)}}

    fitter = Fitter(
        objective_function=obj_func, initial_params=initial_params, bounds=bounds
    )

    optimal_params = fitter.fit_scipy()
    print(f"{optimal_params = }")
    print(f"{fitter.info = }")
    assert np.isclose(optimal_params["params"]["x"], 1.5)
    assert np.isclose(optimal_params["y"], -1.0)
    assert fitter.info.initial_value is not None
    assert fitter.info.final_value is not None
    assert np.isclose(obj_func(initial_params), fitter.info.initial_value)
    assert np.isclose(obj_func(optimal_params), fitter.info.final_value)

    optimal_params = fitter.fit_nevergrad(budget=NG_BUDGET)

    print(f"{optimal_params = }")
    print(f"{fitter.info = }")
    assert np.isclose(optimal_params["params"]["x"], 1.5, atol=NG_ATOL)
    assert np.isclose(optimal_params["y"], -1.0, atol=NG_ATOL)
    assert np.isclose(obj_func(initial_params), fitter.info.initial_value)
    assert np.isclose(obj_func(optimal_params), fitter.info.final_value)


def test_with_complicated_dict():
    def ob(params: dict):
        res = 0
        for _k, v in items_nested(params):
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
    def check_solution(opt_params: dict):
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
    assert fitter.info.initial_value is not None
    assert fitter.info.final_value is not None
    assert np.isclose(ob(initial_params), fitter.info.initial_value)
    assert np.isclose(ob(optimal_params), fitter.info.final_value)

    optimal_params = fitter.fit_nevergrad(budget=NG_BUDGET)
    print(f"{optimal_params = }")
    print(f"{fitter.info = }")
    check_solution(optimal_params)
    assert fitter.info.initial_value is not None
    assert fitter.info.final_value is not None

    assert np.isclose(ob(initial_params), fitter.info.initial_value)
    assert np.isclose(ob(optimal_params), fitter.info.final_value)

    print(f"{fitter.info = }")


if __name__ == "__main__":
    test_with_square_func()
