from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from pydictnest import get_nested, has_nested, items_nested

from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.executor_wrapper_cob import ExecutorWrapperCOB
from chemfit.fitter import Fitter, FitterEvaluateContext
from chemfit.utils import check_params_near_bounds

NG_SOLVERS = ["NgIohTuned", "Carola3", "CMA"]
NG_ATOL = 1e-1
NSTEPS_CB = 100
NG_BUDGET = 500


def collect_progress(
    step: int,
    ctxs: list[FitterEvaluateContext],
    progress: list,
    print_to_console: bool = False,
):
    for ctx in ctxs:
        info = {
            "step": step,
            "n_evals": ctx.n_evals,
            "cur_params": ctx.parameters,
            "cur_loss": ctx.loss,
            "opt_loss": ctx.opt_loss,
            "opt_params": ctx.opt_params,
        }

        if print_to_console:
            print(info)

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
        lambda step, ctxs: collect_progress(step, ctxs, progress=progress),
        n_steps=NSTEPS_CB,
    )
    optimal_params = fitter.fit_scipy()

    print(f"{optimal_params = }")
    assert np.isclose(optimal_params["x"], 2.0)
    assert np.isclose(optimal_params["y"], -1.0)

    for opt in NG_SOLVERS:
        progress = []
        optimal_params = fitter.fit_nevergrad(budget=NG_BUDGET, optimizer_str=opt)

        print(f"{opt = }")
        print(f"{optimal_params = }")
        print(f"{len(progress) = }")
        print(f"{NG_BUDGET // NSTEPS_CB = }")

        print(f"{progress[-1]['opt_loss'] = }")
        print(f"{progress[-1]['opt_params'] = }")
        print(f"{obj_func(optimal_params) = }")

        # This assert is interesting because intuitively we would expect,
        # these to be exactly equal, but this is solver dependent!!
        # The "CMA" solver, for instance, may recommend parameters it has not actually visited yet
        # Therefore, the `opt_loss`, which is only computed from actually visited parameters and the
        # obj_func(optimal_params) value may be very slightly different
        assert np.isclose(
            progress[-1]["opt_loss"], obj_func(optimal_params), atol=NG_ATOL
        )
        assert np.isclose(optimal_params["x"], 2.0, atol=NG_ATOL)
        assert np.isclose(optimal_params["y"], -1.0, atol=NG_ATOL)


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

    assert len(check_params_near_bounds(optimal_params, bounds, 1e-2)) == 1
    assert np.isclose(optimal_params["x"], 1.5)
    assert np.isclose(optimal_params["y"], -1.0)

    for opt in NG_SOLVERS:
        optimal_params = fitter.fit_nevergrad(budget=NG_BUDGET, optimizer_str=opt)
        print(f"{opt = }")
        print(f"{optimal_params = }")

        assert np.isclose(optimal_params["x"], 1.5, atol=NG_ATOL)
        assert np.isclose(optimal_params["y"], -1.0, atol=NG_ATOL)


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
    assert np.isclose(optimal_params["params"]["x"], 1.5)
    assert np.isclose(optimal_params["y"], -1.0)

    optimal_params = fitter.fit_nevergrad(budget=NG_BUDGET)

    print(f"{optimal_params = }")
    assert np.isclose(optimal_params["params"]["x"], 1.5, atol=NG_ATOL)
    assert np.isclose(optimal_params["y"], -1.0, atol=NG_ATOL)


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
    check_solution(optimal_params)

    optimal_params = fitter.fit_nevergrad(budget=NG_BUDGET)
    print(f"{optimal_params = }")

    check_solution(optimal_params)


def test_with_square_func_threadpool():
    def cont1(params: dict):
        return 2.0 * (params["x"] - 2) ** 2

    def cont2(params: dict):
        return 3.0 * (params["y"] + 1) ** 2

    obj_func = CombinedObjectiveFunction([cont1, cont2])
    async_obj_func = ExecutorWrapperCOB(obj_func)

    initial_params = {"x": 0.0, "y": 0.0}
    fitter = Fitter(objective_function=async_obj_func, initial_params=initial_params)

    NUM_WORKERS = 5

    for opt in NG_SOLVERS:
        progress = []

        fitter.register_callback(
            lambda step, ctxs, progress=progress: collect_progress(
                step, ctxs, progress=progress, print_to_console=True
            ),
            n_steps=NSTEPS_CB,
        )

        contexts = [FitterEvaluateContext() for _ in range(NUM_WORKERS)]

        optimal_params = fitter.fit_nevergrad(
            budget=NG_BUDGET,
            optimizer_str=opt,
            num_workers=NUM_WORKERS,
            executor=ThreadPoolExecutor(NUM_WORKERS),
            contexts=contexts,
        )

        print(f"{opt = }")
        print(f"{optimal_params = }")
        print(f"{len(progress) = }")
        print(f"{NG_BUDGET // NSTEPS_CB = }")
        print(f"{obj_func(optimal_params) = }")

        # This assert is interesting because intuitively we would expect,
        # these to be exactly equal, but this is solver dependent!!
        # The "CMA" solver, for instance, may recommend parameters it has not actually visited yet
        # Therefore, the `opt_loss`, which is only computed from actually visited parameters and the
        # obj_func(optimal_params) value may be very slightly different
        assert np.isclose(optimal_params["x"], 2.0, atol=NG_ATOL)
        assert np.isclose(optimal_params["y"], -1.0, atol=NG_ATOL)


def test_with_square_func_processpool():
    loky = pytest.importorskip("loky", reason="Missing loky")

    def cont1(params: dict):
        return 2.0 * (params["x"] - 2) ** 2

    def cont2(params: dict):
        return 3.0 * (params["y"] + 1) ** 2

    obj_func = CombinedObjectiveFunction([cont1, cont2])
    async_obj_func = ExecutorWrapperCOB(obj_func)

    initial_params = {"x": 0.0, "y": 0.0}
    fitter = Fitter(objective_function=async_obj_func, initial_params=initial_params)

    NUM_WORKERS = 5

    for opt in NG_SOLVERS:
        progress = []

        fitter.register_callback(
            lambda step, ctxs, progress=progress: collect_progress(
                step, ctxs, progress=progress, print_to_console=True
            ),
            n_steps=NSTEPS_CB,
        )

        contexts = [FitterEvaluateContext() for _ in range(NUM_WORKERS)]

        optimal_params = fitter.fit_nevergrad(
            budget=NG_BUDGET,
            optimizer_str=opt,
            num_workers=NUM_WORKERS,
            executor=loky.ProcessPoolExecutor(NUM_WORKERS),
            contexts=contexts,
        )

        print(f"{opt = }")
        print(f"{optimal_params = }")
        print(f"{len(progress) = }")
        print(f"{NG_BUDGET // NSTEPS_CB = }")
        print(f"{obj_func(optimal_params) = }")

        # This assert is interesting because intuitively we would expect,
        # these to be exactly equal, but this is solver dependent!!
        # The "CMA" solver, for instance, may recommend parameters it has not actually visited yet
        # Therefore, the `opt_loss`, which is only computed from actually visited parameters and the
        # obj_func(optimal_params) value may be very slightly different
        assert np.isclose(optimal_params["x"], 2.0, atol=NG_ATOL)
        assert np.isclose(optimal_params["y"], -1.0, atol=NG_ATOL)


def test_seed_observations():
    n_calls = 0

    def objective(params: dict[str, float]) -> float:
        nonlocal n_calls
        n_calls += 1
        return (params["x"] - 3.0) ** 2

    fitter = Fitter(
        objective_function=objective,
        initial_params={"x": 0.0},
        bounds={"x": (0.0, 5.0)},
    )

    contexts = [FitterEvaluateContext(), FitterEvaluateContext()]

    opt_params = fitter.fit_nevergrad(
        budget=2,
        num_workers=2,
        contexts=contexts,
        initial_observations=[
            ({"x": 2.0}, 1.0),  # valid
            ({"x": 10.0}, 0.0),  # invalid, should be skipped
        ],
    )

    # replayed observations should not consume live evaluation budget
    assert n_calls == 2

    # valid replayed point should have been used to seed incumbent state
    assert contexts[0].opt_loss is not None
    assert contexts[0].opt_loss <= 1.0

    # invalid replayed point should not become incumbent
    assert contexts[0].opt_params is not None
    assert 0.0 <= contexts[0].opt_params["x"] <= 5.0

    # optimizer should still return an in-bounds result
    assert 0.0 <= opt_params["x"] <= 5.0


def test_nevergrad_evaluates_partial_final_batch():
    n_calls = 0

    def objective(params: dict[str, float]) -> float:
        nonlocal n_calls
        n_calls += 1
        return params["x"] ** 2

    fitter = Fitter(objective, initial_params={"x": 1.0})
    fitter.fit_nevergrad(budget=3, num_workers=2)

    assert n_calls == 3


def test_user_supplied_ask_tell_interface():
    candidates = iter([{"x": 0.0}, {"x": 2.0}, {"x": 4.0}])
    observations = []

    fitter = Fitter(lambda params: (params["x"] - 2.0) ** 2, {"x": 0.0})
    fitter.init()
    for params in candidates:
        loss = fitter.ask(params)
        observations.append((params, loss))
        fitter.tell()
    result = fitter.finish()

    assert result == {"x": 2.0}
    assert observations == [
        ({"x": 0.0}, 4.0),
        ({"x": 2.0}, 0.0),
        ({"x": 4.0}, 4.0),
    ]
    assert fitter.contexts[0].n_evals == 3


def test_user_supplied_ask_tell_recommendation_and_partial_batch():
    fitter = Fitter(lambda params: params["x"] ** 2, {"x": 0.0})

    with ThreadPoolExecutor(2) as executor:
        fitter.init(num_workers=2, executor=executor)
        losses = fitter.ask([{"x": 0.0}, {"x": 1.0}])
        fitter.tell()
        final_loss = fitter.ask([{"x": 2.0}])
        fitter.tell()
        result = fitter.finish({"x": 0.5})

    assert losses == [0.0, 1.0]
    assert final_loss == [4.0]
    assert result == {"x": 0.5}


if __name__ == "__main__":
    test_with_square_func()
