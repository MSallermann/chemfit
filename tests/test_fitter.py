from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

import nevergrad as ng
import numpy as np
import pytest

from chemfit.abstract_objective_function import EvaluateContext
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.fitter import Fitter, FitterEvaluateContext
from chemfit.utils import check_params_near_bounds
from chemfit.wrap_funcs import WrappedObjectiveFunctor
from pydictnest import get_nested, has_nested, items_nested

NG_SOLVERS = ["NgIohTuned", "Carola3", "CMA"]
NG_SMOKE_BUDGET = 8
SCIPY_ATOL = 1e-4


def square_x(params: dict[str, float]) -> float:
    return params["x"] ** 2


def square_x_with_quantities(params: dict[str, float], ctx: EvaluateContext) -> float:
    ctx.quantities = {"evaluated_x": params["x"]}
    return square_x(params)


def collect_progress(
    step: int,
    ctxs: list[FitterEvaluateContext],
    progress: list,
):
    progress.extend(
        [
            {
                "step": step,
                "n_evals": ctx.n_evals,
                "cur_params": ctx.parameters,
                "cur_loss": ctx.loss,
                "opt_loss": ctx.opt_loss,
                "opt_params": ctx.opt_params,
            }
            for ctx in ctxs
        ]
    )


def _combined_quadratic() -> CombinedObjectiveFunction:
    def cont1(params: dict[str, float]) -> float:
        return 2.0 * (params["x"] - 2.0) ** 2

    def cont2(params: dict[str, float]) -> float:
        return 3.0 * (params["y"] + 1.0) ** 2

    return CombinedObjectiveFunction([cont1, cont2])


def test_scipy_converges_on_combined_objective():
    fitter = Fitter(
        objective_function=_combined_quadratic(),
        initial_params={"x": 0.0, "y": 0.0},
    )

    optimal_params = fitter.fit_scipy()

    assert np.isclose(optimal_params["x"], 2.0)
    assert np.isclose(optimal_params["y"], -1.0)


@pytest.mark.parametrize("optimizer", NG_SOLVERS)
def test_nevergrad_supported_solvers_smoke(optimizer: str):
    """Exercise ChemFit's Nevergrad integration without testing optimizer quality."""
    fitter = Fitter(
        objective_function=_combined_quadratic(),
        initial_params={"x": 0.0, "y": 0.0},
        bounds={"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
    )

    result = fitter.fit_nevergrad(
        budget=NG_SMOKE_BUDGET,
        optimizer_str=optimizer,
    )

    assert set(result) == {"x", "y"}
    assert -5.0 <= result["x"] <= 5.0
    assert -5.0 <= result["y"] <= 5.0
    assert np.isfinite(result["x"])
    assert np.isfinite(result["y"])


def test_nevergrad_callbacks_run_at_requested_interval():
    progress: list[dict[str, Any]] = []
    fitter = Fitter(square_x, initial_params={"x": 1.0})
    fitter.register_callback(
        lambda step, ctxs: collect_progress(step, ctxs, progress),
        n_steps=2,
    )

    fitter.fit_nevergrad(
        budget=4,
        optimizer_str="OnePlusOne",
    )

    assert progress
    assert progress[-1]["n_evals"] == 4
    assert progress[-1]["opt_loss"] is not None
    assert progress[-1]["opt_params"] is not None


def test_scipy_respects_bounds():
    bounds = {"x": (0.0, 1.5)}
    fitter = Fitter(
        objective_function=_combined_quadratic(),
        initial_params={"x": 0.0, "y": 0.0},
        bounds=bounds,
        near_bound_tol=1e-2,
    )

    optimal_params = fitter.fit_scipy()

    assert len(check_params_near_bounds(optimal_params, bounds, 1e-2)) == 1
    assert np.isclose(optimal_params["x"], 1.5)
    assert np.isclose(optimal_params["y"], -1.0)


def test_scipy_supports_nested_parameter_dicts():
    def cont1(params: dict) -> float:
        return 2.0 * (params["params"]["x"] - 2.0) ** 2

    def cont2(params: dict) -> float:
        return 3.0 * (params["y"] + 1.0) ** 2

    fitter = Fitter(
        objective_function=CombinedObjectiveFunction([cont1, cont2]),
        initial_params={"params": {"x": 0.0}, "y": 0.0},
        bounds={"params": {"x": (0.0, 1.5)}},
    )

    optimal_params = fitter.fit_scipy()

    assert np.isclose(optimal_params["params"]["x"], 1.5)
    assert np.isclose(optimal_params["y"], -1.0)


def test_scipy_supports_complicated_nested_parameter_dicts():
    def objective(params: dict) -> float:
        return sum(value**2 for _key, value in items_nested(params))

    initial_params = {
        "electrostatic": {
            "bla": {"a": 1.0, "b": 1.0, "c": 1.0},
            "foo": 1.0,
        },
        "dispersion": 0.4,
        "params": {"a": 1.0, "b": 1.0},
    }
    bounds = {
        "dispersion": [0.2, 2.0],
        "electrostatic": {"bla": {"a": [0.5, 1.0]}},
    }

    fitter = Fitter(
        objective_function=objective,
        initial_params=initial_params,
        bounds=bounds,
    )
    optimal_params = fitter.fit_scipy()

    for key, value in items_nested(optimal_params):
        if has_nested(bounds, key):
            lower, _upper = get_nested(bounds, key)
            assert np.isclose(value, lower, atol=SCIPY_ATOL)
        else:
            assert np.isclose(value, 0.0, atol=SCIPY_ATOL)


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


def test_process_pool_preserves_fitter_context_state():
    objective = WrappedObjectiveFunctor(square_x_with_quantities, pass_ctx=True)
    fitter = Fitter(objective, {"x": 0.0})
    with ProcessPoolExecutor(2) as executor:
        fitter.init(num_workers=2, executor=executor)
        assert fitter.ask([{"x": 2.0}, {"x": 3.0}]) == [4.0, 9.0]
        assert fitter.ask([{"x": 1.0}, {"x": 4.0}]) == [1.0, 16.0]
    first, second = fitter.contexts
    assert first.n_evals == 2
    assert first.opt_loss == 1.0
    assert first.opt_params == {"x": 1.0}
    assert first.opt_quantities == {"evaluated_x": 1.0}
    assert second.n_evals == 2
    assert second.opt_loss == 9.0
    assert second.opt_params == {"x": 3.0}
    assert second.opt_quantities == {"evaluated_x": 3.0}


def test_new_best_without_quantities_clears_previous_best_quantities():
    fitter = Fitter(square_x, {"x": 0.0})
    ctx = FitterEvaluateContext()
    ctx.quantities = {"source": "previous best"}
    fitter.objective_function.post_process_return_value({"x": 2.0}, 4.0, ctx)
    ctx.quantities = None
    fitter.objective_function.post_process_return_value({"x": 1.0}, 1.0, ctx)
    assert ctx.opt_loss == 1.0
    assert ctx.opt_params == {"x": 1.0}
    assert ctx.opt_quantities is None


def test_initial_parameters_must_be_mapping():
    with pytest.raises(TypeError, match="must be a mapping"):
        Fitter(lambda _params: 0.0, [1.0, 2.0])  # type: ignore


def test_nevergrad_parameter_leaves():
    evaluated = []

    def objective(params: dict[str, Any]) -> float:
        evaluated.append(params)
        model_loss = 0.0 if params["model"] == "quadratic" else 1.0
        return (
            model_loss
            + params["core"]["x"] ** 2
            + np.sum(params["core"]["weights"] ** 2)
        )

    fitter = Fitter(
        objective,
        initial_params={
            "model": "quadratic",
            "core": {
                "x": 1.0,
                "weights": np.array([1.0, 2.0]),
            },
            "metadata": "fixed",
        },
    )
    assert fitter.initial_parameters["model"] == "quadratic"
    assert fitter.initial_parameters["core"]["x"] == 1.0
    assert np.array_equal(fitter.initial_parameters["core"]["weights"], [1.0, 2.0])
    result = fitter.fit_nevergrad(
        budget=3,
        optimizer_str="OnePlusOne",
        parametrization={
            "model": ng.p.Choice(["quadratic", "absolute"]),
            "core": {
                "x": ng.p.Log(init=1.0, lower=0.1, upper=10.0),
                "weights": ng.p.Array(init=[1.0, 2.0], lower=-3.0, upper=3.0),
            },
            "metadata": ng.p.Constant("fixed"),
        },
    )
    assert result["model"] in {"quadratic", "absolute"}
    assert 0.1 <= result["core"]["x"] <= 10.0
    assert isinstance(result["core"]["weights"], np.ndarray)
    assert result["core"]["weights"].shape == (2,)
    assert result["metadata"] == "fixed"
    assert evaluated
    assert all(params["metadata"] == "fixed" for params in evaluated)


def test_nevergrad_parametrization_can_be_partial():
    choice = ng.p.Choice(["linear", "quadratic"])
    fitter = Fitter(
        lambda params: params["x"] ** 2,
        initial_params={
            "x": 1.0,
            "model": {"kind": "linear"},
            "label": "fixed",
        },
        bounds={"x": (0.0, 2.0)},
    )
    instrumentation = fitter._make_nevergrad_parameterization(  # noqa: SLF001
        {
            "model": {"kind": choice},
            "label": ng.p.Constant("fixed"),
        }
    )
    positional_parameters = instrumentation[0]
    assert isinstance(positional_parameters, ng.p.Tuple)
    parameter_leaves = positional_parameters[0]
    assert isinstance(parameter_leaves, ng.p.Dict)
    x_parameter = parameter_leaves["x"]
    assert isinstance(x_parameter, ng.p.Scalar)
    lower_bound, upper_bound = x_parameter.bounds
    assert lower_bound is not None
    assert upper_bound is not None
    assert np.array_equal(lower_bound, [0.0])
    assert np.array_equal(upper_bound, [2.0])
    model_kind_parameter = parameter_leaves["model.kind"]
    assert isinstance(model_kind_parameter, ng.p.Choice)
    assert model_kind_parameter is not choice
    label_parameter = parameter_leaves["label"]
    assert isinstance(label_parameter, ng.p.Constant)
    assert label_parameter.value == "fixed"


def test_nevergrad_requires_explicit_non_numeric_leaves():
    fitter = Fitter(lambda _params: 0.0, {"model": "linear"})
    with pytest.raises(TypeError, match=r"ng\.p\.Constant"):
        fitter.fit_nevergrad(budget=1)


def test_nevergrad_parametrization_requires_parameter_leaves():
    fitter = Fitter(square_x, {"x": 1.0})
    with pytest.raises(TypeError, match="Nevergrad parameters"):
        fitter.fit_nevergrad(budget=1, parametrization={"x": 2.0})
