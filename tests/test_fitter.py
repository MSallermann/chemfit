import math

import nevergrad as ng
import pytest

from chemfit.abstract_objective_function import PipelineComponent, component
from chemfit.fitter import Fitter
from chemfit.graph_schedule_base import SerialGraphScheduler


def square(parameters):
    return parameters["x"] ** 2


def test_scipy_converges_with_nested_parameters_and_bounds():
    def objective(parameters):
        return (parameters["outer"]["x"] - 2.0) ** 2

    fitter = Fitter(
        objective,
        initial_params={"outer": {"x": 0.0}},
        bounds={"outer": {"x": (-1.0, 3.0)}},
    )

    result = fitter.fit_scipy()

    assert result["outer"]["x"] == pytest.approx(2.0, abs=1e-5)


def test_user_session_supports_batches_and_reuses_contexts_cleanly():
    @component()
    def double(parameters):
        return 2 * parameters["x"]

    @component()
    def add_one(value):
        return value + 1

    fitter = Fitter(
        PipelineComponent(double, add_one),
        initial_params={"x": 1.0},
    )
    fitter.init(batch_size=2)

    assert fitter.evaluate([{"x": 1.0}, {"x": 3.0}]) == [3.0, 7.0]
    assert fitter.evaluate({"x": 2.0}) == 5.0
    assert len(fitter.contexts[0].incoming) == 2
    assert fitter.contexts[0].n_evals == 2
    assert fitter.finish() == {"x": 1.0}


def test_internal_schedule_is_reprepared_and_closed_per_session():
    fitter = Fitter(square, initial_params={"x": 1.0})
    fitter.init()
    first = fitter._schedule  # noqa: SLF001
    assert first is not None
    fitter.evaluate({"x": 1.0})
    fitter.finish()
    assert first.closed

    fitter.init()
    second = fitter._schedule  # noqa: SLF001
    assert second is not None and second is not first
    fitter.evaluate({"x": 2.0})
    fitter.finish()
    assert second.closed


def test_borrowed_schedule_remains_open_after_objective_failure():
    @component()
    def failing(_parameters):
        raise ValueError("bad objective")

    with SerialGraphScheduler().prepare(failing) as schedule:
        fitter = Fitter(schedule, initial_params={"x": 1.0})
        fitter.init()
        with pytest.raises(ValueError, match="bad objective"):
            fitter.evaluate({"x": 1.0})
        assert not schedule.closed


def test_callbacks_run_on_intervals_and_the_final_partial_step():
    calls = []
    fitter = Fitter(square, initial_params={"x": 1.0})
    fitter.register_callback(lambda step, _contexts: calls.append(step), n_steps=2)
    fitter.init()
    for step in range(1, 4):
        fitter.evaluate({"x": float(step)})
        fitter.step()
    fitter.finish()

    assert calls == [2, 3]


@pytest.mark.parametrize("value", (math.nan, math.inf, "not-a-number"))
def test_non_finite_or_non_numeric_results_use_the_penalty(value):
    fitter = Fitter(lambda _parameters: value, initial_params={"x": 1.0})
    fitter.init()

    assert fitter.evaluate({"x": 1.0}) == 1e5


def test_swallowed_exception_uses_the_penalty_and_counts_the_evaluation():
    def failing(_parameters):
        raise ValueError("bad parameters")

    fitter = Fitter(
        failing,
        initial_params={"x": 1.0},
        swallow_exceptions=True,
        log_exceptions=False,
    )
    fitter.init()

    assert fitter.evaluate({"x": 1.0}) == 1e5
    assert fitter.contexts[0].n_evals == 1


def test_nevergrad_accepts_partial_explicit_parameterization():
    fitter = Fitter(
        lambda parameters: parameters["x"] ** 2,
        initial_params={"x": 1.0, "label": "fixed"},
    )

    result = fitter.fit_nevergrad(
        budget=1,
        optimizer_str="OnePlusOne",
        parametrization={"label": ng.p.Constant("fixed")},
    )

    assert result["label"] == "fixed"


@pytest.mark.parametrize(
    ("action", "message"),
    (
        (lambda fitter: fitter.init(batch_size=0), "batch_size"),
        (lambda fitter: fitter.evaluate({"x": 1.0}), "init"),
        (lambda fitter: fitter.step(), "init"),
    ),
)
def test_session_validation(action, message):
    fitter = Fitter(square, initial_params={"x": 1.0})

    with pytest.raises((ValueError, RuntimeError), match=message):
        action(fitter)


def test_rejects_unknown_bound_keys():
    with pytest.raises(ValueError, match="unknown"):
        Fitter(square, initial_params={"x": 1.0}, bounds={"typo": (0.0, 1.0)})
