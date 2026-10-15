from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

import chemfit
from chemfit.graph_schedule_base import SerialGraphScheduler


def test_combine_accepts_an_iterable_and_context_aware_reducer():
    @chemfit.objective()
    def term(parameters: dict[str, float], *, offset):
        return parameters["x"] + offset

    def reduce(values, *, ctx) -> float:
        ctx.meta["values"] = tuple(values)
        return max(values)

    objective = chemfit.combine(
        (term.bind(offset=value) for value in (1, 3)),
        weights=(2.0, 1.0),
        reduction=reduce,
    )
    ctx = chemfit.EvaluateContext()

    assert objective({"x": 2}, ctx) == 6
    assert ctx.meta["values"] == (6, 5)


def test_evaluate_many_restores_input_order_with_caller_owned_executor():
    @chemfit.objective(pass_ctx=True)
    def objective(parameters, *, ctx):
        ctx.meta["value"] = parameters["x"]
        return parameters["x"] ** 2

    executor = ThreadPoolExecutor(max_workers=2)
    try:
        contexts = chemfit.evaluate_many(
            objective,
            [{"x": 3.0}, {"x": 1.0}, {"x": 2.0}],
            executor=executor,
        )
        assert executor.submit(lambda: 1).result() == 1
    finally:
        executor.shutdown()

    assert [ctx.output for ctx in contexts] == [9.0, 1.0, 4.0]
    assert [ctx.meta["value"] for ctx in contexts] == [3.0, 1.0, 2.0]


def test_evaluate_many_accepts_a_custom_scheduler():
    contexts = chemfit.evaluate_many(
        lambda parameters: parameters["x"] + 1,
        [{"x": 1}, {"x": 4}],
        scheduler=SerialGraphScheduler(),
    )

    assert [ctx.output for ctx in contexts] == [2, 5]


def test_fit_nevergrad_records_best_evaluation_and_closes_internal_schedule():
    class RecordingScheduler:
        schedule = None

        def prepare(self, objective):
            self.schedule = SerialGraphScheduler().prepare(objective)
            return self.schedule

    scheduler = RecordingScheduler()
    result = chemfit.fit_nevergrad(
        lambda parameters: parameters["x"] ** 2,
        initial={"x": 1.0},
        budget=2,
        optimizer="OnePlusOne",
        scheduler=scheduler,
    )

    assert result.best_loss == result.best_context.opt_loss
    assert result.best_parameters == result.best_context.opt_params
    assert scheduler.schedule is not None
    assert scheduler.schedule.closed


def test_fit_nevergrad_borrows_a_prepared_schedule():
    @chemfit.objective()
    def objective(parameters):
        return parameters["x"] ** 2

    with SerialGraphScheduler().prepare(objective) as schedule:
        result = chemfit.fit_nevergrad(
            schedule,
            initial={"x": 1.0},
            budget=1,
            optimizer="OnePlusOne",
        )

        assert result.best_loss == 1.0
        assert not schedule.closed


@pytest.mark.parametrize("option", ["scheduler", "executor", "execution_workers"])
def test_prepared_schedule_rejects_backend_options(option):
    @chemfit.objective()
    def objective(parameters):
        return parameters["x"] ** 2

    kwargs: dict[str, Any]
    executor = None
    with SerialGraphScheduler().prepare(objective) as schedule:
        if option == "scheduler":
            kwargs = {"scheduler": SerialGraphScheduler()}
        elif option == "execution_workers":
            kwargs = {"execution_workers": 2}
        else:
            executor = ThreadPoolExecutor(max_workers=1)
            kwargs = {"executor": executor}

        try:
            with pytest.raises(ValueError, match=option):
                chemfit.fit_nevergrad(
                    schedule,
                    initial={"x": 1.0},
                    budget=1,
                    **kwargs,
                )
        finally:
            if executor is not None:
                executor.shutdown()
