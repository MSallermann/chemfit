from __future__ import annotations

import asyncio
import functools
import pickle
import random
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from chemfit import abstract_objective_function, async_helpers
from chemfit.abstract_objective_function import EvaluateContext
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.wrap_funcs import to_quantity_computer
from pydictnest import get_nested, items_nested

if TYPE_CHECKING:
    from collections.abc import Callable


class MyFunctor(abstract_objective_function.ObjectiveFunctor):
    def _evaluate(
        self,
        parameters: dict[str, float],
        ctx: EvaluateContext,  # noqa: ARG002
    ) -> float:
        return parameters["a"] ** 2 - parameters["b"]


class MyComputer(abstract_objective_function.QuantityComputer):
    def _compute(
        self,
        parameters: dict[str, float],
        ctx: abstract_objective_function.EvaluateContext,
    ) -> dict[str, float]:
        ctx.temp.a2 = parameters["a"] ** 2
        ctx.meta["meta_b2"] = parameters["b"] ** 2

        # Sleep for a random time to simulate a variable amount of work
        time.sleep(random.random() * 0.1)  # noqa: S311

        return {"res": ctx.temp.a2 - parameters["b"]}


class _PreHook:
    def __init__(self, callback: Callable[[EvaluateContext], None]) -> None:
        self.callback = callback

    def pre_eval(self, ctx: EvaluateContext) -> None:
        self.callback(ctx)


class _PostHook:
    def __init__(self, callback: Callable[[EvaluateContext], None]) -> None:
        self.callback = callback

    def post_eval(self, ctx: EvaluateContext) -> None:
        self.callback(ctx)


def test_objective_functor_hooks_run_in_registration_order():
    objective = MyFunctor()
    ctx = EvaluateContext()
    parameters = {"a": 2.0, "b": 3.0}

    # Record observations on the per-evaluation context. Capturing and mutating
    # a shared list from hooks would be unsafe during concurrent evaluation and
    # would not propagate back when a hook runs in another process.
    ctx.meta["hook_calls"] = []

    objective.register_eval_hook(
        _PreHook(
            lambda hook_ctx: hook_ctx.meta["hook_calls"].append(
                ("pre-1", hook_ctx.parameters, hook_ctx.loss)
            )
        )
    )
    objective.register_eval_hook(
        _PreHook(
            lambda hook_ctx: hook_ctx.meta["hook_calls"].append(
                ("pre-2", hook_ctx.parameters, hook_ctx.loss)
            )
        )
    )
    objective.register_eval_hook(
        _PostHook(
            lambda hook_ctx: hook_ctx.meta["hook_calls"].append(
                ("post-1", hook_ctx.loss, hook_ctx.temp.exception)
            )
        )
    )
    objective.register_eval_hook(
        _PostHook(
            lambda hook_ctx: hook_ctx.meta["hook_calls"].append(
                ("post-2", hook_ctx.loss, hook_ctx.temp.exception)
            )
        )
    )

    assert objective(parameters, ctx) == 1.0
    assert ctx.meta["hook_calls"] == [
        ("pre-1", parameters, None),
        ("pre-2", parameters, None),
        ("post-1", 1.0, None),
        ("post-2", 1.0, None),
    ]


def test_objective_functor_collects_all_post_hook_exceptions():
    objective = MyFunctor()
    ctx = EvaluateContext()
    ctx.meta["hook_calls"] = []

    def fail(hook_ctx: EvaluateContext, message: str) -> None:
        hook_ctx.meta["hook_calls"].append(message)
        raise ValueError(message)

    objective.register_eval_hook(
        _PostHook(functools.partial(fail, message="first failure"))
    )
    objective.register_eval_hook(
        _PostHook(
            lambda hook_ctx: hook_ctx.meta["hook_calls"].append("successful hook")
        )
    )
    objective.register_eval_hook(
        _PostHook(functools.partial(fail, message="second failure"))
    )

    with pytest.raises(
        abstract_objective_function.ObjectiveFunctor.PostEvalHookError
    ) as exc_info:
        objective({"a": 2.0, "b": 3.0}, ctx)

    assert ctx.meta["hook_calls"] == [
        "first failure",
        "successful hook",
        "second failure",
    ]
    assert ctx.loss == 1.0
    assert ctx.temp.exception is None
    assert [str(exc) for exc in exc_info.value.exceptions] == [
        "first failure",
        "second failure",
    ]

    restored = pickle.loads(pickle.dumps(exc_info.value))  # noqa: S301
    assert isinstance(
        restored, abstract_objective_function.ObjectiveFunctor.PostEvalHookError
    )
    assert [str(exc) for exc in restored.exceptions] == [
        "first failure",
        "second failure",
    ]


def test_evaluation_exception_remains_primary_when_post_hook_fails():
    evaluation_error = ValueError("evaluation failed")
    ctx = EvaluateContext()
    ctx.meta["hook_calls"] = []

    class FailingFunctor(abstract_objective_function.ObjectiveFunctor):
        def _evaluate(
            self,
            parameters: dict[str, float],  # noqa: ARG002
            ctx: EvaluateContext,  # noqa: ARG002
        ) -> float:
            raise evaluation_error

    def failing_hook(hook_ctx: EvaluateContext) -> None:
        hook_ctx.meta["hook_calls"].append("failing hook")
        msg = "post hook failed"
        raise RuntimeError(msg)

    def observing_hook(hook_ctx: EvaluateContext) -> None:
        hook_ctx.meta["hook_calls"].append("observing hook")
        assert hook_ctx.loss is None
        assert hook_ctx.temp.exception is evaluation_error

    objective = FailingFunctor()
    objective.register_eval_hook(_PostHook(failing_hook))
    objective.register_eval_hook(_PostHook(observing_hook))

    with pytest.raises(ValueError, match="evaluation failed") as exc_info:
        objective({"a": 2.0}, ctx)

    assert exc_info.value is evaluation_error
    assert ctx.meta["hook_calls"] == ["failing hook", "observing hook"]


def loss1(q: dict[str, float]):
    return q["res"]


def loss2(q: dict[str, float], something_else: float):
    return q["res"] + something_else


def loss3(q: dict[str, float], p: dict[str, float]):
    return q["res"] + p["b"]


def test():
    my_func = MyFunctor()

    params = {"a": 2.0, "b": 3.0}
    excepted_res = 2**2 - 3.0
    assert np.isclose(my_func(params), excepted_res)

    computer = MyComputer()
    computer.static_meta_data = {"computer_tag": "dolphin"}
    quants = computer(params, ctx=abstract_objective_function.EvaluateContext())
    assert np.isclose(quants["res"], excepted_res)

    my_ob1 = abstract_objective_function.QuantityComputerObjectiveFunction(
        loss_function=loss1, quantity_computer=computer
    )

    assert np.isclose(my_ob1(params), excepted_res)

    my_ob2 = abstract_objective_function.QuantityComputerObjectiveFunction(
        loss_function=functools.partial(loss2, something_else=2.0),
        quantity_computer=computer,
    )

    assert np.isclose(my_ob2(params), excepted_res + 2.0)

    my_ob3 = abstract_objective_function.QuantityComputerObjectiveFunction(
        loss_function=loss3,
        quantity_computer=computer,
    )
    my_ob3.static_meta_data = {"ob_tag": "also_dolphin"}

    ctx = EvaluateContext()
    assert np.isclose(my_ob3(params, ctx), excepted_res + params["b"])

    meta_data = ctx.to_meta_data()

    meta_data_expected = {
        "quantities": {"res": 1.0},
        "parameters": {"a": 2.0, "b": 3.0},
        "loss": 4.0,
        "meta": {
            "computer_tag": "dolphin",
            "ob_tag": "also_dolphin",
            "meta_b2": params["b"] ** 2,
        },
    }

    for k, v in items_nested(meta_data):
        expected = get_nested(meta_data_expected, k)

        if isinstance(v, float):
            assert np.isclose(expected, v)
        else:
            assert v == expected


def test_context_stuff():
    # spawn children
    ctx = EvaluateContext()
    ctx.config.some_dict = {"bla": 3}
    children = ctx.spawn_children(3)

    # have the children spawn children
    [c.spawn_children(i) for i, c in enumerate(children)]
    ctx.collect_child_meta_data()

    # make sure all the copies of the dict are different entitites
    assert children[0].config.some_dict == children[1].config.some_dict
    children[0].config.some_dict["bla"] = (
        4  # this should only change the value of "bla" in the first child
    )
    assert (
        children[0].config.some_dict != children[1].config.some_dict
    )  # so these must be different now

    meta_data = ctx.to_meta_data()
    assert len(meta_data["meta"]["children"]) == 3

    for i, child in enumerate(meta_data["meta"]["children"]):
        if i == 0:  # the first child has no children
            assert "children" not in child["meta"]
        else:
            assert len(child["meta"]["children"]) == i


def test_async_evaluation():
    computer = MyComputer()
    computer.static_meta_data = {"computer_tag": "dolphin"}

    my_ob = abstract_objective_function.QuantityComputerObjectiveFunction(
        loss_function=loss3,
        quantity_computer=computer,
    )
    my_ob.static_meta_data = {"ob_tag": "also_dolphin"}

    n_terms = 10
    a_list = np.linspace(1, 5, n_terms)
    b_list = np.linspace(2, 7, n_terms)
    params = [{"a": a, "b": b} for a, b in zip(a_list, b_list, strict=True)]

    sync_results = [0.0] * n_terms

    for i, p in enumerate(params):
        sync_results[i] = my_ob(p)

    async_results = [0.0] * n_terms

    contexts = [abstract_objective_function.EvaluateContext() for _ in range(n_terms)]

    async_results = asyncio.run(
        async_helpers.async_eval_many(my_ob, params, ctxs=contexts)
    )

    print(f"{sync_results = }")
    print(f"{async_results = }")
    assert np.all(np.isclose(sync_results, async_results))


def test_quickstart():
    @to_quantity_computer()
    def computer(params: dict[str, float]) -> dict[str, float]:
        return {"x2": params["x"] ** 2, "y2": params["y"] ** 2}

    def loss(q: dict[str, float], target: float):
        return ((q["x2"] + q["y2"]) - target) ** 2

    TARGET = 2
    ob = computer.with_loss(functools.partial(loss, target=TARGET))

    PARAMS = {"x": 1.0, "y": 2.0}
    assert np.isclose(ob(PARAMS), (PARAMS["x"] ** 2 + PARAMS["y"] ** 2 - 2) ** 2)

    ctx = EvaluateContext()
    ob(PARAMS, ctx)
    print(ctx.to_meta_data())


def test_quickstart2():
    @to_quantity_computer()
    def computer(params: dict[str, float], f: float):
        return {"fx2": f * params["x"] ** 2, "fy2": f * params["y"] ** 2}

    def loss(q: dict[str, float], target: float) -> float:
        return (q["fx2"] + q["fy2"] - target) ** 2

    terms = [
        computer.bind(f=1).with_loss(loss, target=1),
        computer.bind(f=2).with_loss(loss, target=2),
    ]

    PARAMS = {"x": 1.0, "y": 2.0}
    combined = CombinedObjectiveFunction(terms)
    ctx = EvaluateContext()
    combined(PARAMS, ctx)
    print(ctx.to_meta_data())
