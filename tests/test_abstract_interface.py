from __future__ import annotations

import asyncio
import functools
import random
import time

import numpy as np
from pydictnest import get_nested, items_nested

from chemfit import abstract_objective_function, async_helpers
from chemfit.abstract_objective_function import EvaluateContext


class MyFunctor(abstract_objective_function.ObjectiveFunctor):
    def __call__(
        self,
        parameters: dict[str, float],
        ctx: EvaluateContext | None = None,  # noqa: ARG002
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
    ctx.temp.some_dict = {"bla": 3}
    children = ctx.spawn_children(3)

    # have the children spawn children
    [c.spawn_children(i) for i, c in enumerate(children)]
    ctx.collect_child_meta_data()

    # make sure all the copies of the dict are different entitites
    assert children[0].temp.some_dict == children[1].temp.some_dict
    children[0].temp.some_dict["bla"] = (
        4  # this should only change the value of "bla" in the first child
    )
    assert (
        children[0].temp.some_dict != children[1].temp.some_dict
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
    params = [{"a": a, "b": b} for a, b in zip(a_list, b_list)]

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
