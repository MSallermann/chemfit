import functools

import numpy as np
from pydictnest import get_nested, items_nested

from chemfit import abstract_objective_function


class MyFunctor(abstract_objective_function.ObjectiveFunctor):
    def __call__(self, parameters: dict[str, float]) -> float:
        return parameters["a"] ** 2 - parameters["b"]

    def get_meta_data(self) -> dict[str, float]:
        return {}


class MyComputer(abstract_objective_function.QuantityComputer):
    def _compute(
        self,
        parameters: dict[str, float],
        ctx: abstract_objective_function.EvaluateContext,  # noqa: ARG002
    ) -> dict[str, float]:
        return {"res": parameters["a"] ** 2 - parameters["b"]}


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
    quants = computer.evaluate(
        params, ctx=abstract_objective_function.EvaluateContext()
    )
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

    assert np.isclose(my_ob3(params), excepted_res + params["b"])

    meta_data = my_ob3.get_meta_data()

    meta_data_expected = {
        "quantities": {"res": 1.0},
        "parameters": {"a": 2.0, "b": 3.0},
        "loss": 4.0,
        "static": {"computer_tag": "dolphin", "ob_tag": "also_dolphin"},
    }

    for k, v in items_nested(meta_data):
        expected = get_nested(meta_data_expected, k)

        if isinstance(v, float):
            assert np.isclose(expected, v)
        else:
            assert v == expected
