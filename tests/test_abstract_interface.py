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
    def _compute(self, parameters: dict[str, float]) -> dict[str, float]:
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
    quants = computer(params)
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

    assert np.isclose(my_ob3(params), excepted_res + params["b"])

    meta_data = my_ob3.get_meta_data()

    meta_data_expected = {
        "computer": {"last": {"res": 1.0}, "last_params": {"a": 2.0, "b": 3.0}},
        "last_loss": 4.0,
    }

    for k, v in items_nested(meta_data):
        assert np.isclose(get_nested(meta_data_expected, k), v)
