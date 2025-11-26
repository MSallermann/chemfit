from __future__ import annotations

import pytest

from chemfit.abstract_objective_function import (
    EvaluateContext,
    ObjectiveFunctor,
    QuantityComputerObjectiveFunction,
)
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.wrap_funcs import to_objective_functor, to_quantity_computer


class MyFunctor(ObjectiveFunctor):
    def __init__(self, f: float) -> None:
        """Initialize My Functor."""
        self.f = f
        self.meta_data = {}

    def __call__(self, parameters: dict, ctx: EvaluateContext | None = None) -> float:
        if ctx is None:
            ctx = EvaluateContext()

        val = self.f * parameters["x"] ** 2
        ctx.meta["last_value"] = val
        return val


@to_objective_functor
def a(p: dict):
    return p["y"] ** 2


@to_quantity_computer
def quants(p: dict):
    return {"x_plus_y": p["x"] + p["y"]}


def loss(q: dict, p: dict):
    return q["x_plus_y"] + p["y"]


INITIAL_PARAMS = {"x": 1.0, "y": 2.0}

COB = CombinedObjectiveFunction(
    [
        a,
        MyFunctor(1),
        QuantityComputerObjectiveFunction(loss_function=loss, quantity_computer=quants),
    ]
)

EXPECTED = [
    {
        "quantities": None,
        "parameters": {"x": 1.0, "y": 2.0},
        "loss": 4.0,
        "meta": {},
    },
    {
        "quantities": None,
        "parameters": None,
        "loss": None,
        "meta": {"last_value": 1.0},
    },
    {
        "quantities": {"x_plus_y": 3.0},
        "parameters": {"x": 1.0, "y": 2.0},
        "loss": 5.0,
        "meta": {},
    },
]


def test_gather_meta_data():
    # Evaluate the objective function
    COB(INITIAL_PARAMS, ctx := EvaluateContext())
    meta_data = ctx.to_meta_data()["meta"]["cob_terms"]

    print(f"{meta_data = }")
    print(f"{EXPECTED = }")

    assert meta_data == EXPECTED


def test_gather_meta_data_mpi():
    mpi_wrapper_cob = pytest.importorskip("chemfit.mpi_wrapper_cob")

    # Use the MPI Wrapper to make the combined objective function "MPI aware"
    with mpi_wrapper_cob.MPIWrapperCOB(COB, mpi_debug_log=False) as ob_mpi:
        if ob_mpi.rank == 0:
            ob_mpi(INITIAL_PARAMS, ctx := EvaluateContext())
            meta_data = ctx.to_meta_data()["meta"]["cob_terms"]

            print(f"{meta_data = }")
            print(f"{EXPECTED = }")

            assert meta_data == EXPECTED

        else:
            ob_mpi.worker_loop()
