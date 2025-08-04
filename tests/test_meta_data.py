from chemfit.abstract_objective_function import ObjectiveFunctor
from chemfit.combined_objective_function import CombinedObjectiveFunction

try:
    import mpi4py
except ImportError:
    mpi4py = None

import pytest


class MyFunctor(ObjectiveFunctor):
    def __init__(self, f: float) -> None:
        """Initialize My Functor."""
        self.f = f
        self.meta_data = {}

    def get_meta_data(self):
        return self.meta_data

    def __call__(self, params: dict) -> float:
        val = self.f * params["x"] ** 2
        self.meta_data["last_value"] = val
        return val


def a(p: dict):
    return p["y"] ** 2


INITIAL_PARAMS = {"x": 1.0, "y": 2.0}


def test_gather_meta_data():
    cob = CombinedObjectiveFunction(
        [a, MyFunctor(1), MyFunctor(2)]
    )  # is equivalent to y**2 + x**2 + 2.0*x**2

    # Evaluate the objective function
    cob(INITIAL_PARAMS)
    meta_data = cob.gather_meta_data()

    expected = [None, {"last_value": 1.0}, {"last_value": 2.0}]

    assert meta_data == expected


@pytest.mark.skipif(mpi4py is None, reason="Cannot import mpi4py")
def test_gather_meta_data_mpi():
    import logging

    from chemfit.mpi_wrapper_cob import MPIWrapperCOB

    cob = CombinedObjectiveFunction(
        [a, MyFunctor(1), MyFunctor(2)]
    )  # is equivalent to y**2 + x**2 + 2.0*x**2

    # Use the MPI Wrapper to make the combined objective function "MPI aware"
    with MPIWrapperCOB(cob, mpi_debug_log=False) as ob_mpi:
        logging.basicConfig(
            filename=f"meta_data_{ob_mpi.rank}.log",
            filemode="w",
            force=False,
            level=logging.INFO,
        )
        if ob_mpi.rank == 0:
            ob_mpi(INITIAL_PARAMS)
            meta_data = ob_mpi.gather_meta_data()
            print(f"{meta_data = }")
            expected = [None, {"last_value": 1.0}, {"last_value": 2.0}]
            assert meta_data == expected
        else:
            ob_mpi.worker_loop()
