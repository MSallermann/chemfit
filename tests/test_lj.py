try:
    import mpi4py

    from chemfit.mpi_wrapper_cob import MPIWrapperCOB

except ImportError:
    mpi4py = None

import functools

import numpy as np
import pytest
from conftest import LJAtomsFactory, apply_params_lj, construct_lj, e_lj

from chemfit.abstract_objective_function import QuantityComputerObjectiveFunction
from chemfit.ase_objective_function import SinglePointASEComputer
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.fitter import Fitter


class LJComputer(SinglePointASEComputer):
    def __init__(self, r: float):  # noqa: D107
        super().__init__(
            calc_factory=construct_lj,
            param_applier=apply_params_lj,
            atoms_factory=LJAtomsFactory(r),
            tag=f"lj_{r:.2f}",
        )


def loss_function(quants: dict, e_ref: float):
    return (quants["energy"] - e_ref) ** 2


def get_ob_func(eps: float, sigma: float) -> CombinedObjectiveFunction:
    r_min = 2.0 ** (1 / 6) * sigma
    r_list = np.linspace(0.925 * r_min, 3.0 * sigma)

    return CombinedObjectiveFunction(
        objective_functions=[
            QuantityComputerObjectiveFunction(
                functools.partial(loss_function, e_ref=e_lj(r, eps, sigma)),
                quantity_computer=LJComputer(r),
            )
            for r in r_list
        ]
    )


def test_lj():
    eps = 1.0
    sigma = 1.0

    ob = get_ob_func(eps, sigma)

    initial_params = {"epsilon": 2.0, "sigma": 1.5}

    fitter = Fitter(ob, initial_params=initial_params)
    opt_params = fitter.fit_scipy()

    ob.gather_meta_data()
    meta_data = ob.gather_meta_data()

    assert ob.n_terms() == len(meta_data)
    assert np.isclose(opt_params["epsilon"], eps)
    assert np.isclose(opt_params["sigma"], sigma)


@pytest.mark.skipif(mpi4py is None, reason="Cannot import mpi4py")
def test_lj_mpi():
    ### Construct the objective function on *all* ranks
    eps = 1.0
    sigma = 1.0

    ob = get_ob_func(eps, sigma)

    initial_params = {"epsilon": 2.0, "sigma": 1.5}

    # Use the MPI Wrapper to make the combined objective function "MPI aware"
    with MPIWrapperCOB(ob) as mpi:
        if mpi.rank == 0:
            fitter = Fitter(mpi, initial_params=initial_params)
            opt_params = fitter.fit_scipy()
            meta_data = mpi.gather_meta_data()

            assert ob.n_terms() == len(meta_data)
            assert np.isclose(opt_params["epsilon"], eps)
            assert np.isclose(opt_params["sigma"], sigma)
        else:
            mpi.worker_loop()


if __name__ == "__main__":
    import logging

    logging.basicConfig(filename="test_lj.log")

    # test_lj()
    test_lj_mpi()
