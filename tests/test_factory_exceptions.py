import pytest

try:
    import mpi4py
except ImportError:
    mpi4py = None

from chemfit.fitter import Fitter
from chemfit.exceptions import FactoryException
from chemfit.multi_energy_objective_function import MultiEnergyObjectiveFunction
from ase import Atoms
from conftest import construct_lj, apply_params_lj, LJAtomsFactory, e_lj
import numpy as np


def construct_atoms_bad():
    raise Exception("Construct atoms exception")


def construct_calc_bad(atoms):
    raise Exception("Construct calc exception")


def apply_params_bad(atoms: Atoms, params: dict[str, float]):
    raise Exception("Apply params exception")


def get_ob_func(
    eps: float,
    sigma: float,
    bad_calc_factory: bool = False,
    bad_atoms_factory: bool = False,
    bad_param_applier: bool = False,
) -> MultiEnergyObjectiveFunction:

    r_min = 2 ** (1 / 6) * sigma
    r_list = np.linspace(0.925 * r_min, 3.0 * sigma)

    ob = MultiEnergyObjectiveFunction(
        calc_factory=construct_calc_bad if bad_calc_factory else construct_lj,
        param_applier=apply_params_bad if bad_param_applier else apply_params_lj,
        tag_list=[f"lj_{r:.2f}" for r in r_list],
        reference_energy_list=[e_lj(r, eps, sigma) for r in r_list],
        path_or_factory_list=[
            construct_atoms_bad if bad_atoms_factory else LJAtomsFactory(r)
            for r in r_list
        ],
    )

    return ob


@pytest.mark.skipif(mpi4py is None, reason="Cannot import mpi4py")
def test_exceptions_mpi():
    from chemfit.mpi_wrapper_cob import MPIWrapperCOB
    import logging
    from mpi4py import MPI

    logging.basicConfig(
        filename=f"text_exceptions_rank_{MPI.COMM_WORLD.Get_rank()}.log",
        level=logging.INFO,
    )

    ### Construct the objective function on *all* ranks
    eps = 1.0
    sigma = 1.0

    ob = get_ob_func(
        eps,
        sigma,
        bad_calc_factory=True,
        bad_atoms_factory=True,
        bad_param_applier=True,
    )

    initial_params = {"epsilon": 2.0, "sigma": 1.5}

    # Use the MPI Wrapper to make the combined objective function "MPI aware"
    # Note: we set finalize_mpi to False, because we use a session-scoped fixture to finalize MPI instead
    with MPIWrapperCOB(ob, finalize_mpi=False) as ob_mpi:

        with pytest.raises(FactoryException) as excinfo:
            # The optimization needs to run on the first rank only
            if ob_mpi.rank == 0:
                fitter = Fitter(ob_mpi, initial_params=initial_params)
                fitter.fit_nevergrad(budget=1)
            else:
                ob_mpi.worker_loop()


if __name__ == "__main__":
    test_exceptions_mpi()
