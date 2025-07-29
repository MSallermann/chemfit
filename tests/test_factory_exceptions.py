import pytest

try:
    import mpi4py

    rank = mpi4py.MPI.COMM_WORLD.Get_rank()
except ImportError:
    mpi4py = None
    rank = 0

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
    bad_atoms_factory: tuple[bool, bool] = (False, False),
    n_terms: int = 2,
) -> MultiEnergyObjectiveFunction:

    r_min = 2 ** (1 / 6) * sigma
    r_list = np.linspace(0.925 * r_min, 3.0 * sigma, num=n_terms)

    ob = MultiEnergyObjectiveFunction(
        calc_factory=construct_lj,
        param_applier=apply_params_lj,
        tag_list=[f"lj_{r:.2f}" for r in r_list],
        reference_energy_list=[e_lj(r, eps, sigma) for r in r_list],
        path_or_factory_list=[
            construct_atoms_bad if bad else LJAtomsFactory(r)
            for r, bad in zip(r_list, bad_atoms_factory)
        ],
    )

    return ob


### Construct the objective function on *all* ranks
EPS = 1.0
SIGMA = 1.0
INITIAL_PARAMS = {"epsilon": 2.0, "sigma": 1.5}


@pytest.mark.skipif(mpi4py is None, reason="Cannot import mpi4py")
def test_exceptions_mpi_all_good():
    from chemfit.mpi_wrapper_cob import MPIWrapperCOB
    import logging
    from mpi4py import MPI

    logging.basicConfig(
        filename=f"text_exceptions_rank_{MPI.COMM_WORLD.Get_rank()}.log",
        level=logging.INFO,
    )

    ###################################
    # Case 0: all good
    ###################################
    ob_good = get_ob_func(
        EPS,
        SIGMA,
        bad_atoms_factory=(False, False),
    )

    with MPIWrapperCOB(ob_good, mpi_debug_log=True) as ob_mpi:
        # The optimization needs to run on the first rank only
        if ob_mpi.rank == 0:
            fitter = Fitter(ob_mpi, initial_params=INITIAL_PARAMS)
            fitter.fit_nevergrad(budget=1)
        else:
            ob_mpi.worker_loop()


@pytest.mark.skipif(mpi4py is None, reason="Cannot import mpi4py")
def test_exceptions_mpi_all_bad():
    from chemfit.mpi_wrapper_cob import MPIWrapperCOB
    import logging
    from mpi4py import MPI

    logging.basicConfig(
        filename=f"text_exceptions_rank_{MPI.COMM_WORLD.Get_rank()}.log",
        level=logging.INFO,
    )

    ###################################
    # Case 1: all bad
    ###################################
    ob_bad = get_ob_func(
        EPS,
        SIGMA,
        bad_atoms_factory=(True, True),
    )

    with pytest.raises(FactoryException):
        with MPIWrapperCOB(ob_bad, mpi_debug_log=True) as ob_mpi:
            # The optimization needs to run on the first rank only
            if ob_mpi.rank == 0:
                fitter = Fitter(ob_mpi, initial_params=INITIAL_PARAMS)
                fitter.fit_nevergrad(budget=1)
            else:
                ob_mpi.worker_loop()


@pytest.mark.xfail(rank != 1, reason="Can only raise exception on rank 1")
@pytest.mark.skipif(mpi4py is None, reason="Cannot import mpi4py")
def test_exceptions_mpi_good_master_bad_worker():
    from chemfit.mpi_wrapper_cob import MPIWrapperCOB
    import logging
    from mpi4py import MPI

    logging.basicConfig(
        filename=f"text_exceptions_rank_{MPI.COMM_WORLD.Get_rank()}.log",
        level=logging.INFO,
    )

    ####################################
    # Case 2: good master bad worker
    ####################################
    ob_good_master_bad_worker = get_ob_func(
        EPS,
        SIGMA,
        bad_atoms_factory=(False, True),
    )

    with pytest.raises(FactoryException):
        with MPIWrapperCOB(ob_good_master_bad_worker, mpi_debug_log=True) as ob_mpi:
            # The optimization needs to run on the first rank only
            if ob_mpi.rank == 0:
                fitter = Fitter(ob_mpi, initial_params=INITIAL_PARAMS)
                fitter.fit_nevergrad(budget=1)
            else:
                ob_mpi.worker_loop()


@pytest.mark.xfail(rank != 0, reason="Can only raise exception on rank 0")
@pytest.mark.skipif(mpi4py is None, reason="Cannot import mpi4py")
def test_exceptions_mpi_bad_master_good_worker():
    from chemfit.mpi_wrapper_cob import MPIWrapperCOB

    import logging
    from mpi4py import MPI

    logging.basicConfig(
        filename=f"text_exceptions_rank_{MPI.COMM_WORLD.Get_rank()}.log",
        level=logging.INFO,
        force=True,
    )

    ####################################
    # Case 3: bad master good worker
    ####################################
    ob_bad_master_good_worker = get_ob_func(
        EPS,
        SIGMA,
        bad_atoms_factory=(True, False),
    )

    with pytest.raises(FactoryException) as exc_info:
        with MPIWrapperCOB(ob_bad_master_good_worker, mpi_debug_log=True) as ob_mpi:
            # The optimization needs to run on the first rank only
            if ob_mpi.rank == 0:
                fitter = Fitter(ob_mpi, initial_params=INITIAL_PARAMS)
                fitter.fit_nevergrad(budget=1)
            else:
                ob_mpi.worker_loop()

    print(f"[Rank {rank}] {exc_info = }")


if __name__ == "__main__":
    test_exceptions_mpi_bad_master_good_worker()
