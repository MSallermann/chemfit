try:
    import mpi4py
except ImportError:
    mpi4py = None

import pytest

from ase.calculators.lj import LennardJones
from ase import Atoms
import numpy as np

from chemfit.multi_energy_objective_function import MultiEnergyObjectiveFunction
from chemfit.fitter import Fitter
from pathlib import Path


class LJAtomsFactory:
    def __init__(self, r: float):
        p0 = np.zeros(3)
        p1 = np.array([r, 0.0, 0.0])
        self.atoms = Atoms(positions=[p0, p1])

    def __call__(self):
        return self.atoms


def e_lj(r, eps, sigma):
    return 4.0 * eps * ((sigma / r) ** 6 - 1.0) * (sigma / r) ** 6


def construct_lj(atoms: Atoms):
    atoms.calc = LennardJones(rc=2000)


def apply_params_lj(atoms: Atoms, params: dict[str, float]):
    atoms.calc.parameters.sigma = params["sigma"]
    atoms.calc.parameters.epsilon = params["epsilon"]


def test_lj():
    eps = 1.0
    sigma = 1.0

    r_min = 2 ** (1 / 6) * sigma
    r_list = np.linspace(0.925 * r_min, 3.0 * sigma)

    ob = MultiEnergyObjectiveFunction(
        calc_factory=construct_lj,
        param_applier=apply_params_lj,
        tag_list=[f"lj_{r:.2f}" for r in r_list],
        reference_energy_list=[e_lj(r, eps, sigma) for r in r_list],
        path_or_factory_list=[LJAtomsFactory(r) for r in r_list],
    )

    initial_params = {"epsilon": 2.0, "sigma": 1.5}

    fitter = Fitter(ob, initial_params=initial_params)
    opt_params = fitter.fit_scipy(options=dict(disp=True))

    print(opt_params)

    output_folder = Path(__file__).parent / "output/lj"

    ob.write_output(
        output_folder,
        initial_params=initial_params,
        optimal_params=opt_params,
    )

    assert np.isclose(opt_params["epsilon"], eps)
    assert np.isclose(opt_params["sigma"], sigma)


@pytest.mark.skipif(mpi4py is None, reason="Cannot import mpi4py")
def test_lj_mpi():
    from chemfit import HAS_MPI

    if not HAS_MPI:
        return

    from chemfit.mpi_wrapper_cob import MPIWrapperCOB

    ### Construct the objective function on *all* ranks
    eps = 1.0
    sigma = 1.0

    r_min = 2 ** (1 / 6) * sigma
    r_list = np.linspace(0.925 * r_min, 3.0 * sigma)

    ob = MultiEnergyObjectiveFunction(
        calc_factory=construct_lj,
        param_applier=apply_params_lj,
        tag_list=[f"lj_{r:.2f}" for r in r_list],
        reference_energy_list=[e_lj(r, eps, sigma) for r in r_list],
        path_or_factory_list=[LJAtomsFactory(r) for r in r_list],
    )

    # Use the MPI Wrapper to make the combined objective function "MPI aware"
    with MPIWrapperCOB(ob) as ob_mpi:
        # The optimization needs to run on the first rank only
        if ob_mpi.rank == 0:
            initial_params = {"epsilon": 2.0, "sigma": 1.5}
            bounds = {"epsilon": (0.1, 10), "sigma": (0.5, 3.0)}
            fitter = Fitter(ob_mpi, initial_params=initial_params, bounds=bounds)
            # opt_params = fitter.fit_scipy(options=dict(disp=True))
            opt_params = fitter.fit_nevergrad(budget=100)

            output_folder = Path(__file__).parent / "output/lj_mpi"

            ob.write_output(
                output_folder,
                initial_params=initial_params,
                optimal_params=opt_params,
            )

            assert np.isclose(opt_params["epsilon"], eps)
            assert np.isclose(opt_params["sigma"], sigma)


if __name__ == "__main__":
    # test_lj()
    test_lj_mpi()
