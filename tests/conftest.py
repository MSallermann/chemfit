try:
    import mpi4py
except ImportError:
    mpi4py = None

import pytest
from ase.calculators.lj import LennardJones
from ase import Atoms
import numpy as np


def e_lj(r, eps, sigma):
    return 4.0 * eps * ((sigma / r) ** 6 - 1.0) * (sigma / r) ** 6


@pytest.fixture(scope="session", autouse=True)
def mpi_finalize_fixture():
    yield

    if mpi4py is not None:
        # This teardown part runs after the session
        if not mpi4py.MPI.Is_finalized():
            mpi4py.MPI.Finalize()


class LJAtomsFactory:
    def __init__(self, r: float):
        p0 = np.zeros(3)
        p1 = np.array([r, 0.0, 0.0])
        self.atoms = Atoms(positions=[p0, p1])

    def __call__(self):
        return self.atoms


def construct_lj(atoms: Atoms):
    atoms.calc = LennardJones(rc=2000)


def apply_params_lj(atoms: Atoms, params: dict[str, float]):
    atoms.calc.parameters.sigma = params["sigma"]
    atoms.calc.parameters.epsilon = params["epsilon"]
