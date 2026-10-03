import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.lj import LennardJones

from chemfit.abstract_objective_function import EvaluateContext


def e_lj(r: float, eps: float, sigma: float) -> float:
    return 4.0 * eps * ((sigma / r) ** 6 - 1.0) * (sigma / r) ** 6


class LJAtomsFactory:
    def __init__(self, r: float) -> None:
        """Construct two atoms at a distance r."""
        self.p0 = np.zeros(3)
        self.p1 = np.array([r, 0.0, 0.0])

    def __call__(self) -> Atoms:
        return Atoms(positions=[self.p0, self.p1])


def construct_lj(
    parameters: dict[str, float],
    _atoms: Atoms,
    _ctx: EvaluateContext,
) -> Calculator:
    return LennardJones(
        rc=2000,
        sigma=parameters["sigma"],
        epsilon=parameters["epsilon"],
    )
