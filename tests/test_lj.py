from ase.calculators.lj import LennardJones
from ase import Atoms
import numpy as np

from scme_fitting.multi_energy_objective_function import MultiEnergyObjectiveFunction
from scme_fitting.fitter import Fitter


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

    ob.write_output(
        "./output/lj",
        initial_params=initial_params,
        optimal_params=opt_params,
    )

    assert np.isclose(opt_params["epsilon"], eps)
    assert np.isclose(opt_params["sigma"], sigma)


if __name__ == "__main__":
    test_lj()
