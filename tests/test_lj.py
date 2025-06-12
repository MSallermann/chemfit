from ase.io import write
from ase.calculators.lj import LennardJones
from ase import Atoms
import numpy as np
from pathlib import Path

from scme_fitting.multi_energy_objective_function import MultiEnergyObjectiveFunction
from scme_fitting.fitter import Fitter


def e_lj(r, eps, sigma):
    return 4.0 * eps * ((sigma / r) ** 6 - 1.0) * (sigma / r) ** 6


def prepare_data(r_list: list[float], output_folder: Path, eps: float, sigma: float):
    p0 = np.zeros(3)

    output_folder.mkdir(exist_ok=True)

    paths = []
    tags = []
    energies = []

    for r in r_list:
        p1 = np.array([r, 0.0, 0.0])
        atoms = Atoms(positions=[p0, p1])
        energy = e_lj(r, eps=eps, sigma=sigma)

        p = output_folder / f"atoms_lj_{r:.2f}.xyz"
        write(p, atoms)

        paths.append(p)
        tags.append(f"lj_{r:.2f}")
        energies.append(energy)

    return paths, tags, energies


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

    output_folder = Path("./test_lj")

    paths, tags, energies = prepare_data(r_list, output_folder, eps=eps, sigma=sigma)

    ob = MultiEnergyObjectiveFunction(
        calc_factory=construct_lj,
        param_applier=apply_params_lj,
        tag_list=tags,
        path_or_factory_list=paths,
        reference_energy_list=energies,
    )

    fitter = Fitter(ob)

    initial_params = {"epsilon": 2.0, "sigma": 1.5}

    opt_params = fitter.fit_scipy(initial_params, options=dict(disp=True, tol=1e-5))

    print(opt_params)

    ob.write_output(
        "test_lj_output",
        initial_params=initial_params,
        optimal_params=opt_params,
    )

    assert np.isclose(opt_params["epsilon"], eps)
    assert np.isclose(opt_params["sigma"], sigma)


if __name__ == "__main__":
    test_lj()
