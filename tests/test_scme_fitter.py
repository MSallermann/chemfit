from scme_fitting.fitter import Fitter
from scme_fitting.scme_setup import SCMEParams
from scme_fitting.scme_energy_objective_function import SCMEEnergyObjectiveFunction
from scme_fitting.combined_objective_function import CombinedObjectiveFunction

import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt


logging.basicConfig(filename="test_scme_fitter.log", level=logging.INFO)


def create_scme_fit_data(base_path: Path):
    energies = np.loadtxt(base_path / "PES_dimer_c1_PBE.txt")[:, 1]
    paths = list(base_path.glob("*/CONTCAR"))
    sorted_paths = sorted(paths, key=lambda p: float(p.parent.name))
    tags = [p.parent.name for p in sorted_paths]
    return sorted_paths, energies, tags


def test_scme_fitting():
    base_path = Path(
        "/home/moritz/SCME/generalized_SCME_interatomic_fit/SCMEFitting/scme_fitting/resources/PBE"
    )
    paths_to_reference_configurations, reference_energies, tags = create_scme_fit_data(
        base_path
    )

    scme_objective_functions = [
        SCMEEnergyObjectiveFunction(
            default_scme_params=SCMEParams(),
            path_to_scme_expansions=None,
            parametrization_key=None,
            path_to_reference_configuration=xyz_file,
            reference_energy=energy,
            divide_by_n_atoms=True,
            tag=tag,
        )
        for xyz_file, energy, tag in zip(
            paths_to_reference_configurations, reference_energies, tags
        )
    ]

    objective_function = CombinedObjectiveFunction(
        objective_functions=scme_objective_functions
    )

    DEFAULT_PARAMS = SCMEParams()
    ADJUSTABLE_PARAMS = ["td", "Ar_OO", "Br_OO", "Cr_OO", "r_Br"]

    [
        o.dump_test_configuration("test_configurations_scme")
        for o in scme_objective_functions
    ]

    fitter = Fitter(
        objective_function=objective_function,
    )

    initial_params = {k: dict(DEFAULT_PARAMS)[k] for k in ADJUSTABLE_PARAMS}

    optimal_params = fitter.fit_scipy(
        initial_parameters=initial_params, tol=0, options=dict(maxiter=50, disp=True)
    )

    print(f"{initial_params = }")
    print(f"{optimal_params = }")

    plt.plot(reference_energies, label="reference")
    plt.plot(
        [o.get_energy(initial_params) for o in scme_objective_functions],
        label="initial parameters",
    )
    plt.plot(
        [o.get_energy(optimal_params) for o in scme_objective_functions],
        label="fitted parameters",
    )
    plt.legend()
    plt.savefig("fig.png")


if __name__ == "__main__":
    test_scme_fitting()
