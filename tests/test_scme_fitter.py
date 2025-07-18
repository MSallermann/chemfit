from scme_fitting.fitter import Fitter

from scme_fitting.ase_objective_function import (
    EnergyObjectiveFunction,
    DimerDistanceObjectiveFunction,
)
from scme_fitting.scme_factories import (
    SCMECalculatorFactory,
    SCMEParameterApplier,
)

from scme_fitting.utils import dump_dict_to_file

from scme_fitting.multi_energy_objective_function import MultiEnergyObjectiveFunction
from scme_fitting.data_utils import process_csv
import logging
from pathlib import Path
from ase.units import Bohr, Hartree

logging.basicConfig(filename="./output/test_scme_fitter.log", level=logging.INFO)

### Common to all tests
PATH_TO_CSV = [
    Path(__file__).parent / "test_configurations_scme/energies.csv",
    Path(__file__).parent / "test_configurations_scme/energies2.csv",
]
REFERENCE_CONFIGS, TAGS, REFERENCE_ENERGIES = process_csv(PATH_TO_CSV)


DEFAULT_PARAMS = {
    "dispersion": {
        "td": 7.5548 * Bohr,
        "rc": 8.0 / Bohr,
        "C6": 46.4430e0,
        "C8": 1141.7000e0,
        "C10": 33441.0000e0,
    },
    "repulsion": {
        "Ar_OO": 8149.63 / Hartree,
        "Br_OO": -0.5515,
        "Cr_OO": -3.4695 * Bohr,
        "r_Br": 1.0 / Bohr,
        "td": 7.5548 * Bohr,
        "rc": 7.5 / Bohr,
    },
    "electrostatic": {
        "te": 1.2 / Bohr,
        "rc": 9.0 / Bohr,
        "NC": [1, 2, 1],
        "scf_convcrit": 1e-8,
        "max_iter_scf": 500,
    },
    "dms": True,
    "qms": True,
}

INITIAL_PARAMS = {
    "repulsion": {"td": 2.0},
    "electrostatic": {"te": 2.0},
    "dispersion": {
        "C6": 40.0,
        "C8": 800,
        "C10": 30000,
    },
}


def test_single_energy_objective_function():
    scme_factories = EnergyObjectiveFunction(
        calc_factory=SCMECalculatorFactory(DEFAULT_PARAMS, None, None),
        param_applier=SCMEParameterApplier(),
        path_to_reference_configuration=REFERENCE_CONFIGS[10],
        reference_energy=REFERENCE_ENERGIES[10],
        tag=TAGS[10],
    )

    fitter = Fitter(objective_function=scme_factories, initial_params=INITIAL_PARAMS)

    optimal_params = fitter.fit_scipy(tol=1e-4, options=dict(maxiter=50, disp=True))

    output_folder = Path(__file__).parent / "output/single_energy"

    scme_factories.dump_test_configuration(output_folder)

    dump_dict_to_file(output_folder / "optimal_params.json", optimal_params)


def test_dimer_distance_objective_function():
    scme_factories = DimerDistanceObjectiveFunction(
        calc_factory=SCMECalculatorFactory(DEFAULT_PARAMS, None, None),
        param_applier=SCMEParameterApplier(),
        path_to_reference_configuration=REFERENCE_CONFIGS[5],
        dt=1.0,
        max_steps=500,
        reference_OO_distance=3.2,
        tag="dimer_distance",
    )

    fitter = Fitter(objective_function=scme_factories, initial_params=INITIAL_PARAMS)

    optimal_params = fitter.fit_scipy(tol=1e-4, options=dict(maxiter=50, disp=True))

    output_folder = Path(__file__).parent / "output/dimer_distance"
    scme_factories.dump_test_configuration(output_folder)

    dump_dict_to_file(output_folder / "optimal_params.json", optimal_params)


def test_multi_energy_ob_function_fitting():
    scme_factories = MultiEnergyObjectiveFunction(
        calc_factory=SCMECalculatorFactory(DEFAULT_PARAMS, None, None),
        param_applier=SCMEParameterApplier(),
        path_or_factory_list=REFERENCE_CONFIGS,
        reference_energy_list=REFERENCE_ENERGIES,
        tag_list=TAGS,
    )

    fitter = Fitter(objective_function=scme_factories, initial_params=INITIAL_PARAMS)

    optimal_params = fitter.fit_scipy(tol=0, options=dict(maxiter=50, disp=True))

    output_folder = Path(__file__).parent / "output/multi_energy"

    scme_factories.write_output(
        output_folder,
        initial_params=INITIAL_PARAMS,
        optimal_params=optimal_params,
    )


if __name__ == "__main__":
    # test_single_energy_objective_function()
    # test_dimer_distance_objective_function()
    test_multi_energy_ob_function_fitting()
