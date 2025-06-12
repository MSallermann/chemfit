from scme_fitting.fitter import Fitter
from scme_fitting.scme_setup import SCMEParams

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


logging.basicConfig(filename="test_scme_fitter.log", level=logging.DEBUG)

### Common to all tests
PATH_TO_CSV = Path(__file__).parent / "test_configurations_scme/energies.csv"
REFERENCE_CONFIGS, TAGS, REFERENCE_ENERGIES = process_csv(PATH_TO_CSV)

DEFAULT_PARAMS = SCMEParams(
    td=4.7,
    Ar_OO=299.5695377280358,
    Br_OO=-0.14632711560656822,
    Cr_OO=-2.0071714442805715,
    r_Br=5.867230272424719,
    dms=True,
    qms=True,
)

ADJUSTABLE_PARAMS = ["td", "te", "C6", "C8", "C10"]
INITIAL_PARAMS = {k: dict(DEFAULT_PARAMS)[k] for k in ADJUSTABLE_PARAMS}


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

    output_folder = Path("test_output_single_energy")
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

    output_folder = Path("test_output_dimer_distance")
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

    scme_factories.write_output(
        "test_output_multi_energy",
        initial_params=INITIAL_PARAMS,
        optimal_params=optimal_params,
    )


if __name__ == "__main__":
    # test_single_energy_objective_function()
    # test_dimer_distance_objective_function()
    test_multi_energy_ob_function_fitting()
