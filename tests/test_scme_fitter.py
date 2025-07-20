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
from ase.units import Bohr
import numpy as np
import time

logging.basicConfig(filename="./output/test_scme_fitter.log", level=logging.INFO)

### Common to all tests
PATH_TO_CSV = [
    Path(__file__).parent / "test_configurations_scme/energies.csv",
    Path(__file__).parent / "test_configurations_scme/energies2.csv",
]
REFERENCE_CONFIGS, TAGS, REFERENCE_ENERGIES = process_csv(PATH_TO_CSV)

DEFAULT_PARAMS = {
    "dispersion": {
        "td": 4.7,
        "rc": 8.0 / Bohr,
        "C6": 46.4430e0,
        "C8": 1141.7000e0,
        "C10": 33441.0000e0,
    },
    "repulsion": {
        "Ar_OO": 299.5695377280358,
        "Br_OO": -0.14632711560656822,
        "Cr_OO": -2.0071714442805715,
        "r_Br": 5.867230272424719,
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
    "electrostatic": {"te": 2.0},
    "dispersion": {
        "td": 4.7,
        "C6": 46.4430e0,
        "C8": 1141.7000e0,
        "C10": 33441.0000e0,
    },
}


def test_factories():
    from ase import Atoms

    atoms = Atoms()
    calc_factory = SCMECalculatorFactory(DEFAULT_PARAMS, None, None)
    calc_factory(atoms)

    def check_if_params_applied(params: dict):
        for k, v_in in params.get("dispersion", {}).items():
            v_out = getattr(atoms.calc.scme.dispersion_params, k)
            print(k, v_in, v_out)
            assert np.all(np.isclose(v_in, v_out))

        for k, v_in in params.get("repulsion", {}).items():
            v_out = getattr(atoms.calc.scme.repulsion_params, k)
            print(k, v_in, v_out)
            assert np.all(np.isclose(v_in, v_out))

        for k, v_in in params.get("electrostatic", {}).items():
            v_out = getattr(atoms.calc.scme.electrostatic_params, k)
            print(k, v_in, v_out)
            assert np.all(np.isclose(v_in, v_out))

        for k, v_in in params.items():
            if isinstance(v_in, dict):
                continue
            v_out = getattr(atoms.calc.scme, k)
            assert np.all(np.isclose(v_in, v_out))

    check_if_params_applied(DEFAULT_PARAMS)

    param_applier = SCMEParameterApplier()
    param_applier(atoms, INITIAL_PARAMS)

    check_if_params_applied(INITIAL_PARAMS)


def test_single_energy_objective_function():
    ob = EnergyObjectiveFunction(
        calc_factory=SCMECalculatorFactory(DEFAULT_PARAMS, None, None),
        param_applier=SCMEParameterApplier(),
        path_to_reference_configuration=REFERENCE_CONFIGS[10],
        reference_energy=REFERENCE_ENERGIES[10],
        tag=TAGS[10],
    )

    fitter = Fitter(objective_function=ob, initial_params=INITIAL_PARAMS)

    optimal_params = fitter.fit_scipy(tol=1e-4, options=dict(maxiter=50, disp=True))

    output_folder = Path(__file__).parent / "output/single_energy"

    dump_dict_to_file(output_folder / "optimal_params.json", optimal_params)


def test_dimer_distance_objective_function():
    ob = DimerDistanceObjectiveFunction(
        calc_factory=SCMECalculatorFactory(DEFAULT_PARAMS, None, None),
        param_applier=SCMEParameterApplier(),
        path_to_reference_configuration=REFERENCE_CONFIGS[5],
        dt=1.0,
        max_steps=500,
        reference_OO_distance=3.2,
        tag="dimer_distance",
    )

    fitter = Fitter(objective_function=ob, initial_params=INITIAL_PARAMS)

    optimal_params = fitter.fit_scipy(tol=1e-4, options=dict(maxiter=50, disp=True))

    output_folder = Path(__file__).parent / "output/dimer_distance"
    ob.write_meta_data(output_folder)

    dump_dict_to_file(output_folder / "optimal_params.json", optimal_params)


def test_multi_energy_ob_function_fitting():
    ob = MultiEnergyObjectiveFunction(
        calc_factory=SCMECalculatorFactory(DEFAULT_PARAMS, None, None),
        param_applier=SCMEParameterApplier(),
        path_or_factory_list=REFERENCE_CONFIGS,
        reference_energy_list=REFERENCE_ENERGIES,
        tag_list=TAGS,
    )

    fitter = Fitter(objective_function=ob, initial_params=INITIAL_PARAMS)

    start = time.time()
    optimal_params = fitter.fit_scipy(tol=0, options=dict(maxiter=50, disp=True))
    end = time.time()
    print(f"time taken = {end - start} seconds")

    output_folder = Path(__file__).parent / "output/multi_energy"

    ob.write_output(
        output_folder,
        initial_params=INITIAL_PARAMS,
        optimal_params=optimal_params,
        plot_initial=True,
    )


def test_multi_energy_ob_function_fitting_mpi():
    from scme_fitting import HAS_MPI

    if not HAS_MPI:
        return

    from scme_fitting.mpi_wrapper_cob import MPIWrapperCOB

    ob = MultiEnergyObjectiveFunction(
        calc_factory=SCMECalculatorFactory(DEFAULT_PARAMS, None, None),
        param_applier=SCMEParameterApplier(),
        path_or_factory_list=REFERENCE_CONFIGS,
        reference_energy_list=REFERENCE_ENERGIES,
        tag_list=TAGS,
    )

    with MPIWrapperCOB(ob) as ob_mpi:
        if ob_mpi.rank == 0:
            start = time.time()

            fitter = Fitter(objective_function=ob_mpi, initial_params=INITIAL_PARAMS)
            optimal_params = fitter.fit_scipy(
                tol=0, options=dict(maxiter=50, disp=True)
            )
            print(f"{optimal_params = }")
            end = time.time()
            print(f"time taken = {end - start} seconds")


if __name__ == "__main__":
    # test_single_energy_objective_function()
    # test_dimer_distance_objective_function()
    # test_multi_energy_ob_function_fitting()
    test_multi_energy_ob_function_fitting_mpi()
