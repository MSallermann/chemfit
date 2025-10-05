import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import Bohr

import chemfit.kabsch as kb
from chemfit.abstract_objective_function import QuantityComputerObjectiveFunction
from chemfit.ase_objective_function import (
    AtomsFactory,
    MinimizationASEComputer,
    PathAtomsFactory,
    SinglePointASEComputer,
)
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.data_utils import process_csv
from chemfit.fitter import Fitter
from chemfit.utils import dump_dict_to_file

# Since the scme is not always installed with chemfit, we have to guard the import
scme_factories = pytest.importorskip(
    "chemfit.scme_factories", reason="Cannot import `pyscme`"
)

logger = logging.getLogger(__name__)
logging.basicConfig(filename="./output/test_scme_fitter.log", level=logging.DEBUG)

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
        "C6_OO": 46.4430e0,
        "C8_OO": 1141.7000e0,
        "C10_OO": 33441.0000e0,
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
        "C6_OO": 46.4430e0,
        "C8_OO": 1141.7000e0,
        "C10_OO": 33441.0000e0,
    },
}


def test_factories():
    atoms = Atoms()

    calc_factory = scme_factories.SCMECalculatorFactory(DEFAULT_PARAMS, None, None)
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

    param_applier = scme_factories.SCMEParameterApplier()
    param_applier(atoms, INITIAL_PARAMS)

    check_if_params_applied(INITIAL_PARAMS)


def test_single_energy_objective_function():
    ob = QuantityComputerObjectiveFunction(
        loss_function=lambda quants: (quants["energy"] - REFERENCE_ENERGIES[10]) ** 2,
        quantity_computer=SinglePointASEComputer(
            calc_factory=scme_factories.SCMECalculatorFactory(
                DEFAULT_PARAMS, None, None
            ),
            param_applier=scme_factories.SCMEParameterApplier(),
            atoms_factory=PathAtomsFactory(REFERENCE_CONFIGS[10]),
            tag=TAGS[10],
        ),
    )

    fitter = Fitter(objective_function=ob, initial_params=INITIAL_PARAMS)

    optimal_params = fitter.fit_scipy(tol=1e-4, options={"maxiter": 50})
    output_folder = Path(__file__).parent / "output/single_energy"

    dump_dict_to_file(output_folder / "optimal_params.json", optimal_params)


def test_dimer_distance_objective_function():
    REF_DISTANCE = 3.2

    def compute_dimer_distance(calc: Calculator, atoms: Atoms):
        quants = calc.results
        quants["dimer_distance"] = atoms.get_distance(0, 3)
        return quants

    ob = QuantityComputerObjectiveFunction(
        loss_function=lambda quants: (quants["dimer_distance"] - REF_DISTANCE) ** 2,
        quantity_computer=MinimizationASEComputer(
            calc_factory=scme_factories.SCMECalculatorFactory(
                DEFAULT_PARAMS, None, None
            ),
            param_applier=scme_factories.SCMEParameterApplier(),
            atoms_factory=PathAtomsFactory(REFERENCE_CONFIGS[10]),
            quantities_processor=compute_dimer_distance,
            tag="dimer_distance",
        ),
    )

    fitter = Fitter(objective_function=ob, initial_params=INITIAL_PARAMS)

    optimal_params = fitter.fit_scipy(tol=1e-4, options={"maxiter": 50})
    print(f"{optimal_params = }")
    print(f"{fitter.info = }")
    print(f"{ob.get_meta_data() = }")


def test_kabsch_objective_function():
    class KabschDistance:
        def __init__(self, atoms_factory: AtomsFactory):
            self.atoms_factory = atoms_factory
            self._positions_ref = None

        def __call__(self, calc: Calculator, atoms: Atoms) -> dict[str, Any]:
            if self._positions_ref is None:
                self._positions_ref = self.atoms_factory().positions

            res = calc.results

            kabsch_r, kabsch_t = kb.kabsch(atoms.positions, self._positions_ref)
            positions_aligned = kb.apply_transform(atoms.positions, kabsch_r, kabsch_t)

            kabsch_rmsd = kb.rmsd(positions_aligned, self._positions_ref)

            res.update(
                {"kabsch_t": kabsch_t, "kabsch_r": kabsch_r, "kabsch_rmsd": kabsch_rmsd}
            )

            return res

    ob = QuantityComputerObjectiveFunction(
        loss_function=lambda quants: quants["kabsch_rmsd"],
        quantity_computer=MinimizationASEComputer(
            calc_factory=scme_factories.SCMECalculatorFactory(
                DEFAULT_PARAMS, None, None
            ),
            param_applier=scme_factories.SCMEParameterApplier(),
            atoms_factory=PathAtomsFactory(REFERENCE_CONFIGS[10]),
            quantities_processor=KabschDistance(
                atoms_factory=PathAtomsFactory(REFERENCE_CONFIGS[10])
            ),
            tag="kabsch",
        ),
    )

    fitter = Fitter(objective_function=ob, initial_params=INITIAL_PARAMS)

    optimal_params = fitter.fit_scipy(tol=1e-4, options={"maxiter": 50})

    print(f"{optimal_params = }")
    print(f"{fitter.info = }")
    print(f"{ob.get_meta_data() = }")


def construct_objective_function(
    paths: list[Path], tags: list[str], energies: list[float]
):
    ob_list = []

    for p, t, e in zip(paths, tags, energies):
        ob_term = QuantityComputerObjectiveFunction(
            loss_function=lambda quants, e=e: (quants["energy"] - e) ** 2
            / quants["n_atoms"] ** 2,
            quantity_computer=SinglePointASEComputer(
                calc_factory=scme_factories.SCMECalculatorFactory(
                    default_scme_params=DEFAULT_PARAMS,
                    path_to_scme_expansions=None,
                    parametrization_key=None,
                ),
                param_applier=scme_factories.SCMEParameterApplier(),
                atoms_factory=PathAtomsFactory(p),
                tag=t,
            ),
        )

        ob_list.append(ob_term)

    return CombinedObjectiveFunction(ob_list)


def test_multi_energy_ob_function_fitting():
    ob = construct_objective_function(
        paths=REFERENCE_CONFIGS, tags=TAGS, energies=REFERENCE_ENERGIES
    )
    print(ob(INITIAL_PARAMS))
    print("--")

    fitter = Fitter(objective_function=ob, initial_params=INITIAL_PARAMS)

    optimal_params = fitter.fit_scipy(tol=1e-4)

    print(f"{fitter.info = }")
    print(f"{optimal_params = }")
    print(f"time taken = {fitter.info.time_taken} seconds")


def test_multi_energy_ob_function_fitting_mpi():
    mpi_wrapper_cob = pytest.importorskip(
        "chemfit.mpi_wrapper_cob", reason="Cannot import `mpi4py`"
    )

    ob = construct_objective_function(REFERENCE_CONFIGS, TAGS, REFERENCE_ENERGIES)

    with mpi_wrapper_cob.MPIWrapperCOB(ob) as ob_mpi:
        if ob_mpi.rank == 0:
            fitter = Fitter(objective_function=ob_mpi, initial_params=INITIAL_PARAMS)
            optimal_params = fitter.fit_scipy(tol=0, options={"maxiter": 50})
            print(f"{optimal_params = }")
            print(f"time taken = {fitter.info.time_taken} seconds")
        else:
            ob_mpi.worker_loop()
