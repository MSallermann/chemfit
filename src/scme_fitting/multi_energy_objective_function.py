from scme_fitting.scme_objective_function import EnergyObjectiveFunction
from scme_fitting.combined_objective_function import CombinedObjectiveFunction
import scme_fitting.plot_utils
from .scme_setup import SCMEParams
import scme_fitting.utils

from pathlib import Path
import pyscme

from typing import Optional

import logging
import pandas as pd

logger = logging.getLogger(__name__)


class MultiEnergyObjectiveFunction(CombinedObjectiveFunction):
    def __init__(
        self,
        default_scme_params: SCMEParams,
        parametrization_key: str,
        path_to_scme_expansions: Optional[Path],
        tag_list: list[str],
        path_to_reference_configuration_list: list[Path],
        reference_energy_list: list[float],
        divide_by_n_atoms: bool = False,
        weight_list: Optional[list[float]] = None,
        plot_initial: bool = False,
    ):
        self.path_to_scme_expansions = path_to_scme_expansions
        self.parametrization_key = parametrization_key
        self.default_scme_params = default_scme_params

        self.plot_initial = plot_initial

        ob_funcs = []
        for t, p_ref, e_ref in zip(
            tag_list, path_to_reference_configuration_list, reference_energy_list
        ):
            ob_funcs.append(
                EnergyObjectiveFunction(
                    default_scme_params=default_scme_params,
                    parametrization_key=parametrization_key,
                    path_to_scme_expansions=path_to_scme_expansions,
                    path_to_reference_configuration=p_ref,
                    reference_energy=e_ref,
                    divide_by_n_atoms=divide_by_n_atoms,
                    tag=t,
                )
            )

        super().__init__(ob_funcs, weight_list)

    def write_output(
        self,
        folder_name: str,
        initial_params: dict[str, float],
        optimal_params: dict[str, float],
    ):
        output_folder = scme_fitting.utils.next_free_folder(Path(folder_name))
        output_folder.mkdir(exist_ok=True)

        print(f"Output folder: {output_folder}")
        logger.info(f"Output folder: {output_folder}")

        meta = dict()
        meta["name"] = folder_name
        meta["parametrization_key"] = self.parametrization_key
        meta["path_to_scme_expansions"] = self.path_to_scme_expansions
        meta["scme_version"] = {
            "branch": pyscme.version.branch(),
            "commit": pyscme.version.commit(),
            "date": pyscme.version.date(),
        }

        scme_fitting.utils.dump_dict_to_file(output_folder / "meta.json", meta)
        scme_fitting.utils.dump_dict_to_file(
            output_folder / "initial_params.json", initial_params
        )
        scme_fitting.utils.dump_dict_to_file(
            output_folder / "optimal_params.json", optimal_params
        )
        scme_fitting.utils.dump_dict_to_file(
            output_folder / "default_params.json", dict(self.default_scme_params)
        )

        for o in self.objective_functions:
            try:
                o.dump_test_configuration(output_folder / "reference_configs")
            except Exception():
                ...

        weights_energy = [ob.weight for ob in self.objective_functions]
        weights_combination = self.weights
        ob_value = [ob(optimal_params) for ob in self.objective_functions]

        weights_total = [w1 * w2 for w1, w2 in zip(weights_energy, weights_combination)]

        energies_scme = {
            "tag": [ob.tag for ob in self.objective_functions],
            "energy_initial": [
                ob.get_energy(initial_params) for ob in self.objective_functions
            ],
            "energy_fitted": [
                ob.get_energy(optimal_params) for ob in self.objective_functions
            ],
            "energy_reference": [
                ob.reference_energy for ob in self.objective_functions
            ],
            "n_atoms": [ob.n_atoms for ob in self.objective_functions],
            "weight_energy": [w for w in weights_energy],
            "weight_combination": [w for w in weights_combination],
            "weight": [w for w in weights_total],
            "ob_value": ob_value,
        }

        energies_scme_df = pd.DataFrame(energies_scme)

        energies_scme_df.to_csv(output_folder / "energies_scme.csv")

        scme_fitting.plot_utils.plot_energies_and_residuals(
            df=energies_scme_df,
            output_folder=output_folder,
            plot_initial=self.plot_initial,
        )
