from scme_fitting.ase_objective_function import (
    EnergyObjectiveFunction,
    CalculatorFactory,
    ParameterApplier,
    AtomsFactory,
    AtomsPostProcessor,
)
from scme_fitting.combined_objective_function import CombinedObjectiveFunction
import scme_fitting.plot_utils
import scme_fitting.utils

from pathlib import Path

from typing import Optional, Union
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class MultiEnergyObjectiveFunction(CombinedObjectiveFunction):
    """
    A CombinedObjectiveFunction that aggregates multiple energy-based objective functions.

    For each reference configuration, an EnergyObjectiveFunction is created with its own
    reference energy, and all these objective functions are combined (with optional weights)
    into a single callable. This class also supports writing out a detailed report of initial,
    fitted, and reference energies along with associated metadata.

    Inherits from:
        CombinedObjectiveFunction
    """

    def __init__(
        self,
        calc_factory: CalculatorFactory,
        param_applier: ParameterApplier,
        tag_list: list[str],
        reference_energy_list: list[float],
        path_or_factory_list: list[Union[Path, AtomsFactory]],
        divide_by_n_atoms: bool = False,
        weight_list: Optional[list[float]] = None,
        atom_post_processor_list: Optional[list[AtomsPostProcessor]] = None,
    ):
        """
        Initialize a MultiEnergyObjectiveFunction by constructing individual EnergyObjectiveFunctions.

        Each element of `tag_list`, `path_to_reference_configuration_list`, and `reference_energy_list`
        defines one EnergyObjectiveFunction instance. Those instances are collected and passed to the
        parent CombinedObjectiveFunction with the provided weights.

        Args:
            tag_list (list[str]):
                A list of labels (tags) for each reference configuration (e.g., "cluster1", "bulk").
            reference_energy_list (list[float]):
                A list of target energies corresponding to each reference configuration.
            path_or_factory_list (list[Union[Path, AtomsFactory]]):
                A list of filesystem paths, each pointing to a reference configuration file.
            divide_by_n_atoms (bool, default False):
                If True, energies will be normalized by the number of atoms in each configuration.
            weight_list (Optional[list[float]], optional):
                A list of non-negative floats specifying the combination weight for each
                EnergyObjectiveFunction. If None, all weights default to 1.0.

        Raises:
            AssertionError: If lengths of `tag_list`, `path_to_reference_configuration_list`, and
                `reference_energy_list` differ, or if any provided weight is negative.
        """

        self.calc_factory = calc_factory
        self.param_applier = param_applier

        ob_funcs: list[EnergyObjectiveFunction] = []

        if atom_post_processor_list is None:
            atom_post_processor_list = [None] * len(path_or_factory_list)

        for t, p_ref, e_ref, post_proc in zip(
            tag_list,
            path_or_factory_list,
            reference_energy_list,
            atom_post_processor_list,
        ):
            # First try to find out if p_ref is just a path,
            # or the more general AtomsFactory
            # depending on which it is, we call the constructor differently
            if isinstance(p_ref, Path):
                ob = EnergyObjectiveFunction(
                    calc_factory=self.calc_factory,
                    param_applier=self.param_applier,
                    path_to_reference_configuration=p_ref,
                    reference_energy=e_ref,
                    divide_by_n_atoms=divide_by_n_atoms,
                    tag=t,
                    atoms_post_processor=post_proc,
                )
            else:
                ob = EnergyObjectiveFunction(
                    calc_factory=self.calc_factory,
                    param_applier=self.param_applier,
                    atoms_factory=p_ref,
                    reference_energy=e_ref,
                    divide_by_n_atoms=divide_by_n_atoms,
                    tag=t,
                    atoms_post_processor=post_proc,
                )

            ob_funcs.append(ob)

        super().__init__(ob_funcs, weight_list)

    def write_output(
        self,
        folder_name: str,
        initial_params: dict[str, float],
        optimal_params: dict[str, float],
        plot_initial: bool = False,
        write_config: bool = False,
    ):
        """
        Generate output files and plots summarizing fitting results.

        Creates a new output folder (using the next free name under `folder_name`), dumps metadata
        and parameter sets as JSON, writes per-configuration SCME energy data to CSV, and
        produces plots of energies and residuals.

        Args:
            folder_name (str):
                Base name (or path) under which to create a uniquely named output directory.
            initial_params (dict[str, float]):
                Parameter values before fitting; saved to "initial_params.json" and used to compute
                initial energies if `plot_initial` is True.
            optimal_params (dict[str, float]):
                Parameter values after fitting; saved to "optimal_params.json" and used to compute
                fitted energies.
            plot_initial (bool): If `True` the curves will also be plotted for the initial parameters


        Raises:
            IOError: If creating directories or writing files fails.
        """
        output_folder = scme_fitting.utils.next_free_folder(Path(folder_name))
        output_folder.mkdir(exist_ok=True, parents=True)

        print(f"Output folder: {output_folder}")
        logger.info(f"Output folder: {output_folder}")

        meta: dict[str, object] = {"name": folder_name}

        scme_fitting.utils.dump_dict_to_file(output_folder / "meta.json", meta)
        scme_fitting.utils.dump_dict_to_file(
            output_folder / "initial_params.json", initial_params
        )
        scme_fitting.utils.dump_dict_to_file(
            output_folder / "optimal_params.json", optimal_params
        )

        for o in self.objective_functions:
            try:
                o.write_meta_data(output_folder / "reference_configs", write_config)
            except Exception:
                # Continue even if dumping a particular configuration fails
                pass

        # Extract per-objective weights and energy values
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
            "weight_energy": weights_energy,
            "weight_combination": weights_combination,
            "weight": weights_total,
            "ob_value": ob_value,
        }

        energies_df = pd.DataFrame(energies_scme)
        energies_df.to_csv(output_folder / "energies.csv")

        scme_fitting.plot_utils.plot_energies_and_residuals(
            df=energies_df,
            output_folder=output_folder,
            plot_initial=plot_initial,
        )
