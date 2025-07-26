from chemfit.ase_objective_function import (
    EnergyObjectiveFunction,
    CalculatorFactory,
    ParameterApplier,
    AtomsFactory,
    AtomsPostProcessor,
)
from chemfit.combined_objective_function import CombinedObjectiveFunction
import chemfit.plot_utils
import chemfit.utils

from collections.abc import Sequence

from pathlib import Path
from ase import Atoms
from typing import Optional, Union, Callable
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
        weight_cb: Union[
            None, list[Callable[[Atoms], float]], Callable[[Atoms], float]
        ] = None,
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
                A list of filesystem paths or atom factories.
            weight_cb (Union[None, list[Callable[[Atoms, float]]], Callable[[Atoms], float], default None):
                Either a single callable or a list of callables for the weight callback or None
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

        n_terms = len(path_or_factory_list)

        if atom_post_processor_list is None:
            atom_post_processor_list = [None] * n_terms

        if weight_cb is None:
            weight_cb_list = [None] * n_terms
        elif not isinstance(weight_cb, Sequence):
            weight_cb_list = [weight_cb] * n_terms
        else:
            assert n_terms == len(weight_cb)
            weight_cb_list = weight_cb

        for t, p_ref, e_ref, post_proc, w_cb in zip(
            tag_list,
            path_or_factory_list,
            reference_energy_list,
            atom_post_processor_list,
            weight_cb_list,
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
                    tag=t,
                    atoms_post_processor=post_proc,
                    weight_cb=w_cb,
                )
            else:
                ob = EnergyObjectiveFunction(
                    calc_factory=self.calc_factory,
                    param_applier=self.param_applier,
                    atoms_factory=p_ref,
                    reference_energy=e_ref,
                    tag=t,
                    atoms_post_processor=post_proc,
                    weight_cb=w_cb,
                )

            ob_funcs.append(ob)

        super().__init__(ob_funcs, weight_list)

    def write_output(
        self,
        folder_name: str,
        initial_params: dict[str, float],
        optimal_params: dict[str, float],
        plot_initial: bool = False,
        write_configs: bool = False,
        write_meta_data: bool = False,
    ):
        """
        Generate output files and plots summarizing fitting results.

        Creates a new output folder (using the next free name under `folder_name`), dumps metadata
        and parameter sets as JSON, writes per-configuration energy data to CSV, and
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
            write_configs (bool): Whether to write an individual atomic configuration for each
                reference configuration as xyz files
            write_meta_data (bool): Whether to write an individual metadata json for each reference
                configuration

        Raises:
            IOError: If creating directories or writing files fails.
        """
        output_folder = chemfit.utils.next_free_folder(Path(folder_name))
        output_folder.mkdir(exist_ok=True, parents=True)

        logger.info(f"Output folder: {output_folder}")

        meta: dict[str, object] = {"name": folder_name}

        chemfit.utils.dump_dict_to_file(output_folder / "meta.json", meta)
        chemfit.utils.dump_dict_to_file(
            output_folder / "initial_params.json", initial_params
        )
        chemfit.utils.dump_dict_to_file(
            output_folder / "optimal_params.json", optimal_params
        )

        if write_meta_data:
            for o in self.objective_functions:
                try:
                    o.write_meta_data(
                        output_folder / "reference_configs", write_configs
                    )
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
                ob.compute_energy(initial_params) for ob in self.objective_functions
            ],
            "energy_fitted": [
                ob.compute_energy(optimal_params) for ob in self.objective_functions
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

        chemfit.plot_utils.plot_energies_and_residuals(
            df=energies_df,
            output_folder=output_folder,
            plot_initial=plot_initial,
        )
