from .scme_setup import (
    setup_calculator,
    SCMEParams,
    check_water_is_in_OHH_order,
    arange_water_in_OHH_order,
)
from ase import Atoms
from ase.io import read, write
from typing import Optional, Dict, Callable
from pathlib import Path
import json
import abc

import logging

logger = logging.getLogger(__name__)


class SCMEObjectiveFunction:
    def __init__(
        self,
        default_scme_params: SCMEParams,
        parametrization_key: str,
        path_to_scme_expansions: Optional[Path],
        path_to_reference_configuration: Path,
        tag: Optional[str] = None,
        weight: float = 1.0,
        weight_cb: Optional[Callable[[Atoms], float]] = None,
    ):
        if tag is None:
            self.tag = "tag_None"
        else:
            self.tag = tag

        self.default_scme_params: SCMEParams = default_scme_params
        self.path_to_scme_expansions: Path = path_to_scme_expansions
        self.parametrization_key: str = parametrization_key
        self.paths_to_reference_configuration = path_to_reference_configuration

        self.atoms = self.create_atoms_object_from_configuration(
            path_to_reference_configuration
        )

        self.n_atoms = len(self.atoms)

        self.weight_cb = weight_cb

        if self.weight_cb is not None:
            weight *= self.weight_cb(self.atoms)

        self.weight = weight

    def get_meta_data(self):
        name = f"atoms_{self.tag}.xyz"
        return {
            "tag": self.tag,
            "original_file": str(self.paths_to_reference_configuration),
            "saved_file": name,
            "n_atoms": len(self.atoms),
            "weight": self.weight,
        }

    def dump_test_configuration(self, path_to_folder: Path):
        """
        Write reference configurations and energies to disk for inspection.
        """
        path_to_folder = Path(path_to_folder)
        path_to_folder.mkdir(exist_ok=True, parents=True)

        meta_data = self.get_meta_data()
        name = meta_data["saved_file"]
        write(path_to_folder / name, self.atoms)

        with open(path_to_folder / f"meta_{self.tag}.json", "w") as f:
            json.dump(meta_data, f, indent=4)

    def create_atoms_object_from_configuration(
        self, path_to_configuration: Path
    ) -> Atoms:
        """
        Load atoms from a configuration file, reorder them to conform to OHH order
        and attach the SCME calculator.

        Parameters
        ----------
        path_to_configuration : Path
            File path to an ASE-readable structure (e.g. .xyz).

        Returns
        -------
        Atoms
            Atoms object with SCME calculator attached and ready for energy eval.
        """
        logger.debug(f"Loading configuration from {path_to_configuration}")
        atoms = read(path_to_configuration, index=0)

        # If the first check does not pass, we will try to fix the order of atoms
        if not check_water_is_in_OHH_order(atoms):
            logger.warning("Will try to fix atoms object order")
            atoms = arange_water_in_OHH_order(atoms)

        # If we are not able to fix the order of atoms we raise an exception
        if not check_water_is_in_OHH_order(atoms):
            logger.critical("Could not fix atoms object order")
            raise ValueError("Atoms not in OHH order")

        scme_params = self.default_scme_params.model_copy()
        setup_calculator(
            atoms,
            scme_params=scme_params,
            parametrization_key=self.parametrization_key,
            path_to_scme_expansions=self.path_to_scme_expansions,
        )
        return atoms

    def apply_parameters(self, parameters: Dict[str, float]):
        for key, value in parameters.items():
            if hasattr(self.atoms.calc.scme, key):
                setattr(self.atoms.calc.scme, key, value)
            else:
                raise KeyError(
                    f"There was a key in the parameters dict, which cannot be set on the scmecpp.Potential object. The offending key was {key}"
                )

    def get_energy(self, parameters: Dict[str, float]) -> float:
        """
        Compute SCME energy for configuration `idx` with `parameters` applied.

        Parameters
        ----------
        idx : int
            Index of the reference configuration.
        parameters : Dict[str, float]
            SCME parameter values to set before evaluation.

        Returns
        -------
        float
            Potential energy from the ASE Atoms object.
        """

        self.apply_parameters(parameters)

        # We have to make sure to trigger the update of the energy manually,
        # because ase will think it can use the cached energy values,
        # since none of the coordinates has changed.
        # Therefore, we explicitly call the `calculate` function
        self.atoms.calc.calculate(self.atoms)
        energy = self.atoms.get_potential_energy()
        logger.debug(f"Calculated energy (tag = {self.tag}): {energy}")
        return energy

    @abc.abstractmethod
    def __call__(self, parameters: dict) -> float:
        """
        The implementation of the contribution to the objective function

        Parameters
        ----------

        parameters : Dict[str, float]
            SCME parameter values to apply.

        Returns
        -------
        float
            objective function value
        """
        ...
