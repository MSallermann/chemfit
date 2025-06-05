from .scme_setup import (
    setup_calculator,
    SCMEParams,
    check_water_is_in_OHH_order,
    arange_water_in_OHH_order,
)
from ase import Atoms
from ase.io import read, write
from typing import Optional, Dict, Callable, Union
from pathlib import Path
import json
import abc
import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)


class SCMEObjectiveFunction(abc.ABC):
    """
    Base class for SCME-based objective functions. Loads a reference configuration,
    attaches an SCME calculator, and provides an interface for evaluating energies
    given a set of SCME parameters. Subclasses must implement `__call__` to return
    the objective value (e.g., squared error against some target).
    """

    def __init__(
        self,
        default_scme_params: SCMEParams,
        parametrization_key: str,
        path_to_scme_expansions: Optional[Path],
        path_to_reference_configuration: Path,
        tag: Optional[str] = None,
        weight: float = 1.0,
        weight_cb: Optional[Callable[[Atoms], float]] = None,
    ) -> None:
        """
        Initialize an SCMEObjectiveFunction.

        Args:
            default_scme_params: SCMEParams
                A base set of SCME parameters; a copy of this will be attached to the calculator
                so that each objective instance starts from these defaults.

            parametrization_key: str
                Identifier used by `setup_calculator` to select a particular SCME parameter set from the provided expansion HDF5 files.

            path_to_scme_expansions: Optional[Path]
                Path to the HDF5 file containing the SCME monomer expansions. If None, the monomer expansions will not be used.

            path_to_reference_configuration: Path
                Path to an ASE-readable file (e.g., .xyz) that contains the molecular configuration
                to be evaluated. This file will be read, reordered (if necessary), and assigned
                the SCME calculator.
                **Important:** If the file contains multiple configuration snapshots,
                only the **first** will be used.

            tag: Optional[str]
                User-defined label for this objective function. If None, defaults to "tag_None".

            weight: float
                Base weight for this objective function. Must be non-negative.

            weight_cb: Optional[Callable[[Atoms], float]]
                An optional callback that accepts the `Atoms` object and returns a scaling factor.
                If provided, `weight` will be multiplied by `weight_cb(self.atoms)`.

        Raises:
            ValueError: If the atoms cannot be reordered into OHH order.
            AssertionError: If `weight` is negative or if `weight_cb(self.atoms)` returns a negative value.
        """
        # Assign tag
        self.tag: str = tag if tag is not None else "tag_None"

        # Store parameters for later calculator setup
        self.default_scme_params: SCMEParams = default_scme_params
        self.parametrization_key: str = parametrization_key
        self.path_to_scme_expansions: Optional[Path] = path_to_scme_expansions
        self.paths_to_reference_configuration: Path = path_to_reference_configuration

        # Load and prepare Atoms object
        self.atoms: Atoms = self.create_atoms_object_from_configuration(
            path_to_reference_configuration
        )
        self.n_atoms: int = len(self.atoms)

        # Validate and compute weight
        if weight < 0:
            raise AssertionError("Weight must be non-negative.")
        self.weight_cb: Optional[Callable[[Atoms], float]] = weight_cb

        if self.weight_cb is not None:
            scale = self.weight_cb(self.atoms)
            if scale < 0:
                raise AssertionError(
                    "Weight callback must return a non-negative scaling factor."
                )
            weight *= scale

        self.weight: float = weight

    def get_meta_data(self) -> Dict[str, Union[str, int, float]]:
        """
        Return metadata about this objective function, including tag, original file,
        saved filename, number of atoms, and weight.

        Returns:
            Dict[str, Union[str, int, float]]: A dictionary containing:
              - "tag": str
              - "original_file": str (path to the reference configuration)
              - "saved_file": str (filename under which the atoms will be written)
              - "n_atoms": int
              - "weight": float
        """
        name = f"atoms_{self.tag}.xyz"
        return {
            "tag": self.tag,
            "original_file": str(self.paths_to_reference_configuration),
            "saved_file": name,
            "n_atoms": self.n_atoms,
            "weight": self.weight,
        }

    def dump_test_configuration(self, path_to_folder: Path) -> None:
        """
        Write the reference configuration and metadata to disk for inspection.

        Args:
            path_to_folder: Path
                Directory where the .xyz file and metadata JSON will be written.
                The directory will be created if it does not exist.

        Returns:
            None
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
        (if necessary), and attach the SCME calculator.

        Parameters
        ----------
        path_to_configuration : Path
            File path to an ASE-readable structure (e.g. .xyz).

        Returns
        -------
        Atoms
            ASE Atoms object with SCME calculator attached and ready for energy evaluation.

        Raises
        ------
        ValueError
            If the atoms cannot be reordered into OHH order after attempting a fix.
        """
        logger.debug(f"Loading configuration from {path_to_configuration}")

        atoms = read(path_to_configuration, index=0)

        if len(atoms) == 0:
            raise ValueError("File contains zero atoms")

        # Attempt to fix ordering if not already in OHH order
        if not check_water_is_in_OHH_order(atoms):
            logger.warning("Will try to fix atoms object order")
            atoms = arange_water_in_OHH_order(atoms)

        if not check_water_is_in_OHH_order(atoms):
            logger.critical("Could not fix atoms object order")
            raise ValueError("Atoms not in OHH order")

        # Attach a fresh copy of default SCME parameters to this Atoms object
        scme_params_copy = self.default_scme_params.model_copy()
        setup_calculator(
            atoms,
            scme_params=scme_params_copy,
            parametrization_key=self.parametrization_key,
            path_to_scme_expansions=self.path_to_scme_expansions,
        )
        return atoms

    def apply_parameters(self, parameters: Dict[str, float]) -> None:
        """
        Assign SCME parameter values to the attached calculator.

        Parameters
        ----------
        parameters : Dict[str, float]
            Dictionary of SCME parameter names to float values.

        Raises
        ------
        KeyError
            If a key in `parameters` does not correspond to an attribute on the SCME potential.
        """
        for key, value in parameters.items():
            if hasattr(self.atoms.calc.scme, key):
                setattr(self.atoms.calc.scme, key, value)
            else:
                raise KeyError(
                    f"Cannot set parameter '{key}': "
                    "not a valid attribute on the SCME potential."
                )

    def get_energy(self, parameters: Dict[str, float]) -> float:
        """
        Compute SCME energy for the reference configuration given a set of parameters.

        This method updates the SCME potential parameters, forces a recalculation of
        the energy (to bypass ASE caching), and then returns the new potential energy.

        Parameters
        ----------
        parameters : Dict[str, float]
            Dictionary of SCME parameter names to float values to apply before evaluation.

        Returns
        -------
        float
            The potential energy of the Atoms object under the updated SCME potential.
        """
        self.apply_parameters(parameters)

        # Force a fresh energy evaluation (ASE might otherwise use a cached value)
        self.atoms.calc.calculate(self.atoms)
        energy = self.atoms.get_potential_energy()
        logger.debug(f"Calculated energy (tag = {self.tag}): {energy}")
        return energy

    @abc.abstractmethod
    def __call__(self, parameters: Dict[str, float]) -> float:
        """
        Compute the objective function value given a set of SCME parameters.

        Subclasses should override this to compare `get_energy(parameters)` against
        reference data or other criteria and return a scalar objective value.

        Parameters
        ----------
        parameters : Dict[str, float]
            Dictionary of SCME parameter names to float values to apply.

        Returns
        -------
        float
            The computed objective value (e.g., error metric) for these parameters.
        """
        ...
