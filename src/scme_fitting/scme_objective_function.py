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

from ase.optimize import FIRE2
import numpy as np

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
        parametrization_key: Optional[str],
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
        self.parametrization_key: Optional[str] = parametrization_key
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
        self.energy = self.atoms.get_potential_energy()
        logger.debug(f"Calculated energy (tag = {self.tag}): {self.energy}")
        return self.energy

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


class EnergyObjectiveFunction(SCMEObjectiveFunction):
    """
    SCME-based objective function that penalizes deviation of computed energy
    from a reference energy. The contribution is weight * (E_computed - E_reference)^2,
    optionally normalized by the number of atoms, and weighted.
    """

    def __init__(
        self,
        default_scme_params: SCMEParams,
        parametrization_key: Optional[str],
        path_to_scme_expansions: Optional[Path],
        path_to_reference_configuration: Path,
        reference_energy: float,
        divide_by_n_atoms: bool = False,
        tag: Optional[str] = None,
        weight: float = 1.0,
        weight_cb: Optional[Callable[[Atoms], float]] = None,
    ) -> None:
        """
        Initialize an SCMEEnergyObjectiveFunction.
        Args:
            reference_energy: float
                The target energy (in energy units compatible with ASE) to compare against.
        """
        # Call parent initializer (loads atoms, applies SCME calculator, sets up weight via weight_cb if provided)
        super().__init__(
            default_scme_params=default_scme_params,
            parametrization_key=parametrization_key,
            path_to_scme_expansions=path_to_scme_expansions,
            path_to_reference_configuration=path_to_reference_configuration,
            tag=tag,
            weight=weight,
            weight_cb=weight_cb,
        )

        # We store the current energy here
        self.energy: Optional[float] = None

        # Store the reference energy
        self.reference_energy: float = reference_energy

        # Whether to normalize weight by the number of atoms
        self.divide_by_n_atoms: bool = divide_by_n_atoms

        if self.divide_by_n_atoms:
            n_atoms = len(self.atoms)
            self.weight /= n_atoms

    def get_meta_data(self) -> Dict[str, object]:
        """
        Extend parent metadata with reference energy.

        Returns:
            Dict[str, object]: Metadata from the parent, plus:
              - "reference_energy": float
        """
        data = super().get_meta_data()
        data["reference_energy"] = self.reference_energy
        return data

    def __call__(self, parameters: Dict[str, float]) -> float:
        """
        Compute squared-error contribution to the objective function:
            (E_computed(parameters) - E_reference)^2 * weight.

        Parameters:
            parameters: Dict[str, float]
                SCME parameter names to values; these will be applied before evaluating energy.

        Returns:
            float: The weighted squared difference between computed and reference energies.
        """
        energy = self.get_energy(parameters)
        target_energy = self.reference_energy

        error = (energy - target_energy) ** 2
        objective_contribution = error * self.weight

        logger.debug(f"Parameters applied: {parameters}")
        logger.debug(f"Computed energy: {energy}")
        logger.debug(f"Reference energy: {target_energy}")
        logger.debug(f"Weight: {self.weight}")
        logger.debug(
            f"Objective contribution (squared error × weight): {objective_contribution}"
        )

        return objective_contribution


class DimerDistanceObjectiveFunction(SCMEObjectiveFunction):
    """
    Objective function that, for a water dimer configuration, relaxes the structure
    under the SCME potential (with optional noise), then measures the O-O distance
    and returns a weighted squared error relative to a target O-O distance.
    """

    def __init__(
        self,
        default_scme_params: SCMEParams,
        parametrization_key: str,
        path_to_scme_expansions: Path,
        path_to_reference_configuration: Path,
        OO_distance_target: float,
        tag: Optional[str] = None,
        dt: float = 1e-2,
        fmax: float = 1e-3,
        noise_magnitude: float = 0.0,
        weight: float = 1.0,
        weight_cb: Optional[Callable[[Atoms], float]] = None,
    ) -> None:
        """
        Initialize the DimerDistanceObjectiveFunction.

          - OO_distance_target: float
              Target oxygen-oxygen distance (in Å) after relaxation.
          - dt: float
              Initial timestep for FIRE2 optimization. Default is 1e-2.
          - fmax: float
              Convergence criterion for maximum force in FIRE2. Default is 1e-3.
                      - fmax: float
          - noise_magnitude: float
              Amount of random noise to add to the atom positions before each relaxation
        """
        # Store parameters used in relaxation & distance evaluation
        self.OO_distance_target: float = OO_distance_target

        self.dt: float = dt
        self.max_steps: int = 2000
        self.fmax: float = fmax
        self.n_atoms_required: int = 6  # Two water molecules (3 atoms each)
        self.noise_magnitude: float = noise_magnitude

        # Call parent initializer (loads atoms, applies SCME calculator, sets up weight via weight_cb if provided)
        super().__init__(
            default_scme_params=default_scme_params,
            parametrization_key=parametrization_key,
            path_to_scme_expansions=path_to_scme_expansions,
            path_to_reference_configuration=path_to_reference_configuration,
            tag=tag,
            weight=weight,
            weight_cb=weight_cb,
        )

        # The O–O distance after relaxation will be stored here
        self.OO_distance: float = 0.0

    def create_atoms_object_from_configuration(
        self, path_to_configuration: Path
    ) -> Atoms:
        """
        Override to ensure that the loaded Atoms object represents exactly a water dimer
        (6 atoms). After loading and ordering via the parent method, assert the atom count.

        Args:
            path_to_configuration: Path
                Path to an ASE-readable file (e.g., .xyz) containing the dimer geometry.

        Returns:
            Atoms: ASE Atoms object with SCME calculator attached.

        Raises:
            AssertionError: If the loaded Atoms object does not contain exactly 6 atoms.
        """
        atoms = super().create_atoms_object_from_configuration(path_to_configuration)
        assert len(atoms) == self.n_atoms_required, (
            f"Expected {self.n_atoms_required} atoms for a water dimer, but got {len(atoms)}."
        )
        return atoms

    def get_meta_data(self) -> Dict[str, object]:
        """
        Extend parent metadata with current and target O-O distances.

        Returns:
            Dict[str, object]: Metadata including:
              - "tag": str
              - "original_file": str
              - "saved_file": str
              - "n_atoms": int
              - "weight": float
              - "oo_distance": float (current O-O distance, may be 0.0 before evaluation)
              - "oo_distance_target": float
        """
        data = super().get_meta_data()
        data["oo_distance"] = self.OO_distance
        data["oo_distance_target"] = self.OO_distance_target
        return data

    def __call__(self, parameters: Dict[str, float]) -> float:
        """
        Apply the given SCME parameters, add small random noise to atomic positions,
        relax the dimer with FIRE2 until forces < fmax or max_steps reached, then
        compute the O-O distance between atoms 0 and 3. Return the weighted squared
        error relative to OO_distance_target.

        Args:
            parameters: Dict[str, float]
                SCME parameter names to float values to set before relaxation.

        Returns:
            float: Weighted squared difference: weight * (OO_distance - OO_distance_target)^2
        """
        # Apply new SCME parameters
        self.apply_parameters(parameters)

        # Force a recalculation to update forces/energy before adding noise
        self.atoms.calc.calculate(self.atoms)

        # Add a bit of random noise to the positions to avoid getting stuck in symmetric minima
        self.atoms.positions += self.noise_magnitude * np.random.uniform(
            -1.0, 1.0, size=self.atoms.positions.shape
        )

        # Run FIRE2 relaxation until convergence
        optimizer = FIRE2(self.atoms, dt=self.dt)
        optimizer.run(fmax=self.fmax, steps=self.max_steps)

        # Compute the O–O distance (atom indices 0 and 3, allowing periodic images)
        self.OO_distance = self.atoms.get_distance(0, 3, mic=True)
        logger.debug(f"Relaxed O-O distance: {self.OO_distance:.6f} Å")

        # Compute weighted squared error
        diff = self.OO_distance - self.OO_distance_target
        objective_value = self.weight * diff**2

        logger.debug(f"Parameters applied: {parameters}")
        logger.debug(f"Target O-O distance: {self.OO_distance_target:.6f} Å")
        logger.debug(f"Weight: {self.weight}")
        logger.debug(f"Objective (weight * squared error): {objective_value:.6e}")

        return objective_value
