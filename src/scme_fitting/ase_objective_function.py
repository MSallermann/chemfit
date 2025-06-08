from ase import Atoms
from ase.io import read, write
from typing import Optional, Dict, Callable, Union
from pathlib import Path
import json
import abc
import logging
import numpy as np
from ase.optimize import BFGS


from typing import Protocol, Any


class CalculatorFactory(Protocol):
    def __call__(self, atoms: Atoms) -> Any: ...


class ParameterApplier(Protocol):
    def __call__(self, atoms: Atoms, params: Dict[str, float]) -> None: ...


logger = logging.getLogger(__name__)


class ASEObjectiveFunction(abc.ABC):
    """
    Base class for ASE-based objective functions. Loads a reference configuration,
    attaches a calculator, and provides an interface for evaluating energies
    given a set of parameters.

    Subclasses must implement:

    - `apply_parameters` to apply a set of new parameters
    - `setup_calculator` to construct the calculator instance and attach it to an atoms object
    - `__call__` to return the objective value (e.g., squared error against some target).

    They may optionally implement:
    - `check_atoms` to check if the constructed atoms object conforms to expectation and if not correct it
      (the default implementation does nothing)
    """

    def __init__(
        self,
        calc_factory: CalculatorFactory,
        param_applier: ParameterApplier,
        path_to_reference_configuration: Path,
        tag: Optional[str] = None,
        weight: float = 1.0,
        weight_cb: Optional[Callable[[Atoms], float]] = None,
        divide_by_n_atoms: bool = False,
    ) -> None:
        """
        Initialize an ASEObjectiveFunction.

        Args:

            path_to_reference_configuration: Path
                Path to an ASE-readable file (e.g., .xyz) that contains the molecular configuration
                to be evaluated. This file will be read, reordered (if necessary), and assigned
                the calculator.
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

        self.calc_factory: CalculatorFactory = calc_factory
        self.param_applier: ParameterApplier = param_applier

        # Assign tag
        self.tag: str = tag if tag is not None else "tag_None"

        # Store parameters for later calculator setup
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

        # Whether to normalize weight by the number of atoms
        self.divide_by_n_atoms: bool = divide_by_n_atoms

        if self.divide_by_n_atoms:
            n_atoms = len(self.atoms)
            weight /= n_atoms

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
        (if necessary), and attach sets up the calculator.

        Parameters
        ----------
        path_to_configuration : Path
            File path to an ASE-readable structure (e.g. .xyz).

        Returns
        -------
        Atoms
            ASE Atoms object with calculator attached and ready for energy evaluation.

        """
        logger.debug(f"Loading configuration from {path_to_configuration}")

        atoms = read(path_to_configuration, index=0)

        self.check_atoms(atoms)
        self.calc_factory(atoms)

        return atoms

    def get_energy(self, parameters: Dict[str, float]) -> float:
        """
        Compute energy for the reference configuration given a set of parameters.

        This method updates the calculator parameters, forces a recalculation of
        the energy (to bypass ASE caching), and then returns the new potential energy.

        Parameters
        ----------
        parameters : Dict[str, float]
            Dictionary of parameter names to float values to apply before evaluation.

        Returns
        -------
        float
            The potential energy of the Atoms object under the updated calculator
        """

        self.param_applier(self.atoms, parameters)

        # Force a fresh energy evaluation (ASE might otherwise use a cached value)
        self.atoms.calc.calculate(self.atoms)
        self.energy = self.atoms.get_potential_energy()
        logger.debug(f"Calculated energy (tag = {self.tag}): {self.energy}")
        return self.energy

    def check_atoms(self, atoms: Atoms) -> bool:
        return True

    @abc.abstractmethod
    def __call__(self, parameters: Dict[str, float]) -> float:
        """
        Compute the objective function value given a set of parameters.

        Subclasses should override this to compare `get_energy(parameters)` against
        reference data or other criteria and return a scalar objective value.

        Parameters
        ----------
        parameters : Dict[str, float]
            Dictionary of parameter names to float values to apply.

        Returns
        -------
        float
            The computed objective value (e.g., error metric) for these parameters.
        """
        ...


class EnergyObjectiveFunction(ASEObjectiveFunction):
    def __init__(
        self,
        calc_factory: CalculatorFactory,
        param_applier: ParameterApplier,
        path_to_reference_configuration: Path,
        reference_energy: float,
        tag: Optional[str] = None,
        weight: float = 1.0,
        weight_cb: Optional[Callable[[Atoms], float]] = None,
        divide_by_n_atoms: bool = False,
    ):
        self.reference_energy = reference_energy

        super().__init__(
            calc_factory=calc_factory,
            param_applier=param_applier,
            path_to_reference_configuration=path_to_reference_configuration,
            tag=tag,
            weight=weight,
            weight_cb=weight_cb,
            divide_by_n_atoms=divide_by_n_atoms,
        )

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
                parameter names to values; these will be applied before evaluating energy.

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


class DimerDistanceObjectiveFunction(ASEObjectiveFunction):
    """
    Objective function that, for a water dimer configuration, relaxes the structure
    under the SCME potential (with optional noise), then measures the O-O distance
    and returns a weighted squared error relative to a target O-O distance.
    """

    def __init__(
        self,
        calc_factory: CalculatorFactory,
        param_applier: ParameterApplier,
        path_to_reference_configuration: Path,
        reference_OO_distance: float,
        dt: float = 1e-2,
        fmax: float = 1e-5,
        max_steps: int = 2000,
        noise_magnitude: float = 0.0,
        tag: Optional[str] = None,
        weight: float = 1.0,
        weight_cb: Optional[Callable[[Atoms], float]] = None,
        divide_by_n_atoms: bool = False,
    ):
        self.reference_OO_distance: float = reference_OO_distance
        self.dt: float = dt
        self.fmax: float = fmax
        self.max_steps: int = max_steps
        self.noise_magnitude: float = noise_magnitude

        super().__init__(
            calc_factory=calc_factory,
            param_applier=param_applier,
            path_to_reference_configuration=path_to_reference_configuration,
            tag=tag,
            weight=weight,
            weight_cb=weight_cb,
            divide_by_n_atoms=divide_by_n_atoms,
        )

        self.positions_reference = np.array(self.atoms.positions)

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
        data["reference_OO_distance"] = self.reference_OO_distance
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
        self.param_applier(self.atoms, parameters)

        # Set the positions to the reference and zero out the velocities
        self.atoms.set_velocities(np.zeros(shape=(len(self.atoms), 3)))
        self.atoms.set_positions(self.positions_reference)

        # Force a recalculation to update forces/energy before adding noise
        self.atoms.calc.calculate(self.atoms)

        # Add a bit of random noise to the positions to avoid getting stuck in symmetric minima
        self.atoms.positions += self.noise_magnitude * np.random.uniform(
            -1.0, 1.0, size=self.atoms.positions.shape
        )

        # Run FIRE2 relaxation until convergence
        # optimizer = FIRE2(self.atoms, dtmax=self.dt)
        optimizer = BFGS(self.atoms)  # cf, dtmax=self.dt)

        optimizer.run(fmax=self.fmax, steps=self.max_steps)

        # Compute the O–O distance (atom indices 0 and 3, allowing periodic images)
        self.OO_distance = self.atoms.get_distance(0, 3, mic=True)
        logger.debug(f"Relaxed O-O distance: {self.OO_distance:.6f} Å")

        # Compute weighted squared error
        diff = self.OO_distance - self.reference_OO_distance
        objective_value = self.weight * diff**2

        logger.debug(f"Parameters applied: {parameters}")
        logger.debug(f"Target O-O distance: {self.reference_OO_distance:.6f} Å")
        logger.debug(f"Weight: {self.weight}")
        logger.debug(f"Objective (weight * squared error): {objective_value:.6e}")

        return objective_value
