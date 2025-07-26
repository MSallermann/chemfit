from ase import Atoms
from ase.io import read, write
from typing import Optional, Callable, Union, Protocol, Any
from pathlib import Path
import json
import abc
import logging
import numpy as np
from ase.optimize import BFGS
from chemfit.utils import dump_dict_to_file

logger = logging.getLogger(__name__)


class CalculatorFactory(Protocol):
    """
    Protocol for a factory that constructs an ASE calculator in-place and attaches it to `atoms`

    Methods:
        __call__(atoms): Construct a calculator and overwrite atoms.calc.
    """

    def __call__(self, atoms: Atoms) -> None:  # pragma: no cover
        ...


class ParameterApplier(Protocol):
    """
    Protocol for a function that applies parameters to an ASE calculator.

    Methods:
        __call__(atoms, params): Applies a parameter dictionary to `atoms.calc` in-place.
    """

    def __call__(self, atoms: Atoms, params: dict) -> None:  # pragma: no cover
        ...


class AtomsPostProcessor(Protocol):
    """
    Protocol for a function that post-processes an ASE Atoms object.

    Methods:
        __call__(atoms): Performs in-place modifications or checks on the Atoms object.
    """

    def __call__(self, atoms: Atoms) -> None:  # pragma: no cover
        ...


class AtomsFactory(Protocol):
    """
    Protocol for a function that creates an ASE Atoms object.

    Methods:
        __call__(): Creates the atoms object
    """

    def __call__(self) -> Atoms:  # pragma: no cover
        ...


class PathAtomsFactory:
    """Implementation of AtomsFactory which reads the atoms from a path."""

    from ase.io import read

    def __init__(self, path: Path, index: Optional[int] = None):
        self.path = path
        self.index = index

    def __call__(self) -> Atoms:
        logger.debug(f"Loading configuration from {self.path}")
        atoms = read(self.path, self.index, parallel=False)
        return atoms


class ASEObjectiveFunction(abc.ABC):
    """
    Base class for ASE-based objective functions.

    This class loads a reference configuration, optionally post-processes the structure,
    attaches a calculator, and provides an interface for evaluating energies
    given a set of parameters.

    Subclasses must implement:
      - `__call__`: Return the objective value (e.g., squared error against a target).

    Attributes:
        calc_factory (CalculatorFactory): Factory to create ASE calculators.
        param_applier (ParameterApplier): Function to apply parameters to the calculator.
        atoms_post_processor (Optional[AtomsPostProcessor]): Optional hook to process Atoms.
        tag (str): Label for this objective function.
    """

    def __init__(
        self,
        calc_factory: CalculatorFactory,
        param_applier: ParameterApplier,
        path_to_reference_configuration: Optional[Path] = None,
        tag: Optional[str] = None,
        weight: float = 1.0,
        weight_cb: Optional[Callable[[Atoms], float]] = None,
        atoms_factory: Optional[AtomsFactory] = None,
        atoms_post_processor: Optional[AtomsPostProcessor] = None,
    ) -> None:
        """
        Initialize an ASEObjectiveFunction.

        Args:
            calc_factory: Factory to create an ASE calculator given an `Atoms` object.
            param_applier: Function that applies a dict of parameters to `atoms.calc`.
            path_to_reference_configuration: Optional path to an ASE-readable file (e.g., .xyz) containing
                the molecular configuration. Only the first snapshot is used.
            tag: Optional label for this objective. Defaults to "tag_None" if None.
            weight: Base weight for this objective. Must be non-negative.
            weight_cb: Optional callback that returns a non-negative scaling factor
                given the `Atoms` object. The base weight is multiplied by this factor.
            atoms_factory: Optional[AtomsFactory] Optional function to create the Atoms object.
            atoms_post_processor: Optional function to modify or validate the Atoms object
                immediately after loading and before attaching the calculator.

        **Important**: One of `atoms_factory` or `path_to_reference_configuration` has to be specified.
        If both are specified `atoms_factory` takes precedence.

        Raises:
            AssertionError: If `weight` is negative or if `weight_cb` returns a negative value.
        """
        self.calc_factory = calc_factory
        self.param_applier = param_applier
        self.atoms_post_processor = atoms_post_processor

        self.tag = tag or "tag_None"

        # If no custom `atoms_factory` has been supplied, we try to create a `PathAtomsFactory` from the path to the reference configuration
        if atoms_factory is None:
            if path_to_reference_configuration is None:
                raise Exception(
                    "Neither `path_to_reference_configuration` nor a custom `atoms_factory` has been supplied"
                )
            self.atoms_factory = PathAtomsFactory(
                path_to_reference_configuration, index=0
            )
        else:
            self.atoms_factory = atoms_factory

        self._last_energy: Optional[float] = None

        # NOTE: You should probably use the `self.atoms` property
        # When the atoms object is requested for the first time, it will be lazily loaded via the atoms_factory
        self._atoms = None  # <- signals that atoms havent been loaded yet

        # NOTE: You should probably use the `self.weight` property
        # The final weight depends on the atoms object which is loaded lazily,
        # therefore we can only find it after the atoms object has been created
        self._weight = None  # <- signals that weights havent been created yet

        # This is the initial weight, which is a simple float so we can just assign it
        self.weight_init: float = weight

        if self.weight_init < 0:
            raise AssertionError("Weight must be non-negative.")

        self.weight_cb = weight_cb

    def get_meta_data(self) -> dict[str]:
        """
        Retrieve metadata for this objective function.

        Returns:
            dict[str, Union[str, int, float]]: Dictionary containing:
                tag: User-defined label.
                n_atoms: Number of atoms in the configuration.
                weight: Final weight after any scaling.
                last_energy: The last computed energy
        """
        return {
            "tag": self.tag,
            "n_atoms": self.n_atoms,
            "weight": self.weight,
            "last_energy": self._last_energy,
        }

    def write_meta_data(self, path_to_folder: Path, write_config: bool = False) -> None:
        """
        Write the reference configuration and metadata to disk.

        Args:
            path_to_folder: Directory where the .xyz file (if write_config is True) and metadata JSON
                will be written. The directory is created if it does not exist.
            write_config: If True, will also write .xyz file for the configuration
        """
        path_to_folder = Path(path_to_folder)
        path_to_folder.mkdir(exist_ok=True, parents=True)

        meta_data = self.get_meta_data()

        dump_dict_to_file(path_to_folder / f"meta_{self.tag}.json", meta_data)

        if write_config:
            write(path_to_folder / f"atoms_{self.tag}.xyz", self.atoms)

    def create_atoms_object(self) -> Atoms:
        """
        Create the atoms object, check it, optionally post-processes it, and attach the calculator.

        Returns:
            Atoms: ASE Atoms object with calculator attached.
        """

        atoms = self.atoms_factory()

        self.check_atoms(atoms)

        if self.atoms_post_processor is not None:
            self.atoms_post_processor(atoms)

        self.calc_factory(atoms)

        return atoms

    @property
    def atoms(self):
        """The atoms object. Accessing this property for the first time will create the atoms object."""
        # Check if the atoms have been created already and if not create them
        if self._atoms is None:
            self._atoms = self.create_atoms_object()
        return self._atoms

    @property
    def n_atoms(self):
        """The number of atoms in the atoms object. May trigger creation of the atoms object."""
        return len(self.atoms)

    @property
    def weight(self):
        """The weight. May trigger creation of the atoms object."""
        if self._weight is None:
            self._weight = self.weight_init
            if self.weight_cb is not None:
                scale = self.weight_cb(self.atoms)
                if scale < 0:
                    raise AssertionError(
                        "Weight callback must return a non-negative scaling factor."
                    )
                self._weight *= scale

        return self._weight

    def compute_energy(self, parameters: dict) -> float:
        """
        Compute the potential energy for a given set of parameters.

        Args:
            parameters: Dictionary of parameter names to float values.

        Returns:
            float: Potential energy after applying parameters.
        """
        self.param_applier(self.atoms, parameters)
        self.atoms.calc.calculate(self.atoms)
        self._last_energy = self.atoms.get_potential_energy()

        logger.debug(f"Calculated energy (tag = {self.tag}): {self._last_energy}")
        return self._last_energy

    def check_atoms(self, atoms: Atoms) -> bool:
        """
        Optional hook to validate or correct the Atoms object.

        Args:
            atoms: ASE Atoms object to check.

        Returns:
            bool: True if the atoms pass validation, False otherwise.
        """
        return True

    @abc.abstractmethod
    def __call__(self, parameters: dict) -> float:
        """
        Compute the objective value given a set of parameters.

        Args:
            parameters: Dictionary of parameter names to float values.

        Returns:
            float: Computed objective value (e.g., error metric).
        """
        ...


class EnergyObjectiveFunction(ASEObjectiveFunction):
    """
    Objective function comparing computed energy to a reference energy.
    """

    def __init__(
        self,
        calc_factory: CalculatorFactory,
        param_applier: ParameterApplier,
        reference_energy: float,
        path_to_reference_configuration: Optional[Path] = None,
        tag: Optional[str] = None,
        weight: float = 1.0,
        weight_cb: Optional[Callable[[Atoms], float]] = None,
        atoms_factory: Optional[AtomsFactory] = None,
        atoms_post_processor: Optional[AtomsPostProcessor] = None,
    ):
        """
        Initialize an EnergyObjectiveFunction.

        Args:
            calc_factory: Factory to create an ASE calculator.
            param_applier: Function to apply parameters.
            path_to_reference_configuration: Path to the reference configuration.
            reference_energy: Target energy for the objective.
            tag: Optional label for this objective.
            weight: Base weight for the error term.
            weight_cb: Optional weight-scaling callback.
            atoms_factory: Optional function to process the Atoms object after loading.
            atoms_post_processor: Optional function to process the Atoms object after loading.
        """
        self.reference_energy = reference_energy
        super().__init__(
            calc_factory=calc_factory,
            param_applier=param_applier,
            path_to_reference_configuration=path_to_reference_configuration,
            tag=tag,
            weight=weight,
            weight_cb=weight_cb,
            atoms_factory=atoms_factory,
            atoms_post_processor=atoms_post_processor,
        )

    def get_meta_data(self) -> dict[str, Any]:
        """
        Extend parent metadata with reference energy.

        Returns:
            dict[str, Any]: Metadata from the parent, plus:
                reference_energy: Target reference energy.
        """
        data = super().get_meta_data()
        data["reference_energy"] = self.reference_energy

        return data

    def __call__(self, parameters: dict) -> float:
        """
        Compute squared-error contribution to the objective:
        (E_computed(parameters) - E_reference)^2 * weight.

        Args:
            parameters: Parameter names to values; applied before energy evaluation.

        Returns:
            float: Weighted squared difference between computed and reference energies.
        """

        energy = self.compute_energy(parameters)
        error = (energy - self.reference_energy) ** 2
        objective_contribution = error * self.weight
        logger.debug(f"Parameters applied: {parameters}")
        logger.debug(f"Computed energy: {energy}")
        logger.debug(f"Reference energy: {self.reference_energy}")
        logger.debug(f"Weight: {self.weight}")
        logger.debug(f"Objective contribution: {objective_contribution}")
        return objective_contribution


class DimerDistanceObjectiveFunction(ASEObjectiveFunction):
    """
    Objective that relaxes a water dimer and compares its O-O distance to a target.
    """

    def __init__(
        self,
        calc_factory: CalculatorFactory,
        param_applier: ParameterApplier,
        reference_OO_distance: float,
        path_to_reference_configuration: Optional[Path] = None,
        dt: float = 1e-2,
        fmax: float = 1e-5,
        max_steps: int = 2000,
        noise_magnitude: float = 0.0,
        tag: Optional[str] = None,
        weight: float = 1.0,
        weight_cb: Optional[Callable[[Atoms], float]] = None,
        atoms_factory: Optional[AtomsFactory] = None,
        atoms_post_processor: Optional[AtomsPostProcessor] = None,
    ):
        """
        Initialize a DimerDistanceObjectiveFunction.

        Args:
            calc_factory: Factory to create an ASE calculator.
            param_applier: Function to apply calculator parameters.
            path_to_reference_configuration: Path to the water dimer configuration.
            reference_OO_distance: Target O-O distance.
            dt: Time step for relaxation.
            fmax: Force convergence criterion.
            max_steps: Maximum optimizer steps.
            noise_magnitude: Amplitude of random noise added to positions.
            tag: Optional label for this objective.
            weight: Base weight for the error term.
            weight_cb: Optional weight-scaling callback.
            atoms_factory: Optional function to create Atoms object.
            atoms_post_processor: Optional function to process the Atoms object after loading.
        """
        self.reference_OO_distance = reference_OO_distance
        self.dt = dt
        self.fmax = fmax
        self.max_steps = max_steps
        self.noise_magnitude = noise_magnitude
        super().__init__(
            calc_factory=calc_factory,
            param_applier=param_applier,
            path_to_reference_configuration=path_to_reference_configuration,
            tag=tag,
            weight=weight,
            weight_cb=weight_cb,
            atoms_factory=atoms_factory,
            atoms_post_processor=atoms_post_processor,
        )
        self.positions_reference = np.array(self.atoms.positions)

    def get_meta_data(self) -> dict[str, Any]:
        """
        Extend metadata with current and target O-O distances.

        Returns:
            dict[str, Any]: Metadata including:
            oo_distance: Current relaxed O-O distance.
            reference_OO_distance: Target O-O distance.
        """
        data = super().get_meta_data()
        data["oo_distance"] = getattr(self, "OO_distance", 0.0)
        data["reference_OO_distance"] = self.reference_OO_distance
        return data

    def __call__(self, parameters: dict) -> float:
        """
        Apply parameters, optionally add noise, relax the dimer, and compute error.

        Args:
            parameters: dict of parameter names to float values.

        Returns:
            float: Weighted squared difference between relaxed and target O-O distances.
        """
        self.param_applier(self.atoms, parameters)
        self.atoms.set_velocities(np.zeros((self.n_atoms, 3)))
        self.atoms.set_positions(self.positions_reference)
        self.atoms.calc.calculate(self.atoms)
        self.atoms.positions += self.noise_magnitude * np.random.uniform(
            -1.0, 1.0, size=self.atoms.positions.shape
        )
        optimizer = BFGS(self.atoms)
        optimizer.run(fmax=self.fmax, steps=self.max_steps)
        self.OO_distance = self.atoms.get_distance(0, 3, mic=True)
        logger.debug(f"Relaxed O-O distance: {self.OO_distance}")
        diff = self.OO_distance - self.reference_OO_distance
        objective_value = self.weight * diff**2
        logger.debug(f"Parameters applied: {parameters}")
        logger.debug(f"Target O-O distance: {self.reference_OO_distance}")
        logger.debug(f"Weight: {self.weight}")
        logger.debug(f"Objective value: {objective_value}")
        return objective_value
