from .scme_setup import SCMEParams
from .scme_objective_function import SCMEObjectiveFunction
from ase import Atoms
from ase.optimize import FIRE2
from typing import Optional, Callable, Dict
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


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
            float: Weighted squared difference: weight * (OO_distance – OO_distance_target)^2
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
