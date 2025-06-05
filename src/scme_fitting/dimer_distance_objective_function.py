from .scme_setup import (
    SCMEParams,
)

from .scme_objective_function import SCMEObjectiveFunction
from ase import Atoms
from ase.optimize import FIRE2
from typing import Optional, Dict, Callable
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DimerDistanceObjectiveFunction(SCMEObjectiveFunction):
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
        weight: float = 1.0,
        weight_cb: Optional[Callable[[Atoms], float]] = None,
    ):
        # These are unused but we have to give them some value
        self.OO_distance_target = OO_distance_target

        self.dt = dt
        self.max_steps = 2000
        self.fmax = fmax
        self.n_atoms_required = 6

        self.noise_magnitude = 1e-4

        super().__init__(
            default_scme_params,
            parametrization_key,
            path_to_scme_expansions,
            path_to_reference_configuration,
            tag,
            weight,
            weight_cb,
        )

    # We modify this function so it makes sure that the atoms object is indeed a dimer
    def create_atoms_object_from_configuration(self, path_to_configuration) -> Atoms:
        atoms = super().create_atoms_object_from_configuration(path_to_configuration)
        assert len(atoms) == self.n_atoms_required
        return atoms

    def get_meta_data(self):
        data = super().get_meta_data()
        data["oo_distance"] = self.OO_distance
        data["oo_distance_target"] = self.OO_distance_target
        return data

    def __call__(self, parameters: Dict[str, float]) -> float:
        # apply the new parameters
        self.apply_parameters(parameters)
        self.atoms.calc.calculate(self.atoms)
        self.atoms.positions += self.noise_magnitude * np.random.uniform(
            -1.0, 1.0, size=self.atoms.positions.shape
        )

        # minimize the energy of the configuration
        opt = FIRE2(self.atoms, dt=self.dt)
        opt.run(fmax=self.fmax, steps=self.max_steps)

        self.OO_distance = self.atoms.get_distance(0, 3, mic=True)

        logger.debug(f"{self.OO_distance = }")

        result = self.weight * (self.OO_distance - self.OO_distance_target) ** 2

        return result
