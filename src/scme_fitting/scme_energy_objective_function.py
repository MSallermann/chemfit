from .scme_objective_function import SCMEObjectiveFunction

from .scme_setup import SCMEParams
from ase import Atoms
from typing import Optional, Callable
from pathlib import Path


import logging

logger = logging.getLogger(__name__)


class SCMEEnergyObjectiveFunction(SCMEObjectiveFunction):
    def __init__(
        self,
        default_scme_params: SCMEParams,
        parametrization_key: str,
        path_to_scme_expansions: Optional[Path],
        path_to_reference_configuration: Path,
        reference_energy: float,
        divide_by_n_atoms: bool = False,
        tag: Optional[str] = None,
        weight: float = 1.0,
        weight_cb: Optional[Callable[[Atoms], float]] = None,
    ):
        super().__init__(
            default_scme_params,
            parametrization_key,
            path_to_scme_expansions,
            path_to_reference_configuration,
            tag,
            weight,
            weight_cb,
        )

        # Save the reference energy
        self.reference_energy = reference_energy

        # Divide the weight by n_atoms if desired
        self.divide_by_n_atoms = divide_by_n_atoms

        if self.divide_by_n_atoms:
            n_atoms = len(self.atoms)
            self.weight /= n_atoms

    def get_meta_data(self):
        data = super().get_meta_data()
        data["reference_energy"] = self.reference_energy
        return data

    def __call__(self, parameters: dict):
        """
        Compute squared-error contribution to the objective function.

        Parameters
        ----------

        parameters : Dict[str, float]
            SCME parameter values to apply.

        Returns
        -------
        float
            Squared difference between computed and reference energy.
        """

        energy = self.get_energy(parameters)

        target_energy = self.reference_energy
        objective_function_contribution = (energy - target_energy) ** 2 * self.weight

        logger.debug(f"Current params = {parameters}")
        logger.debug(f"Current weight = {self.weight}")
        logger.debug(f"Objective function value = {objective_function_contribution}")

        return objective_function_contribution
