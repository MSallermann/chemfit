from .scme_objective_function import SCMEObjectiveFunction
from .scme_setup import SCMEParams
from ase import Atoms
from typing import Optional, Callable, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SCMEEnergyObjectiveFunction(SCMEObjectiveFunction):
    """
    SCME-based objective function that penalizes deviation of computed energy
    from a reference energy. The contribution is weight * (E_computed - E_reference)^2,
    optionally normalized by the number of atoms, and weighted.
    """

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
