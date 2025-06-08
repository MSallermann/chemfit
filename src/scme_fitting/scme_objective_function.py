from .scme_setup import (
    setup_calculator,
    SCMEParams,
    check_water_is_in_OHH_order,
    arange_water_in_OHH_order,
)
from ase import Atoms
from typing import Dict
from pathlib import Path
import logging

from typing import Any

logger = logging.getLogger(__name__)


class SCMECalculatorFactory:
    def __init__(
        self,
        default_scme_params: SCMEParams,
        path_to_scme_expansions: Path,
        parametrization_key: str,
    ):
        self.default_scme_params = default_scme_params
        self.path_to_scme_expansions = path_to_scme_expansions
        self.parametrization_key = parametrization_key

    def __call__(self, atoms: Atoms) -> Any:
        # Attach a fresh copy of default SCME parameters to this Atoms object
        if not check_water_is_in_OHH_order(atoms=atoms):
            atoms = arange_water_in_OHH_order(atoms)

        scme_params_copy = self.default_scme_params.model_copy()
        setup_calculator(
            atoms,
            scme_params=scme_params_copy,
            parametrization_key=self.parametrization_key,
            path_to_scme_expansions=self.path_to_scme_expansions,
        )


class SCMEParameterApplier:
    def __call__(self, atoms: Atoms, params: Dict[str, float]) -> None:
        """
        Assign SCME parameter values to the attached calculator.

        A
            params : Dict[str, float]
                Dictionary of SCME parameter names to float values.

        Raises
        ------
        KeyError
            If a key in `params` does not correspond to an attribute on the SCME potential.
        """
        for key, value in params.items():
            if hasattr(atoms.calc.scme, key):
                setattr(atoms.calc.scme, key, value)
            else:
                raise KeyError(
                    f"Cannot set parameter '{key}': "
                    "not a valid attribute on the SCME potential."
                )
