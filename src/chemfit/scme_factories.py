from pathlib import Path
from typing import Any

from ase import Atoms

from .scme_setup import (
    arrange_water_in_ohh_order,
    check_water_is_in_ohh_order,
    setup_calculator,
)


class SCMECalculatorFactory:
    def __init__(
        self,
        default_scme_params: dict[str, Any],
        path_to_scme_expansions: Path,
        parametrization_key: str,
    ) -> None:
        """Create an SCME calculator."""
        self.default_scme_params = default_scme_params
        self.path_to_scme_expansions = path_to_scme_expansions
        self.parametrization_key = parametrization_key

    def __call__(self, atoms: Atoms) -> Any:
        # Attach a fresh copy of default SCME parameters to this Atoms object
        if not check_water_is_in_ohh_order(atoms=atoms):
            atoms = arrange_water_in_ohh_order(atoms)

        setup_calculator(
            atoms,
            params=self.default_scme_params,
            parametrization_key=self.parametrization_key,
            path_to_scme_expansions=self.path_to_scme_expansions,
        )


class SCMEParameterApplier:
    def __call__(self, atoms: Atoms, params: dict[str, Any]) -> None:
        """Assign SCME parameter values to the attached calculator."""
        assert atoms.calc is not None
        atoms.calc.apply_params(params)
