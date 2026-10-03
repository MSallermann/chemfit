from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator

from chemfit.abstract_objective_function import EvaluateContext

from .scme_setup import (
    arrange_water_in_ohh_order,
    check_water_is_in_ohh_order,
    setup_calculator,
)


class SCMECalculatorFactory:
    def __init__(
        self,
        default_scme_params: dict[str, Any],
        path_to_scme_expansions: Path | None,
        parametrization_key: str | None,
    ) -> None:
        """Create an SCME calculator."""
        self.default_scme_params = default_scme_params
        self.path_to_scme_expansions = path_to_scme_expansions
        self.parametrization_key = parametrization_key

    def __call__(
        self,
        parameters: dict[str, Any],
        atoms: Atoms,
        _ctx: EvaluateContext,
    ) -> Calculator:
        """Return a fresh SCME calculator configured for one evaluation."""
        if not check_water_is_in_ohh_order(atoms=atoms):
            atoms = arrange_water_in_ohh_order(atoms)

        calculator = setup_calculator(
            atoms,
            params=self.default_scme_params,
            parametrization_key=self.parametrization_key,
            path_to_scme_expansions=self.path_to_scme_expansions,
        )
        calculator.apply_params(parameters)
        return calculator
