from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read
from ase.optimize import BFGS

from chemfit.abstract_objective_function import QuantityComputer
from chemfit.exceptions import FactoryException

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class CalculatorFactory(Protocol):
    """Protocol for a factory that constructs an ASE calculator in-place and attaches it to `atoms`."""

    def __call__(self, atoms: Atoms) -> None:
        """Construct a calculator and overwrite `atoms.calc`."""
        ...


class ParameterApplier(Protocol):
    """Protocol for a function that applies parameters to an ASE calculator."""

    def __call__(self, atoms: Atoms, params: dict[str, Any]) -> None:
        """Applies a parameter dictionary to `atoms.calc` in-place."""
        ...


class AtomsPostProcessor(Protocol):
    """Protocol for a function that post-processes an ASE Atoms object."""

    def __call__(self, atoms: Atoms) -> None:
        """Modify the atoms in-place."""
        ...


class AtomsFactory(Protocol):
    """Protocol for a function that creates an ASE Atoms object."""

    def __call__(self) -> Atoms:
        """Create an atoms object."""
        ...


class QuantitiesProcessor(Protocol):
    """Protocol for a function that returns the quantities after the `calculate` function."""

    def __call__(
        self, calc: Calculator, atoms: Atoms | None = None
    ) -> dict[str, Any]: ...


class PathAtomsFactory(AtomsFactory):
    """Implementation of AtomsFactory which reads the atoms from a path."""

    def __init__(self, path: Path, index: int | None = None) -> None:
        """Initialize a path atoms factory."""
        self.path = path
        self.index = index

    def __call__(self) -> Atoms:
        atoms = read(self.path, self.index, parallel=False)

        if isinstance(atoms, list):
            msg = f"Index {self.index} selects multiple images from path {self.path}. This is not compatible with AtomsFactory."
            raise AtomsFactoryException(msg)

        return atoms


class CalculatorFactoryException(FactoryException): ...


class AtomsFactoryException(FactoryException): ...


class ParameterApplierException(FactoryException): ...


class AtomsPostProcessorException(FactoryException): ...


class SinglePointASEComputer(QuantityComputer):
    """
    Base class for a single point ASE-based computer.

    This class loads a reference configuration, optionally post-processes the structure,
    attaches a calculator, and provides an interface for evaluating parameters
    """

    def __init__(
        self,
        calc_factory: CalculatorFactory,
        param_applier: ParameterApplier,
        atoms_factory: AtomsFactory,
        tag: str | None = None,
        atoms_post_processor: AtomsPostProcessor | None = None,
        quantities_processor: QuantitiesProcessor | None = None,
    ) -> None:
        """
        Initialize an ASEObjectiveFunction.

        Args:
            calc_factory: Factory to create an ASE calculator given an `Atoms` object.
            param_applier: Function that applies a dict of parameters to `atoms.calc`.
            atoms_factory: Optional[AtomsFactory] Optional function to create the Atoms object.
            tag: Optional label for this objective. Defaults to "tag_None" if None.
            atoms_post_processor: Optional function to modify or validate the Atoms object
                immediately after loading and before attaching the calculator.
            quantities_processor: this is called after the calculate function to return the

        **Important**: One of `atoms_factory` or `path_to_reference_configuration` has to be specified.
        If both are specified `atoms_factory` takes precedence.

        """

        self.calc_factory = calc_factory
        self.param_applier = param_applier
        self.atoms_factory = atoms_factory

        self.atoms_post_processor = atoms_post_processor

        if quantities_processor is None:
            self.quantities_processor = lambda calc, atoms: {
                **calc.results,
                "n_atoms": len(atoms),
            }
        else:
            self.quantities_processor = quantities_processor

        self.tag = tag or "tag_None"

        # NOTE: You should probably use the `self.atoms` property
        # When the atoms object is requested for the first time, it will be lazily loaded via the atoms_factory
        self._atoms = None

    def get_meta_data(self) -> dict[str, Any]:
        """
        Retrieve metadata for this objective function.

        Returns:
            dict[str, Union[str, int, float]]: Dictionary containing:
                tag: User-defined label.
                n_atoms: Number of atoms in the configuration.
                weight: Final weight after any scaling.
                last_energy: The last computed energy

        """
        meta_data = super().get_meta_data()
        meta_data.update(
            {
                "tag": self.tag,
                "n_atoms": self.n_atoms,
                "type": type(self).__name__,
            }
        )
        return meta_data

    def create_atoms_object(self) -> Atoms:
        """
        Create the atoms object, check it, optionally post-processes it, and attach the calculator.

        Returns:
            Atoms: ASE Atoms object with calculator attached.

        """
        try:
            atoms = self.atoms_factory()
        except Exception as e:
            raise AtomsFactoryException from e

        if self.atoms_post_processor is not None:
            try:
                self.atoms_post_processor(atoms)
            except Exception as e:
                raise AtomsPostProcessorException from e

        try:
            self.calc_factory(atoms)
        except Exception as e:
            raise CalculatorFactoryException from e

        if atoms.calc is None:
            raise CalculatorFactoryException

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

    def _compute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Compute the quantities. This default implementation simply calls the `calculate` function and then returns the results dict from the calculator.

        Args:
            parameters: Dictionary of parameter names to float values.

        """
        assert self.atoms.calc is not None

        try:
            self.param_applier(self.atoms, parameters)
        except Exception as e:
            raise ParameterApplierException from e

        self.atoms.calc.calculate(self.atoms)

        quants = self.quantities_processor(self.atoms.calc, self.atoms)

        self._last_quantities = quants

        return quants


class MinimizationASEComputer(SinglePointASEComputer):
    """Computer based on the closes local minimum."""

    def __init__(
        self, dt: float = 1e-2, fmax: float = 1e-5, max_steps: int = 2000, **kwargs
    ) -> None:
        """
        Initialize a MinimizationASEComputer.

        All kwargs are passed to `SinglePointASEComputer.__init__`.

        Args:
            dt: Time step for relaxation.
            fmax: Force convergence criterion.
            max_steps: Maximum optimizer steps.

        """

        self.dt = dt
        self.fmax = fmax
        self.max_steps = max_steps
        super().__init__(**kwargs)

        # We load the atoms object and make a copy of its positions
        self.positions_reference = np.array(self.atoms.positions, copy=True)

    def relax_structure(self, parameters: dict[str, Any]) -> None:
        self.param_applier(self.atoms, parameters)

        self.atoms.set_velocities(np.zeros((self.n_atoms, 3)))
        self.atoms.set_positions(self.positions_reference)

        assert self.atoms.calc is not None

        self.atoms.calc.calculate(self.atoms)

        optimizer = BFGS(self.atoms, logfile=None)
        optimizer.run(fmax=self.fmax, steps=self.max_steps)

    def _compute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        # First relax the structure
        self.relax_structure(parameters=parameters)

        # Then call the single point compute function
        return super()._compute(parameters=parameters)


# class DimerDistanceObjectiveFunction(StructureObjectiveFunction):
#     """Objective function based on the oxygen-oxygen distance in a water dimer."""

#     def __init__(self, reference_OO_distance: float | None, *args, **kwargs):
#         """
#         Initialize a DimerDistanceObjectiveFunction.

#         See `StructureObjectiveFunction.__init__` for shared parameters.

#         Args:
#             reference_OO_distance: Target distance between oxygens.

#         """

#         super().__init__(*args, **kwargs)

#         if reference_OO_distance is None:
#             self.reference_OO_distance = cast(
#                 "float", self.atoms.get_distance(0, 3, mic=True)
#             )
#         else:
#             self.reference_OO_distance = reference_OO_distance

#         self._OO_distance: float | None = None

#     def get_meta_data(self) -> dict[str, Any]:
#         data = super().get_meta_data()
#         data["last_OO_distance"] = self._OO_distance
#         return data

#     def __call__(self, parameters: dict[str, Any]) -> float:
#         self.relax_structure(parameters)
#         self._OO_distance = cast(
#             "float", self.atoms.get_distance(0, 3, mic=True)
#         )  # Missing type hint in ASE
#         diff = self._OO_distance - self.reference_OO_distance
#         return self.weight * diff**2


# class KabschObjectiveFunction(StructureObjectiveFunction):
#     """Computes the objective function based on the RMS from Kabsch rotation matrix."""

#     def __init__(self, *args, **kwargs):
#         """
#         Initialize the Kabsch objective function.

#         See `StructureObjectiveFunction.__init__` for shared parameters.
#         """

#         self._kabsch_r: np.ndarray | None = None
#         self._kabsch_t: np.ndarray | None = None
#         self._kabsch_rmsd: float | None = None
#         super().__init__(*args, **kwargs)

#     def get_meta_data(self) -> dict[str, Any]:
#         data = super().get_meta_data()
#         data["last_kabsch_r"] = (
#             self._kabsch_r.tolist() if self._kabsch_r is not None else None
#         )
#         data["last_kabsch_t"] = (
#             self._kabsch_t.tolist() if self._kabsch_t is not None else None
#         )
#         data["last_kabsch_rmsd"] = self._kabsch_rmsd
#         return data

#     def __call__(self, parameters: dict[str, Any]) -> float:
#         self.relax_structure(parameters)

#         self._kabsch_r, self._kabsch_t = kabsch.kabsch(
#             self.atoms.positions, self.positions_reference
#         )

#         positions_aligned = kabsch.apply_transform(
#             self.atoms.positions, self._kabsch_r, self._kabsch_t
#         )
#         self._kabsch_rmsd = kabsch.rmsd(positions_aligned, self.positions_reference)

#         return self.weight * self._kabsch_rmsd
