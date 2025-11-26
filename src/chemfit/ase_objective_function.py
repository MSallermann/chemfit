from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read
from ase.optimize import BFGS

from chemfit.abstract_objective_function import EvaluateContext, QuantityComputer
from chemfit.utils import check_protocol

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@runtime_checkable
class CalculatorFactory(Protocol):
    """Protocol for a factory that constructs an ASE calculator in-place and attaches it to `atoms`."""

    def __call__(self, atoms: Atoms) -> None:
        """Construct a calculator and overwrite `atoms.calc`."""
        ...


@runtime_checkable
class ParameterApplier(Protocol):
    """Protocol for a function that applies parameters to an ASE calculator."""

    def __call__(self, atoms: Atoms, params: dict[str, Any]) -> None:
        """Applies a parameter dictionary to `atoms.calc` in-place."""
        ...


@runtime_checkable
class AtomsPostProcessor(Protocol):
    """Protocol for a function that post-processes an ASE Atoms object."""

    def __call__(self, atoms: Atoms) -> None:
        """Modify the atoms in-place."""
        ...


@runtime_checkable
class AtomsFactory(Protocol):
    """Protocol for a function that creates an ASE Atoms object."""

    def __call__(self) -> Atoms:
        """Create an atoms object."""
        ...


@runtime_checkable
class QuantityProcessor(Protocol):
    """Protocol for a function that returns the quantities after the `calculate` function."""

    def __call__(self, calc: Calculator, atoms: Atoms) -> dict[str, Any]: ...


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
            raise Exception(msg)

        return atoms


def default_quantity_processor(calc: Calculator, atoms: Atoms) -> dict[str, Any]:
    return {**calc.results, "n_atoms": len(atoms)}


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
        atoms_post_processor: AtomsPostProcessor | None = None,
        quantity_processors: list[QuantityProcessor] | None = None,
        tag: str | None = None,
    ) -> None:
        """
        Initialize a SinglePointASEComputer.

        Args:
            calc_factory: Factory to create an ASE calculator given an `Atoms` object.
            param_applier: Function that applies a dict of parameters to `atoms.calc`.
            atoms_factory: Function to create the Atoms object.
            atoms_post_processor: Optional function to modify or validate the Atoms object
                immediately after loading and before attaching the calculator.
            quantities_processors: list of functions called after the calculate function to update the quantities dictionary
            tag: Optional label for this computer. Defaults to "tag_None" if None.

        """

        super().__init__()

        # Make sure all the protocols are properly implemented
        check_protocol(calc_factory, CalculatorFactory)
        check_protocol(param_applier, ParameterApplier)
        check_protocol(atoms_factory, AtomsFactory)
        check_protocol(atoms_post_processor, AtomsPostProcessor)

        self.calc_factory = calc_factory
        self.param_applier = param_applier
        self.atoms_factory = atoms_factory
        self.atoms_post_processor = atoms_post_processor

        self.quantity_processors: list[QuantityProcessor] = [default_quantity_processor]

        if quantity_processors is not None:
            self.quantity_processors += quantity_processors

        for qp in self.quantity_processors:
            check_protocol(qp, QuantityProcessor)

        self.tag = tag or "tag_None"

        self._atoms: Atoms | None = None

        self.static_meta_data = {
            "tag": self.tag,
            "type": type(self).__name__,
        }

    def create_atoms_object(self) -> Atoms:
        """
        Create the atoms object, check it, optionally post-processes it, and attach the calculator.

        Returns:
            Atoms: ASE Atoms object with calculator attached.

        """

        atoms = self.atoms_factory()

        if self.atoms_post_processor is not None:
            self.atoms_post_processor(atoms)

        self.calc_factory(atoms)

        return atoms

    def prepare_ctx(self, parameters: dict[str, Any], ctx: EvaluateContext):
        if self._atoms is None:
            self._atoms = self.atoms_factory()

        # Since the calculation may change the internal state of the calculator
        # we create a new calculator and a new atoms object in the context
        ctx.temp.atoms = self._atoms.copy()
        self.calc_factory(ctx.temp.atoms)
        self.param_applier(ctx.temp.atoms, parameters)

    def _compute(
        self,
        parameters: dict[str, Any],
        ctx: EvaluateContext,
    ) -> dict[str, Any]:
        """
        Compute the quantities. This default implementation simply calls the `calculate` function and then returns the results dict from the calculator.

        Args:
            parameters: Dictionary of parameter names to float values.

        """

        self.prepare_ctx(parameters, ctx)

        assert ctx.temp.atoms.calc is not None
        ctx.temp.atoms.calc.calculate(ctx.temp.atoms)

        quants = {}
        for qp in self.quantity_processors:
            quants.update(qp(ctx.temp.atoms.calc, ctx.temp.atoms))

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

    def _compute(
        self, parameters: dict[str, Any], ctx: EvaluateContext
    ) -> dict[str, Any]:
        self.prepare_ctx(parameters, ctx)

        optimizer = BFGS(ctx.temp.atoms, logfile=None)
        optimizer.run(fmax=self.fmax, steps=self.max_steps)

        quants = {}
        for qp in self.quantity_processors:
            quants.update(qp(ctx.temp.atoms.calc, ctx.temp.atoms))

        return quants
