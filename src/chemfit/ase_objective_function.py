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
    """
    Protocol for a callable that attaches an ASE calculator to atoms.

    Implementations are expected to construct or configure a calculator
    for the given ``Atoms`` object and assign it to ``atoms.calc``.
    """

    def __call__(self, atoms: Atoms) -> None:
        """Construct a calculator and overwrite `atoms.calc`."""
        ...


@runtime_checkable
class ParameterApplier(Protocol):
    """Protocol for a callable that applies parameters to an ASE calculator."""

    def __call__(self, atoms: Atoms, params: dict[str, Any]) -> None:
        """Applies a parameter dictionary to `atoms.calc` in-place."""
        ...


@runtime_checkable
class AtomsPostProcessor(Protocol):
    """Protocol for a callable that post-processes an ASE Atoms object."""

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
    """
    Protocol for a callable that extracts quantities from an ASE evaluation.

    A quantity processor is called after the calculator has evaluated an
    ``Atoms`` object. It receives the calculator and atoms pair and
    returns a dictionary of quantities to include in the final output.
    """

    def __call__(self, calc: Calculator, atoms: Atoms) -> dict[str, Any]:
        """
        Extract quantities from an evaluated calculator and atoms pair.

        Args:
            calc: Calculator that has already evaluated ``atoms``.
            atoms: Evaluated atoms object.

        Returns:
            A dictionary of extracted quantities.

        """

        ...


class PathAtomsFactory(AtomsFactory):
    """Atoms factory that reads a single structure from a filesystem path."""

    def __init__(self, path: Path, index: int | None = None) -> None:
        """
        Initialize the factory.

        Args:
            path: Path to a structure file readable by ASE.
            index: Optional ASE index selecting which image to read. The
                selection must resolve to a single ``Atoms`` object.

        """

        self.path = path
        self.index = index

    def __call__(self) -> Atoms:
        atoms = read(self.path, self.index, parallel=False)

        if isinstance(atoms, list):
            msg = f"Index {self.index} selects multiple images from path {self.path}. This is not compatible with AtomsFactory."
            raise Exception(msg)

        return atoms


class DefaultQuantityProcessor:
    def __init__(self, filter_keys: list[str] | None = None) -> None:
        """
        Initialize a default quantity processor, that returns all of the `results` of the calculator.

        The returned quantity dictionary contains all entries from
        ``calc.results`` plus ``"n_atoms"``. Any keys listed in
        ``filter_keys`` are excluded from the returned dictionary.

        Args:
            filter_keys: Optional list of keys to exclude from the returned
                quantity dictionary.

        """

        self.filter_keys = filter_keys

    def __call__(self, calc: Calculator, atoms: Atoms) -> dict[str, Any]:
        res = {**calc.results, "n_atoms": len(atoms)}
        if self.filter_keys is not None:
            [res.pop(k) for k in self.filter_keys]
        return res


class SinglePointASEComputer(QuantityComputer):
    """
    ASE-based quantity computer for single-point evaluations.

    This class evaluates quantities for a parameterized ASE calculation
    using an atoms factory, optional atoms post-processing, a
    calculator factory, a parameter applier, and one or more quantity
    processors.
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
        Initialize the computer.

        Args:
            calc_factory: Callable that attaches a calculator to an
                ``Atoms`` object.
            param_applier: Callable that applies a parameter dictionary to
                the calculator attached to an ``Atoms`` object.
            atoms_factory: Callable that creates the base ``Atoms`` object.
            atoms_post_processor: Optional callable that modifies the base
            atoms object before it is cached and copied for evaluation.
            quantity_processors: Optional list of callables that extract
                quantities from the evaluated calculator and atoms pair. If
                ``None``, a ``DefaultQuantityProcessor`` is used.
            tag: Optional label for this computer. If ``None``,
                ``"tag_None"`` is used.

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

        if quantity_processors is None:
            self.quantity_processors: list[QuantityProcessor] = [
                DefaultQuantityProcessor()
            ]
        else:
            self.quantity_processors = quantity_processors

        for qp in self.quantity_processors:
            check_protocol(qp, QuantityProcessor)

        self.tag = tag or "tag_None"

        self._atoms: Atoms | None = None

        self.static_meta_data = {
            "tag": self.tag,
            "type": type(self).__name__,
        }

    def prepare_ctx(self, parameters: dict[str, Any], ctx: EvaluateContext):
        """
        Prepare the evaluation context for a single-point calculation.

        This method lazily creates and caches a base atoms object using
        ``atoms_factory``. If provided, ``atoms_post_processor`` is applied
        once to that base object before it is cached. For each evaluation,
        the cached atoms object is copied into ``ctx.temp.atoms``, a fresh
        calculator is attached, and the provided parameters are applied.

        Args:
            parameters: Parameter dictionary for the current evaluation.
            ctx: Evaluation context to populate.

        """

        if self._atoms is None:
            self._atoms = self.atoms_factory()
            if self.atoms_post_processor is not None:
                self.atoms_post_processor(self._atoms)

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
        Compute quantities from a single-point ASE evaluation.

        This implementation prepares the evaluation context, runs the
        calculator on the atoms object, and merges the outputs of all
        configured quantity processors.

        Args:
            parameters: Mapping of parameter names to parameter values.
            ctx: Evaluation context for the current call.

        Returns:
            A dictionary containing the merged quantities returned by the
            configured quantity processors.

        """

        self.prepare_ctx(parameters, ctx)

        assert ctx.temp.atoms.calc is not None
        ctx.temp.atoms.calc.calculate(ctx.temp.atoms)

        quants = {}
        for qp in self.quantity_processors:
            quants.update(qp(ctx.temp.atoms.calc, ctx.temp.atoms))

        return quants


class MinimizationASEComputer(SinglePointASEComputer):
    """
    ASE-based quantity computer using a locally optimized structure.

    This computer evaluates quantities after performing a local geometry
    optimization using the ASE BFGS optimizer. Quantities are extracted
    from the relaxed structure using the configured quantity processors.
    """

    def __init__(
        self, dt: float = 1e-2, fmax: float = 1e-5, max_steps: int = 2000, **kwargs
    ) -> None:
        """
        Initialize a MinimizationASEComputer.

        All additional keyword arguments are forwarded to
        ``SinglePointASEComputer.__init__``.

        Args:
            dt: Relaxation step-size parameter retained for compatibility
                with earlier implementations. Currently unused.
            fmax: Force convergence criterion passed to the optimizer.
            max_steps: Maximum number of optimization steps.
            **kwargs: Additional keyword arguments forwarded to the parent
                initializer.

        """

        self.dt = dt
        self.fmax = fmax
        self.max_steps = max_steps
        super().__init__(**kwargs)

    def _compute(
        self, parameters: dict[str, Any], ctx: EvaluateContext
    ) -> dict[str, Any]:
        """
        Compute quantities after local geometry optimization.

        This method prepares the evaluation context, performs a geometry
        optimization using ASE's BFGS optimizer, and extracts quantities
        from the relaxed structure using the configured quantity
        processors.

        Args:
            parameters: Mapping of parameter names to parameter values.
            ctx: Evaluation context for the current call.

        Returns:
            A dictionary containing the merged quantities returned by the
            configured quantity processors.

        Side Effects:
            - Creates and stores an ``Atoms`` object in ``ctx.temp.atoms``.
            - Attaches a fresh calculator to the atoms object.

        """

        self.prepare_ctx(parameters, ctx)

        optimizer = BFGS(ctx.temp.atoms, logfile=None)
        optimizer.run(fmax=self.fmax, steps=self.max_steps)

        quants = {}
        for qp in self.quantity_processors:
            quants.update(qp(ctx.temp.atoms.calc, ctx.temp.atoms))

        return quants
