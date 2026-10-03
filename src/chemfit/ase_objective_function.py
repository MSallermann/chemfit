from __future__ import annotations

import copy
import functools
import threading
from typing import TYPE_CHECKING, Any, Generic, Protocol, cast, runtime_checkable

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read
from ase.optimize import BFGS

# Python 3.10's typing.Concatenate rejects the ellipsis used in our aliases.
from typing_extensions import Concatenate, Self  # noqa: UP035

from chemfit.abstract_objective_function import (
    EvaluateContext,
    ParametersT_contra,
    QuantitiesT_co,
    QuantityComputer,
)
from chemfit.utils import check_protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path


@runtime_checkable
class CalculatorFactory(Protocol[ParametersT_contra]):
    """Create an ASE calculator for one evaluation."""

    def __call__(
        self,
        parameters: ParametersT_contra,
        atoms: Atoms,
        ctx: EvaluateContext,
        /,
    ) -> Calculator:
        """Return a calculator configured for ``parameters`` and ``atoms``."""
        ...


@runtime_checkable
class ASEEvaluator(Protocol[ParametersT_contra]):
    """Run an ASE calculation before quantities are extracted."""

    def __call__(
        self,
        parameters: ParametersT_contra,
        atoms: Atoms,
        ctx: EvaluateContext,
        /,
    ) -> None:
        """Evaluate ``atoms`` using the configured calculator."""
        ...


@runtime_checkable
class AtomsSetup(Protocol):
    """Configure the cached base atoms structure."""

    def __call__(self, atoms: Atoms, /) -> None:
        """Modify ``atoms`` in place before it is cached."""
        ...


@runtime_checkable
class AtomsFactory(Protocol):
    """Create an ASE atoms object."""

    def __call__(self) -> Atoms:
        """Create an atoms object."""
        ...


@runtime_checkable
class QuantityProcessor(Protocol[QuantitiesT_co]):
    """Extract quantities after an ASE evaluation."""

    def __call__(
        self,
        calc: Calculator,
        atoms: Atoms,
        ctx: EvaluateContext,
        /,
    ) -> QuantitiesT_co:
        """Return quantities extracted from the evaluated atoms and calculator."""
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
        """Read and return one atoms object."""
        atoms = read(self.path, self.index, parallel=False)

        if isinstance(atoms, list):
            msg = (
                f"Index {self.index} selects multiple images from path "
                f"{self.path}. This is not compatible with AtomsFactory."
            )
            raise Exception(msg)

        return atoms


class DefaultQuantityProcessor:
    """Return the calculator results together with the atom count."""

    def __init__(self, filter_keys: list[str] | None = None) -> None:
        """
        Initialize the processor.

        Args:
            filter_keys: Optional keys to omit from the returned quantities.

        """
        self.filter_keys = filter_keys

    def __call__(
        self,
        calc: Calculator,
        atoms: Atoms,
        _ctx: EvaluateContext,
    ) -> dict[str, Any]:
        """Return the available calculator results and atom count."""
        result = {**calc.results, "n_atoms": len(atoms)}
        if self.filter_keys is not None:
            for key in self.filter_keys:
                result.pop(key)
        return result


def _single_point_evaluator(
    _parameters: object,
    atoms: Atoms,
    _ctx: EvaluateContext,
) -> None:
    """Run the default single-point calculation."""
    assert atoms.calc is not None
    atoms.calc.calculate(atoms)


def _minimize_evaluator(
    _parameters: object,
    atoms: Atoms,
    _ctx: EvaluateContext,
    *,
    fmax: float,
    max_steps: int,
) -> None:
    """Relax ``atoms`` with ASE BFGS."""
    optimizer = BFGS(atoms, logfile=None)
    optimizer.run(fmax=fmax, steps=max_steps)


class ASEComputer(
    QuantityComputer[ParametersT_contra, QuantitiesT_co],
    Generic[ParametersT_contra, QuantitiesT_co],
):
    """Compute quantities using one configurable ASE evaluation procedure."""

    def __init__(
        self,
        atoms_factory: AtomsFactory,
        calculator_factory: CalculatorFactory[ParametersT_contra] | None = None,
        atoms_setups: Iterable[AtomsSetup] | None = None,
        quantity_processors: Iterable[QuantityProcessor[QuantitiesT_co]] | None = None,
        evaluator: ASEEvaluator[ParametersT_contra] | None = None,
    ) -> None:
        """
        Initialize an ASE computer.

        The base atoms object is created lazily and cached. Each evaluation
        receives a copy of that structure, a fresh calculator, and the current
        evaluation context. The evaluator runs before quantity extraction.

        Args:
            atoms_factory: Callable that creates the base atoms object.
            calculator_factory: Optional callable that returns a fresh
                calculator for the current parameters, atoms, and context. It
                may instead be configured later with :meth:`with_calculator`.
            atoms_setups: Optional callbacks applied once to the base atoms
                object before it is cached.
            quantity_processors: Optional callbacks that extract quantities
                after evaluation. If omitted or empty, a default processor
                returns calculator results and the atom count.
            evaluator: Evaluation procedure. Defaults to a single-point
                calculator evaluation.

        """
        super().__init__()

        check_protocol(atoms_factory, AtomsFactory)
        check_protocol(calculator_factory, CalculatorFactory)
        check_protocol(evaluator, ASEEvaluator)

        self.atoms_factory = atoms_factory
        self.calculator_factory = calculator_factory
        self.atoms_setups = tuple(atoms_setups or ())
        for setup in self.atoms_setups:
            check_protocol(setup, AtomsSetup)

        self.quantity_processors = tuple(quantity_processors or ())

        for processor in self.quantity_processors:
            check_protocol(processor, QuantityProcessor)

        self.evaluator = (
            cast("ASEEvaluator[ParametersT_contra]", _single_point_evaluator)
            if evaluator is None
            else evaluator
        )

        self._atoms: Atoms | None = None
        self._atoms_init_lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state without the non-pickleable initialization lock."""
        state = self.__dict__.copy()
        state.pop("_atoms_init_lock")
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state and create a lock local to this process."""
        self.__dict__.update(state)
        self._atoms_init_lock = threading.Lock()

    def with_atoms_setup(self, setup: AtomsSetup, /) -> Self:
        """
        Return a copy with an additional base-atoms setup callback.

        Because the callback changes construction of the cached structure, the
        returned computer starts with a fresh atoms cache and initialization
        lock. The source computer is unchanged.
        """
        check_protocol(setup, AtomsSetup)
        new = copy.copy(self)
        new.atoms_setups = (*self.atoms_setups, setup)
        new._atoms = None  # noqa: SLF001
        new._atoms_init_lock = threading.Lock()  # noqa: SLF001
        return new

    def with_calculator(
        self,
        calculator: Callable[
            Concatenate[ParametersT_contra, Atoms, EvaluateContext, ...], Calculator
        ],
        /,
        **kwargs: Any,
    ) -> Self:
        """
        Return a copy configured with a calculator factory.

        Additional keyword-only arguments are bound to ``calculator``. This
        operation does not invalidate an initialized base-atoms cache.
        """
        check_protocol(calculator, CalculatorFactory)
        new = copy.copy(self)
        new.calculator_factory = cast(
            "CalculatorFactory[ParametersT_contra]",
            functools.partial(calculator, **kwargs),
        )
        return new

    def with_processor(
        self,
        processor: Callable[
            Concatenate[Calculator, Atoms, EvaluateContext, ...], QuantitiesT_co
        ],
        /,
        **kwargs: Any,
    ) -> Self:
        """
        Return a copy with an additional quantity processor.

        Additional keyword-only arguments are bound to ``processor``. This
        operation does not invalidate an initialized base-atoms cache. Adding
        the first explicit processor replaces the implicit default-processor
        fallback.
        """
        check_protocol(processor, QuantityProcessor)
        bound_processor = cast(
            "QuantityProcessor[QuantitiesT_co]",
            functools.partial(processor, **kwargs),
        )
        new = copy.copy(self)
        new.quantity_processors = (
            *self.quantity_processors,
            bound_processor,
        )
        return new

    def with_evaluator(
        self,
        evaluator: Callable[
            Concatenate[ParametersT_contra, Atoms, EvaluateContext, ...], None
        ],
        /,
        **kwargs: Any,
    ) -> Self:
        """
        Return a copy configured with an ASE evaluation procedure.

        Additional keyword-only arguments are bound to ``evaluator``. Exactly
        one evaluator is active; this method replaces the previous evaluator
        without invalidating the base-atoms cache.
        """
        check_protocol(evaluator, ASEEvaluator)
        new = copy.copy(self)
        new.evaluator = cast(
            "ASEEvaluator[ParametersT_contra]",
            functools.partial(evaluator, **kwargs),
        )
        return new

    def minimize(self, fmax: float = 1e-5, max_steps: int = 2000) -> Self:
        """
        Return a copy configured to relax atoms with ASE BFGS.

        Args:
            fmax: Force convergence threshold passed to ``BFGS.run``.
            max_steps: Maximum optimization steps passed to ``BFGS.run``.

        """
        return self.with_evaluator(
            _minimize_evaluator,
            fmax=fmax,
            max_steps=max_steps,
        )

    def prepare_ctx(
        self,
        parameters: ParametersT_contra,
        ctx: EvaluateContext,
    ) -> None:
        """
        Populate the evaluation context with copied atoms and a calculator.

        The base structure is initialized at most once per process, including
        under concurrent thread evaluation. Each evaluation receives a copy,
        while calculator construction remains evaluation-local.
        """
        atoms = self._atoms
        if atoms is None:
            with self._atoms_init_lock:
                atoms = self._atoms
                if atoms is None:
                    atoms = self.atoms_factory()
                    for setup in self.atoms_setups:
                        setup(atoms)
                    self._atoms = atoms

        ctx.temp.atoms = atoms.copy()
        if self.calculator_factory is None:
            msg = (
                "ASEComputer requires a calculator. Configure one with "
                "with_calculator()."
            )
            raise RuntimeError(msg)
        ctx.temp.atoms.calc = self.calculator_factory(parameters, ctx.temp.atoms, ctx)

    def _compute(
        self,
        parameters: ParametersT_contra,
        ctx: EvaluateContext,
    ) -> QuantitiesT_co:
        """Prepare, evaluate, and extract quantities from an ASE calculation."""
        self.prepare_ctx(parameters, ctx)
        atoms = ctx.temp.atoms
        self.evaluator(parameters, atoms, ctx)

        assert atoms.calc is not None
        quantities: dict[str, Any] = {}

        if len(self.quantity_processors) == 0:
            processors = [DefaultQuantityProcessor()]
        else:
            processors = self.quantity_processors

        for proc in processors:
            quantities.update(proc(atoms.calc, atoms, ctx))

        return cast("QuantitiesT_co", quantities)
