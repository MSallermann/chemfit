from __future__ import annotations

import copy
import functools
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Protocol, cast, runtime_checkable

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read
from ase.optimize import BFGS
from ase.optimize.optimize import Optimizer

# Python 3.10's typing.Concatenate rejects the ellipsis used in our aliases.
from typing_extensions import Concatenate, Self  # noqa: UP035

from chemfit.abstract_objective_function import (
    EvaluateContext,
    ParametersT_contra,
    QuantitiesT_co,
    QuantityComputer,
)
from chemfit.evaluation_working_directory import EvaluationWorkingDirectory
from chemfit.utils import check_protocol, merge_quantities

_OPTIMIZER_FILE_SETTINGS = ("logfile", "trajectory", "restart")

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


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
class AtomsModifier(Protocol[ParametersT_contra]):
    """Modify the per-evaluation atoms copy using the current parameters."""

    def __call__(
        self,
        parameters: ParametersT_contra,
        atoms: Atoms,
        ctx: EvaluateContext,
        /,
    ) -> None:
        """Modify ``atoms`` in place for the current evaluation."""
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

    def __init__(
        self,
        path: Path | str,
        index: int | slice | str | None = None,
    ) -> None:
        """
        Initialize the factory.

        Args:
            path: Path to a structure file readable by ASE.
            index: Optional ASE index selecting which image to read, including
                integer, slice, and ASE string forms. The selection must resolve
                to exactly one ``Atoms`` object.

        """
        self.path = Path(path)
        self.index = index

    def __call__(self) -> Atoms:
        """Read and return one atoms object."""
        if not self.path.exists():
            msg = f"ASE structure file does not exist: {self.path}"
            raise FileNotFoundError(msg)
        if not self.path.is_file():
            msg = f"ASE structure path is not a file: {self.path}"
            raise ValueError(msg)

        try:
            atoms = read(self.path, self.index, parallel=False)
        except Exception as exception:
            msg = (
                f"ASE could not read one structure from {self.path} "
                f"with index {self.index!r}: {exception}"
            )
            raise ValueError(msg) from exception

        if isinstance(atoms, list):
            if len(atoms) == 1:
                return atoms[0]
            msg = (
                f"ASE index {self.index!r} selected {len(atoms)} structures "
                f"from {self.path}; expected exactly one"
            )
            raise ValueError(msg)

        return atoms


class DefaultQuantityProcessor:
    """Return calculator results and the atom count, optionally omitting keys."""

    def __init__(self, filter_keys: list[str] | None = None) -> None:
        """
        Initialize the processor.

        Args:
            filter_keys: Optional keys to omit from the returned quantities.
                Filtering is applied after ``"n_atoms"`` is added, so that
                key may also be omitted. Every named key must be present in
                the combined result.

        Raises:
            KeyError: If a key requested in ``filter_keys`` is absent when the
                processor is called.

        """
        self.filter_keys = filter_keys

    def __call__(
        self,
        calc: Calculator,
        atoms: Atoms,
        _ctx: EvaluateContext,
    ) -> dict[str, Any]:
        """Return calculator results and atom count after filtering."""
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
    optimizer: type[Optimizer],
    optimizer_kwargs: dict[str, Any],
    fmax: float,
    max_steps: int,
    allow_unconverged: bool,
) -> None:
    """Relax ``atoms`` with an ASE optimizer."""

    optimizer_settings: dict[str, Any] = {"logfile": None, **optimizer_kwargs}
    workdir = getattr(_ctx.temp, "workdir", None)
    for name in _OPTIMIZER_FILE_SETTINGS:
        value = optimizer_settings.get(name)
        if isinstance(value, Path):
            # ``minimize()`` uses Path as a marker for validated filenames.
            # Resolve them here, after this evaluation's unique directory exists.
            assert workdir is not None
            output_path = workdir / value
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # ASE distinguishes filenames from trajectory-like objects with
            # ``isinstance(value, str)``, so pass resolved paths as strings.
            optimizer_settings[name] = str(output_path)

    dynamics = optimizer(atoms, **optimizer_settings)
    converged = dynamics.run(fmax=fmax, steps=max_steps)
    final_fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max(initial=0.0))
    # Keep the diagnostic even when non-convergence raises below; unlike
    # ``ctx.temp``, metadata survives in evaluation summaries and schedulers.
    _ctx.meta["final_fmax"] = final_fmax
    if converged:
        return

    msg = (
        f"ASE {optimizer.__name__} did not converge within "
        f"{max_steps} steps at fmax={fmax} (final fmax={final_fmax}). "
        "Increase max_steps, relax fmax, "
        "or pass allow_unconverged=True to process the unconverged structure."
    )
    if not allow_unconverged:
        raise RuntimeError(msg)


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
        atoms_modifiers: Iterable[AtomsModifier[ParametersT_contra]] | None = None,
        quantity_processors: Iterable[QuantityProcessor[QuantitiesT_co]] | None = None,
        evaluator: ASEEvaluator[ParametersT_contra] | None = None,
        *,
        evaluation_working_directory: EvaluationWorkingDirectory | None = None,
        allow_quantity_overwrites: bool = False,
    ) -> None:
        """
        Initialize an ASE computer.

        The base atoms object is created lazily and cached. Each evaluation
        receives a copy of that structure, applies its atoms modifiers, and
        attaches a fresh calculator. The evaluator then runs before quantity
        extraction.

        Args:
            atoms_factory: Callable that creates the base atoms object.
            calculator_factory: Optional callable that returns a fresh
                calculator for the current parameters, atoms, and context. It
                may instead be configured later with :meth:`with_calculator`.
            atoms_setups: Optional callbacks applied once to the base atoms
                object before it is cached.
            atoms_modifiers: Optional callbacks applied to the copied atoms
                for every evaluation, before calculator construction.
            quantity_processors: Optional callbacks that extract quantities
                after evaluation. If omitted or empty, a default processor
                returns calculator results and the atom count.
            evaluator: Evaluation procedure. Defaults to a single-point
                calculator evaluation.
            evaluation_working_directory: Optional configuration and lifecycle
                policy for per-evaluation working directories.
            allow_quantity_overwrites: Whether later quantity processors may
                replace keys returned by earlier processors. Defaults to
                ``False``.

        """
        super().__init__()

        check_protocol(atoms_factory, AtomsFactory, name="atoms_factory")
        if calculator_factory is not None:
            check_protocol(
                calculator_factory,
                CalculatorFactory,
                name="calculator_factory",
            )
        if evaluator is not None:
            check_protocol(evaluator, ASEEvaluator, name="evaluator")

        if evaluation_working_directory is not None and not isinstance(
            evaluation_working_directory, EvaluationWorkingDirectory
        ):
            msg = (
                "evaluation_working_directory must be an "
                "EvaluationWorkingDirectory or None"
            )
            raise TypeError(msg)

        self.atoms_factory = atoms_factory
        self.calculator_factory = calculator_factory
        self.atoms_setups = tuple(atoms_setups or ())
        for index, setup in enumerate(self.atoms_setups):
            check_protocol(setup, AtomsSetup, name=f"atoms_setups[{index}]")

        self.atoms_modifiers = tuple(atoms_modifiers or ())
        for index, modifier in enumerate(self.atoms_modifiers):
            check_protocol(
                modifier,
                AtomsModifier,
                name=f"atoms_modifiers[{index}]",
            )

        self.quantity_processors = tuple(quantity_processors or ())

        for index, processor in enumerate(self.quantity_processors):
            check_protocol(
                processor,
                QuantityProcessor,
                name=f"quantity_processors[{index}]",
            )

        self.evaluator = (
            cast("ASEEvaluator[ParametersT_contra]", _single_point_evaluator)
            if evaluator is None
            else evaluator
        )

        self.evaluation_working_directory = evaluation_working_directory
        self.allow_quantity_overwrites = allow_quantity_overwrites

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
        check_protocol(setup, AtomsSetup, name="setup")
        new = copy.copy(self)
        new.atoms_setups = (*self.atoms_setups, setup)
        new._atoms = None  # noqa: SLF001
        new._atoms_init_lock = threading.Lock()  # noqa: SLF001
        return new

    def with_atoms_modifier(
        self,
        modifier: Callable[
            Concatenate[ParametersT_contra, Atoms, EvaluateContext, ...], None
        ],
        /,
        **kwargs: Any,
    ) -> Self:
        """
        Return a copy with an additional per-evaluation atoms modifier.

        Additional keyword-only arguments are bound to ``modifier``. Modifiers
        run in registration order on each evaluation-local atoms copy, before
        calculator construction. This operation does not invalidate an
        initialized base-atoms cache.
        """
        check_protocol(modifier, AtomsModifier, name="modifier")
        bound_modifier = cast(
            "AtomsModifier[ParametersT_contra]",
            functools.partial(modifier, **kwargs),
        )
        new = copy.copy(self)
        new.atoms_modifiers = (*self.atoms_modifiers, bound_modifier)
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
        check_protocol(calculator, CalculatorFactory, name="calculator")
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
        check_protocol(processor, QuantityProcessor, name="processor")
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
        check_protocol(evaluator, ASEEvaluator, name="evaluator")
        new = copy.copy(self)
        new.evaluator = cast(
            "ASEEvaluator[ParametersT_contra]",
            functools.partial(evaluator, **kwargs),
        )
        return new

    def minimize(
        self,
        fmax: float = 1e-5,
        max_steps: int = 2000,
        *,
        optimizer: type[Optimizer] = BFGS,
        optimizer_kwargs: dict[str, Any] | None = None,
        allow_unconverged: bool = False,
    ) -> Self:
        """
        Return a copy configured to relax atoms with an ASE optimizer.

        Every completed optimization records the final maximum atomic force
        in ``ctx.meta["final_fmax"]``, including unconverged optimizations.

        Args:
            fmax: Force convergence threshold passed to ``optimizer.run``.
            max_steps: Maximum optimization steps passed to ``optimizer.run``.
            optimizer: ASE optimizer class. Defaults to ``BFGS``.
            optimizer_kwargs: Optional keyword arguments passed to the
                optimizer constructor. Relative ``logfile``, ``trajectory``,
                and ``restart`` paths are resolved inside the evaluation
                working directory when one is configured.
            allow_unconverged: Whether quantity processors may consume the
                final structure when the optimizer reaches ``max_steps`` without
                converging. Defaults to ``False``.

        Raises:
            ValueError: If ``fmax`` or ``max_steps`` is not positive.
            RuntimeError: During evaluation if the optimizer does not converge
                and ``allow_unconverged`` is ``False``.

        """
        if fmax <= 0:
            msg = "fmax must be greater than zero"
            raise ValueError(msg)
        if max_steps <= 0:
            msg = "max_steps must be greater than zero"
            raise ValueError(msg)

        optimizer_kwargs = dict(optimizer_kwargs or {})
        for name in _OPTIMIZER_FILE_SETTINGS:
            value = optimizer_kwargs.get(name)
            # ASE also accepts file-like and trajectory objects. Only filename
            # values need isolation and path validation; preserve other objects.
            if not isinstance(value, (Path, str)):
                continue
            # logfile='-' means stdout logging so we dont need workdir isolation
            if name == "logfile" and value == "-":
                continue
            if self.evaluation_working_directory is None:
                # A shared filename would let concurrent evaluations overwrite
                # one another, so ChemFit only manages files in isolated workdirs.
                msg = (
                    f"optimizer_kwargs[{name!r}] requires an ASE evaluation working "
                    "directory; configure evaluation_working_directory on "
                    "ASEComputer or workdir in ase_quantity()"
                )
                raise ValueError(msg)
            # Retaining a Path marks this as a validated filename. The evaluator
            # will resolve it against the directory created for that invocation.
            optimizer_kwargs[name] = self.evaluation_working_directory.relative_path(
                value,
                description=f"optimizer_kwargs[{name!r}]",
            )

        return self.with_evaluator(
            _minimize_evaluator,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            fmax=fmax,
            max_steps=max_steps,
            allow_unconverged=allow_unconverged,
        )

    def prepare_ctx(
        self,
        parameters: ParametersT_contra,
        ctx: EvaluateContext,
    ) -> None:
        """
        Populate the context with modified copied atoms and a calculator.

        The base structure is initialized at most once per process, including
        under concurrent thread evaluation. Each evaluation receives a copy,
        applies its modifiers to that copy, and constructs its calculator.
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
        for modifier in self.atoms_modifiers:
            modifier(parameters, ctx.temp.atoms, ctx)

        if self.calculator_factory is None:
            msg = (
                "ASEComputer requires a calculator. Configure one with "
                "with_calculator()."
            )
            raise RuntimeError(msg)
        ctx.temp.atoms.calc = self.calculator_factory(parameters, ctx.temp.atoms, ctx)

    def _run(
        self,
        parameters: ParametersT_contra,
        /,
        ctx: EvaluateContext,
    ) -> QuantitiesT_co:
        """Prepare, evaluate, and extract quantities from an ASE calculation."""
        if self.evaluation_working_directory is None:
            return self._evaluate(parameters, ctx)

        with self.evaluation_working_directory.isolated_workdir() as workdir:
            ctx.temp.workdir = workdir
            return self._evaluate(parameters, ctx)

    def _evaluate(
        self,
        parameters: ParametersT_contra,
        ctx: EvaluateContext,
    ) -> QuantitiesT_co:
        """Evaluate one ASE calculation inside its already-created workspace."""
        self.prepare_ctx(parameters, ctx)
        atoms = ctx.temp.atoms
        self.evaluator(parameters, atoms, ctx)

        assert atoms.calc is not None
        quantities: dict[str, Any] = {}

        if len(self.quantity_processors) == 0:
            processors = [DefaultQuantityProcessor()]
        else:
            processors = self.quantity_processors

        for index, processor in enumerate(processors):
            merge_quantities(
                quantities,
                processor(atoms.calc, atoms, ctx),
                source=f"quantity processor {index}",
                allow_overwrites=self.allow_quantity_overwrites,
            )

        return cast("QuantitiesT_co", quantities)
