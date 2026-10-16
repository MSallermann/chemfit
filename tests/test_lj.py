import io
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from conftest import LJAtomsFactory, construct_lj, e_lj

import chemfit
from chemfit.abstract_objective_function import EvaluateContext
from chemfit.ase_objective_function import ASEComputer
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.evaluation_working_directory import EvaluationWorkingDirectory
from chemfit.fitter import Fitter


def loss_function(quants: dict[str, Any], e_ref: float) -> float:
    return (quants["energy"] - e_ref) ** 2


def lj_ob_term(r: float, eps: float, sigma: float):
    return ASEComputer(
        atoms_factory=LJAtomsFactory(r), calculator_factory=construct_lj
    ).with_loss(loss_function, e_ref=e_lj(r, eps, sigma))


def get_ob_func(eps: float, sigma: float):
    r_min = 2.0 ** (1 / 6) * sigma
    r_list = np.linspace(0.925 * r_min, 3.0 * sigma)

    return CombinedObjectiveFunction(
        objective_functions=[lj_ob_term(r, eps, sigma) for r in r_list]
    )


def test_ase_configuration_rejects_invalid_callbacks_immediately():
    invalid: Any = None

    with pytest.raises(
        TypeError,
        match=r"`atoms_factory` must implement AtomsFactory; got NoneType",
    ):
        ASEComputer(atoms_factory=invalid)

    with pytest.raises(
        TypeError,
        match=r"`atoms_modifiers\[0\]` must implement AtomsModifier; got NoneType",
    ):
        ASEComputer(
            atoms_factory=LJAtomsFactory(1.0),
            atoms_modifiers=[invalid],
        )

    computer = ASEComputer(atoms_factory=LJAtomsFactory(1.0))
    with pytest.raises(
        TypeError,
        match=r"`processor` must implement QuantityProcessor; got NoneType",
    ):
        computer.with_processor(invalid)


class CountingLJAtomsFactory(LJAtomsFactory):
    def __init__(self, r: float) -> None:
        """Initialize a factory that records how often it is called."""
        super().__init__(r)
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            self.calls += 1
        time.sleep(0.05)
        return super().__call__()


def test_lj():
    eps = 1.0
    sigma = 1.0

    ob = get_ob_func(eps, sigma)

    initial_params = {"epsilon": 2.0, "sigma": 1.5}

    fitter = Fitter(ob, initial_params=initial_params)

    opt_params = fitter.fit_scipy()

    ctx = EvaluateContext()
    ob(opt_params, ctx)
    terms_meta_data = ctx.to_summary(recursive=True)["children"]

    assert ob.n_components() == len(terms_meta_data)
    assert np.isclose(opt_params["epsilon"], eps)
    assert np.isclose(opt_params["sigma"], sigma)


def test_base_geometry_is_initialized_once_across_threads():
    atoms_factory = CountingLJAtomsFactory(1.0)
    computer = ASEComputer(
        atoms_factory=atoms_factory,
        calculator_factory=construct_lj,
    )
    parameters = {"epsilon": 1.0, "sigma": 1.0}

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(computer, [parameters] * 8))

    assert atoms_factory.calls == 1
    assert all(np.isclose(result["energy"], results[0]["energy"]) for result in results)


def test_ase_computer_remains_pickleable():
    computer = ASEComputer(atoms_factory=LJAtomsFactory(1.0)).with_calculator(
        construct_lj
    )

    restored = pickle.loads(pickle.dumps(computer))  # noqa: S301

    assert restored._atoms_init_lock is not computer._atoms_init_lock  # noqa: SLF001
    quantities = restored({"epsilon": 1.0, "sigma": 1.0})
    assert "energy" in quantities


def test_custom_evaluator_replaces_single_point_and_shares_context():
    evaluator_calls: list[tuple[dict[str, float], Atoms, EvaluateContext]] = []

    def evaluate(
        parameters: dict[str, float],
        atoms: Atoms,
        ctx: EvaluateContext,
        *,
        shift: float,
    ) -> None:
        evaluator_calls.append((parameters, atoms, ctx))
        atoms.positions[1, 0] = parameters["distance"] + shift
        ctx.temp.evaluated = True

    def process(
        calc: Calculator,
        atoms: Atoms,
        ctx: EvaluateContext,
    ) -> dict[str, float]:
        assert calc.results == {}
        assert ctx.temp.evaluated
        return {"distance": atoms.get_distance(0, 1)}

    base = ASEComputer[dict[str, float], dict[str, float]](
        atoms_factory=LJAtomsFactory(1.0),
        calculator_factory=construct_lj,
        quantity_processors=[process],
    )
    custom = base.with_evaluator(evaluate, shift=0.25)
    ctx = EvaluateContext()

    assert custom({"epsilon": 1.0, "sigma": 1.0, "distance": 1.5}, ctx) == {
        "distance": 1.75
    }
    assert evaluator_calls == [
        ({"epsilon": 1.0, "sigma": 1.0, "distance": 1.5}, ctx.temp.atoms, ctx)
    ]
    assert custom.evaluator is not base.evaluator


def test_atoms_modifiers_lifecycle_and_constructor_configuration():
    source = Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    events: list[tuple[str, float]] = []
    received: list[tuple[dict[str, float], Atoms, EvaluateContext]] = []

    def set_distance(
        parameters: dict[str, float], atoms: Atoms, ctx: EvaluateContext
    ) -> None:
        received.append((parameters, atoms, ctx))
        events.append(("set", atoms.get_distance(0, 1)))
        atoms.positions[1, 0] = parameters["distance"]

    def shift_distance(
        _parameters: dict[str, float], atoms: Atoms, _ctx: EvaluateContext
    ) -> None:
        events.append(("shift", atoms.get_distance(0, 1)))
        atoms.positions[1, 0] += 0.25

    def make_calculator(
        parameters: dict[str, float], atoms: Atoms, ctx: EvaluateContext
    ) -> Calculator:
        events.append(("calculator", atoms.get_distance(0, 1)))
        return construct_lj(parameters, atoms, ctx)

    computer = chemfit.ase_quantity(
        source,
        calculator_factory=make_calculator,
        atoms_setups=[lambda atoms: atoms.set_pbc(False)],
        atoms_modifiers=[set_distance, shift_distance],
        quantity_processors=[
            lambda calc, atoms, _ctx: {
                "distance": atoms.get_distance(0, 1),
                "energy": float(calc.results["energy"]),
            }
        ],
    )
    first_ctx, second_ctx = EvaluateContext(), EvaluateContext()
    parameters = {"epsilon": 1.0, "sigma": 1.0, "distance": 1.5}

    assert computer(parameters, first_ctx)["distance"] == 1.75
    assert computer({**parameters, "distance": 2.5}, second_ctx)["distance"] == 2.75
    assert events == [
        ("set", 1.0),
        ("shift", 1.5),
        ("calculator", 1.75),
        ("set", 1.0),
        ("shift", 2.5),
        ("calculator", 2.75),
    ]
    assert received == [
        (parameters, first_ctx.temp.atoms, first_ctx),
        ({**parameters, "distance": 2.5}, second_ctx.temp.atoms, second_ctx),
    ]
    assert first_ctx.temp.atoms is not second_ctx.temp.atoms
    assert source.get_distance(0, 1) == 1.0
    assert computer._atoms is not None  # noqa: SLF001
    assert computer._atoms.get_distance(0, 1) == 1.0  # noqa: SLF001

    fluent = chemfit.ase_quantity(source).with_calculator(construct_lj)
    fluent(parameters)  # Initialize its base-atoms cache before copying.
    modified = fluent.with_atoms_modifier(set_distance).with_atoms_modifier(
        shift_distance
    )
    assert fluent.atoms_modifiers == ()
    assert modified._atoms is fluent._atoms  # noqa: SLF001
    assert modified(parameters)["energy"] == computer(parameters)["energy"]


def test_fluent_configuration_preserves_or_invalidates_atoms_cache():
    atoms_factory = CountingLJAtomsFactory(1.0)
    base = ASEComputer(
        atoms_factory=atoms_factory,
        calculator_factory=construct_lj,
    )
    parameters = {"epsilon": 1.0, "sigma": 1.0}
    base(parameters)

    calculator_copy = base.with_calculator(construct_lj)
    processor_copy = base.with_processor(lambda _calc, _atoms, _ctx: {"extra": 1.0})

    def single_point(
        _parameters: dict[str, float],
        atoms: Atoms,
        _ctx: EvaluateContext,
    ) -> None:
        assert atoms.calc is not None
        atoms.calc.calculate(atoms)

    evaluator_copy = base.with_evaluator(single_point)
    minimized_copy = base.minimize()

    calculator_copy(parameters)
    assert processor_copy(parameters) == {"extra": 1.0}
    evaluator_copy(parameters)

    assert atoms_factory.calls == 1
    assert calculator_copy._atoms is base._atoms  # noqa: SLF001
    assert processor_copy._atoms is base._atoms  # noqa: SLF001
    assert evaluator_copy._atoms is base._atoms  # noqa: SLF001
    assert minimized_copy._atoms is base._atoms  # noqa: SLF001
    assert base.evaluator is not evaluator_copy.evaluator
    assert base.evaluator is not minimized_copy.evaluator

    setup_copy = base.with_atoms_setup(
        lambda atoms: atoms.set_positions(atoms.positions + 1.0)
    )
    setup_ctx = EvaluateContext()
    setup_copy(parameters, setup_ctx)

    assert atoms_factory.calls == 2
    assert base._atoms is not None  # noqa: SLF001
    assert np.allclose(base._atoms.positions[0], 0.0)  # noqa: SLF001
    assert np.allclose(setup_ctx.temp.atoms.positions[0], 1.0)


def test_ase_evaluations_use_isolated_workdirs(tmp_path: Path):
    workdirs: list[Path] = []
    lock = threading.Lock()

    def make_calculator(
        parameters: dict[str, float],
        atoms: Atoms,
        ctx: EvaluateContext,
    ) -> Calculator:
        workdir = ctx.temp.workdir
        assert workdir.is_dir()
        (workdir / "calculator.input").write_text("input", encoding="utf-8")
        with lock:
            workdirs.append(workdir)
        return construct_lj(parameters, atoms, ctx)

    def process(
        calc: Calculator,
        _atoms: Atoms,
        ctx: EvaluateContext,
    ) -> dict[str, float]:
        assert (ctx.temp.workdir / "calculator.input").is_file()
        return {"energy": float(calc.results["energy"])}

    computer = chemfit.ase_quantity(
        LJAtomsFactory(1.0),
        calculator_factory=make_calculator,
        quantity_processors=[process],
        workdir=tmp_path,
    )
    assert computer.evaluation_working_directory is not None
    assert computer.evaluation_working_directory.base_working_directory == tmp_path
    parameters = {"epsilon": 1.0, "sigma": 1.0}

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(computer, [parameters] * 4))

    assert len(set(workdirs)) == 4
    assert all(workdir.parent == tmp_path for workdir in workdirs)
    assert all(not workdir.exists() for workdir in workdirs)


@pytest.mark.parametrize("keep_after_crash", [True, False])
def test_ase_workdir_failure_cleanup(
    tmp_path: Path,
    keep_after_crash: bool,
):
    def fail(
        _parameters: dict[str, float],
        _atoms: Atoms,
        ctx: EvaluateContext,
    ) -> None:
        (ctx.temp.workdir / "diagnostic.txt").write_text("failure", encoding="utf-8")
        msg = "evaluation failed"
        raise ValueError(msg)

    computer = ASEComputer(
        atoms_factory=LJAtomsFactory(1.0),
        calculator_factory=construct_lj,
        evaluator=fail,
        evaluation_working_directory=EvaluationWorkingDirectory(
            tmp_path,
            keep_temp_workdir_after_crash=keep_after_crash,
        ),
    )
    ctx = EvaluateContext()

    with pytest.raises(ValueError, match="evaluation failed"):
        computer({"epsilon": 1.0, "sigma": 1.0}, ctx)

    assert ctx.temp.workdir.exists() is keep_after_crash
    if keep_after_crash:
        assert (ctx.temp.workdir / "diagnostic.txt").read_text() == "failure"


def test_minimize_runs_optimizer_before_quantity_extraction():
    optimizer_calls: list[tuple[float, int]] = []

    class RecordingBFGS:
        def __init__(self, atoms: Atoms, *, logfile: None) -> None:
            assert logfile is None
            self.atoms = atoms

        def run(self, *, fmax: float, steps: int) -> bool:
            optimizer_calls.append((fmax, steps))
            self.atoms.positions[1, 0] = 2.5
            return True

    def process(
        _calc: Calculator,
        atoms: Atoms,
        _ctx: EvaluateContext,
    ) -> dict[str, float]:
        return {"distance": atoms.get_distance(0, 1)}

    base = ASEComputer[dict[str, float], dict[str, float]](
        atoms_factory=LJAtomsFactory(1.0),
        calculator_factory=construct_lj,
        quantity_processors=[process],
    )
    minimized = base.minimize(
        fmax=0.02,
        max_steps=17,
        optimizer=RecordingBFGS,  # type: ignore[arg-type]
    )
    ctx = EvaluateContext()

    assert minimized({"epsilon": 1.0, "sigma": 1.0}, ctx) == {"distance": 2.5}
    assert optimizer_calls == [(0.02, 17)]
    assert ctx.meta["final_fmax"] == pytest.approx(
        np.linalg.norm(ctx.temp.atoms.get_forces(), axis=1).max()
    )
    assert isinstance(minimized, ASEComputer)
    assert base.evaluator is not minimized.evaluator


def test_minimize_accepts_an_optimizer_and_constructor_settings(tmp_path: Path):
    optimizer_calls: list[tuple[Atoms, Path, Path, Path, float]] = []

    class RecordingOptimizer:
        def __init__(
            self,
            atoms: Atoms,
            *,
            logfile: str,
            trajectory: str,
            restart: str,
            maxstep: float,
        ) -> None:
            logfile_path = Path(logfile)
            trajectory_path = Path(trajectory)
            restart_path = Path(restart)
            assert logfile_path.parent.is_dir()
            assert trajectory_path.parent.is_dir()
            assert restart_path.parent.is_dir()
            optimizer_calls.append(
                (atoms, logfile_path, trajectory_path, restart_path, maxstep)
            )

        def run(self, *, fmax: float, steps: int) -> bool:
            assert (fmax, steps) == (0.02, 17)
            return True

    computer = ASEComputer(
        atoms_factory=LJAtomsFactory(1.0),
        calculator_factory=construct_lj,
        evaluation_working_directory=EvaluationWorkingDirectory(tmp_path),
    ).minimize(
        fmax=0.02,
        max_steps=17,
        optimizer=RecordingOptimizer,  # type: ignore[arg-type]
        optimizer_kwargs={
            "logfile": "logs/optimization.log",
            "trajectory": "trajectory/optimization.traj",
            "restart": "restart/optimization.json",
            "maxstep": 0.1,
        },
    )

    computer({"epsilon": 1.0, "sigma": 1.0})

    assert len(optimizer_calls) == 1
    _atoms, logfile, trajectory, restart, maxstep = optimizer_calls[0]
    assert logfile.name == "optimization.log"
    assert trajectory.name == "optimization.traj"
    assert restart.name == "optimization.json"
    assert logfile.parents[1] == trajectory.parents[1] == restart.parents[1]
    assert not logfile.parents[1].exists()
    assert maxstep == 0.1


def test_minimize_preserves_non_filename_optimizer_settings():
    received: list[tuple[object, object]] = []

    class RecordingOptimizer:
        def __init__(
            self,
            _atoms: Atoms,
            *,
            logfile: object,
            trajectory: object,
        ) -> None:
            received.append((logfile, trajectory))

        def run(self, *, fmax: float, steps: int) -> bool:
            assert fmax > 0
            assert steps > 0
            return True

    logfile = io.StringIO()
    trajectory = object()
    computer = ASEComputer(
        atoms_factory=LJAtomsFactory(1.0),
        calculator_factory=construct_lj,
    ).minimize(
        optimizer=RecordingOptimizer,  # type: ignore[arg-type]
        optimizer_kwargs={"logfile": logfile, "trajectory": trajectory},
    )

    computer({"epsilon": 1.0, "sigma": 1.0})

    assert received == [(logfile, trajectory)]


def test_minimize_writes_real_ase_optimizer_files(tmp_path: Path):
    computer = ASEComputer(
        atoms_factory=LJAtomsFactory(1.0),
        calculator_factory=construct_lj,
        evaluation_working_directory=EvaluationWorkingDirectory(
            tmp_path,
            delete_temp_workdirs=False,
        ),
    ).minimize(
        fmax=100.0,
        max_steps=1,
        optimizer_kwargs={
            "logfile": "logs/optimization.log",
            "trajectory": "trajectory/optimization.traj",
        },
    )
    ctx = EvaluateContext()

    computer({"epsilon": 1.0, "sigma": 1.0}, ctx)

    assert (ctx.temp.workdir / "logs/optimization.log").is_file()
    assert (ctx.temp.workdir / "trajectory/optimization.traj").is_file()


@pytest.mark.parametrize("setting", ["logfile", "trajectory", "restart"])
def test_minimize_requires_a_workdir_for_optimizer_files(setting: str):
    computer = ASEComputer(
        atoms_factory=LJAtomsFactory(1.0),
        calculator_factory=construct_lj,
    )

    with pytest.raises(ValueError, match=setting):
        computer.minimize(optimizer_kwargs={setting: "shared.out"})


@pytest.mark.parametrize("path", ["/outside.log", "../outside.log"])
def test_minimize_rejects_optimizer_files_outside_workdir(
    tmp_path: Path,
    path: str,
):
    computer = ASEComputer(
        atoms_factory=LJAtomsFactory(1.0),
        calculator_factory=construct_lj,
        evaluation_working_directory=EvaluationWorkingDirectory(tmp_path),
    )

    with pytest.raises(ValueError, match="relative|safe"):
        computer.minimize(optimizer_kwargs={"logfile": path})


def test_minimize_rejects_unconverged_structure_before_processing():
    processor_called = False

    class NonConvergingBFGS:
        def __init__(self, _atoms: Atoms, *, logfile: None) -> None:
            assert logfile is None

        def run(self, *, fmax: float, steps: int) -> bool:
            assert (fmax, steps) == (0.02, 17)
            return False

    def process(
        _calc: Calculator,
        _atoms: Atoms,
        _ctx: EvaluateContext,
    ) -> dict[str, float]:
        nonlocal processor_called
        processor_called = True
        return {"energy": 0.0}

    computer = ASEComputer[dict[str, float], dict[str, float]](
        atoms_factory=LJAtomsFactory(1.0),
        calculator_factory=construct_lj,
        quantity_processors=[process],
    ).minimize(
        fmax=0.02,
        max_steps=17,
        optimizer=NonConvergingBFGS,  # type: ignore[arg-type]
    )
    ctx = EvaluateContext()

    with pytest.raises(RuntimeError, match="ASE NonConvergingBFGS did not converge"):
        computer({"epsilon": 1.0, "sigma": 1.0}, ctx)

    assert not processor_called
    assert ctx.meta["final_fmax"] == pytest.approx(
        np.linalg.norm(ctx.temp.atoms.get_forces(), axis=1).max()
    )


def test_minimize_can_explicitly_allow_unconverged_structure(
    caplog: pytest.LogCaptureFixture,
):
    class NonConvergingBFGS:
        def __init__(self, atoms: Atoms, *, logfile: None) -> None:
            assert logfile is None
            self.atoms = atoms

        def run(self, *, fmax: float, steps: int) -> bool:
            assert (fmax, steps) == (0.02, 17)
            self.atoms.positions[1, 0] = 2.5
            return False

    computer = ASEComputer[dict[str, float], dict[str, Any]](
        atoms_factory=LJAtomsFactory(1.0),
        calculator_factory=construct_lj,
        quantity_processors=[
            lambda _calc, atoms, _ctx: {"distance": atoms.get_distance(0, 1)}
        ],
    ).minimize(
        fmax=0.02,
        max_steps=17,
        optimizer=NonConvergingBFGS,  # type: ignore[arg-type]
        allow_unconverged=True,
    )

    ctx = EvaluateContext()
    with caplog.at_level("WARNING", logger="chemfit.ase_objective_function"):
        result = computer({"epsilon": 1.0, "sigma": 1.0}, ctx)

    assert result == {"distance": 2.5}
    assert not [
        record
        for record in caplog.records
        if record.name == "chemfit.ase_objective_function"
    ]
    assert ctx.meta["final_fmax"] == pytest.approx(
        np.linalg.norm(ctx.temp.atoms.get_forces(), axis=1).max()
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fmax": 0.0}, "fmax"),
        ({"max_steps": 0}, "max_steps"),
    ],
)
def test_minimize_rejects_nonpositive_limits(
    kwargs: dict[str, float | int],
    message: str,
):
    computer = ASEComputer(
        atoms_factory=LJAtomsFactory(1.0),
        calculator_factory=construct_lj,
    )

    with pytest.raises(ValueError, match=message):
        computer.minimize(**kwargs)
