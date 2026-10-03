import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from conftest import LJAtomsFactory, construct_lj, e_lj

import chemfit.ase_objective_function as ase_module
from chemfit.abstract_objective_function import EvaluateContext
from chemfit.ase_objective_function import ASEComputer
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.fitter import Fitter


def loss_function(quants: dict[str, Any], e_ref: float) -> float:
    return (quants["energy"] - e_ref) ** 2


def lj_ob_term(r: float, eps: float, sigma: float):
    return ASEComputer(
        atoms_factory=LJAtomsFactory(r),
        calculator_factory=construct_lj,
        tag="lj_{r}",
    ).with_loss(loss_function, e_ref=e_lj(r, eps, sigma))


def get_ob_func(eps: float, sigma: float):
    r_min = 2.0 ** (1 / 6) * sigma
    r_list = np.linspace(0.925 * r_min, 3.0 * sigma)

    return CombinedObjectiveFunction(
        objective_functions=[lj_ob_term(r, eps, sigma) for r in r_list]
    )


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
    terms_meta_data = ctx.to_meta_data()["meta"]["children"]

    assert ob.n_terms() == len(terms_meta_data)
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
    processor_copy(parameters)
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


def test_minimize_runs_bfgs_before_quantity_extraction(
    monkeypatch: pytest.MonkeyPatch,
):
    optimizer_calls: list[tuple[float, int]] = []

    class RecordingBFGS:
        def __init__(self, atoms: Atoms, *, logfile: None) -> None:
            assert logfile is None
            self.atoms = atoms

        def run(self, *, fmax: float, steps: int) -> None:
            optimizer_calls.append((fmax, steps))
            self.atoms.positions[1, 0] = 2.5

    def process(
        _calc: Calculator,
        atoms: Atoms,
        _ctx: EvaluateContext,
    ) -> dict[str, float]:
        return {"distance": atoms.get_distance(0, 1)}

    monkeypatch.setattr(ase_module, "BFGS", RecordingBFGS)
    base = ASEComputer[dict[str, float], dict[str, float]](
        atoms_factory=LJAtomsFactory(1.0),
        calculator_factory=construct_lj,
        quantity_processors=[process],
    )
    minimized = base.minimize(fmax=0.02, max_steps=17)

    assert minimized({"epsilon": 1.0, "sigma": 1.0}) == {"distance": 2.5}
    assert optimizer_calls == [(0.02, 17)]
    assert isinstance(minimized, ASEComputer)
    assert base.evaluator is not minimized.evaluator


def test_lj_mpi():
    mpi_scheduler = pytest.importorskip(
        "chemfit.mpi_scheduler", reason="Missing mpi4py"
    )

    # Construct the objective function on *all* ranks
    eps = 1.0
    sigma = 1.0

    ob = get_ob_func(eps, sigma)

    initial_params = {"epsilon": 2.0, "sigma": 1.5}

    with ob.set_scheduler(mpi_scheduler.MPITreeScheduler()) as mpi:
        if mpi.rank == 0:
            fitter = Fitter(ob, initial_params=initial_params)
            opt_params = fitter.fit_scipy()

            ctx = EvaluateContext()
            ob(opt_params, ctx)
            terms_meta_data = ctx.to_meta_data()["meta"]["children"]

            assert ob.n_terms() == len(terms_meta_data)
            assert np.isclose(opt_params["epsilon"], eps)
            assert np.isclose(opt_params["sigma"], sigma)
        else:
            mpi.worker_loop()


if __name__ == "__main__":
    import logging

    logging.basicConfig(filename="test_lj.log")

    # test_lj()
    test_lj_mpi()
