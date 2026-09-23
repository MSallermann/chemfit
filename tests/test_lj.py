import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
from conftest import LJAtomsFactory, apply_params_lj, construct_lj, e_lj

from chemfit.abstract_objective_function import EvaluateContext
from chemfit.ase_objective_function import SinglePointASEComputer
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.fitter import Fitter


def loss_function(quants: dict[str, Any], e_ref: float) -> float:
    return (quants["energy"] - e_ref) ** 2


def lj_ob_term(r: float, eps: float, sigma: float):
    return SinglePointASEComputer(
        calc_factory=construct_lj,
        param_applier=apply_params_lj,
        atoms_factory=LJAtomsFactory(r),
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
    computer = SinglePointASEComputer(
        calc_factory=construct_lj,
        param_applier=apply_params_lj,
        atoms_factory=atoms_factory,
    )
    parameters = {"epsilon": 1.0, "sigma": 1.0}

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(computer, [parameters] * 8))

    assert atoms_factory.calls == 1
    assert all(np.isclose(result["energy"], results[0]["energy"]) for result in results)


def test_single_point_computer_remains_pickleable():
    computer = SinglePointASEComputer(
        calc_factory=construct_lj,
        param_applier=apply_params_lj,
        atoms_factory=LJAtomsFactory(1.0),
    )

    restored = pickle.loads(pickle.dumps(computer))  # noqa: S301

    quantities = restored({"epsilon": 1.0, "sigma": 1.0})
    assert "energy" in quantities


def test_lj_mpi():
    mpi_wrapper_cob = pytest.importorskip(
        "chemfit.mpi_wrapper_cob", reason="Missing mpi4py"
    )

    # Construct the objective function on *all* ranks
    eps = 1.0
    sigma = 1.0

    ob = get_ob_func(eps, sigma)

    initial_params = {"epsilon": 2.0, "sigma": 1.5}

    # Use the MPI Wrapper to make the combined objective function "MPI aware"
    with mpi_wrapper_cob.MPIWrapperCOB(ob) as mpi:
        if mpi.rank == 0:
            fitter = Fitter(mpi, initial_params=initial_params)
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
