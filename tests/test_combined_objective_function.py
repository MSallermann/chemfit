import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pytest

try:
    import loky
except ImportError:
    loky = None

from chemfit import combined_objective_function
from chemfit.abstract_objective_function import EvaluateContext, ExecutorLike
from chemfit.executor_wrapper_cob import ExecutorWrapperCOB

N_TERMS = 10


def make_funcs(n_terms: int = N_TERMS) -> list[Callable[[dict], float]]:
    return [lambda p, i=i: float(i) for i in range(n_terms)]  # noqa: ARG005


def make_weights(n_terms: int = N_TERMS) -> list[float]:
    return list(range(n_terms))


def make_expected_child_losses(n_terms: int = N_TERMS) -> list[float]:
    return [f({}) for f in make_funcs(n_terms)]


def make_expected_terms(n_terms: int = N_TERMS) -> list[float]:
    return [
        w * f
        for w, f in zip(make_weights(n_terms), make_expected_child_losses(n_terms))
    ]


def make_cob(
    reduction: combined_objective_function.Reducer = combined_objective_function.sum_reducer,
) -> combined_objective_function.CombinedObjectiveFunction:
    return combined_objective_function.CombinedObjectiveFunction(
        make_funcs(),
        make_weights(),
        reduction=reduction,
    )


def std_reducer(terms: list[float]) -> float:
    return float(np.std(terms))


REDUCERS = [
    combined_objective_function.sum_reducer,
    std_reducer,
]


EXECUTORS = [ThreadPoolExecutor(2)]

if loky is not None:
    EXECUTORS.append(loky.ProcessPoolExecutor(2))


@pytest.mark.parametrize("reduction", REDUCERS)
def test_combined_objective_reduces_terms_serially(
    reduction: combined_objective_function.Reducer,
):
    cob = make_cob(reduction=reduction)

    ctx = EvaluateContext()
    res = cob({}, ctx)

    assert ctx.parameters == {}
    assert np.isclose(ctx.loss, res)

    assert "children" in ctx.meta
    assert len(ctx.meta["children"]) == N_TERMS

    child_losses = [child["loss"] for child in ctx.meta["children"]]
    assert np.allclose(child_losses, make_expected_child_losses())

    assert np.isclose(res, reduction(make_expected_terms()))


@pytest.mark.parametrize(("reduction", "executor"), list(product(REDUCERS, EXECUTORS)))
def test_combined_objective_reduces_terms_with_executor(
    reduction: combined_objective_function.Reducer, executor: ExecutorLike
):
    cob = make_cob(reduction=reduction)
    wrapped = ExecutorWrapperCOB(cob, executor=executor)

    ctx = EvaluateContext()
    res = wrapped({}, ctx)

    assert ctx.parameters == {}
    assert np.isclose(ctx.loss, res)

    assert "children" in ctx.meta
    assert len(ctx.meta["children"]) == N_TERMS

    child_losses = [child["loss"] for child in ctx.meta["children"]]
    assert np.allclose(child_losses, make_expected_child_losses())

    assert np.isclose(res, reduction(make_expected_terms()))


@pytest.mark.parametrize("reduction", REDUCERS)
def test_combined_objective_reduces_terms_with_mpi(
    reduction: combined_objective_function.Reducer,
):
    mpi_wrapper_cob = pytest.importorskip(
        "chemfit.mpi_wrapper_cob", reason="Missing mpi4py"
    )

    cob = make_cob(reduction=reduction)

    with mpi_wrapper_cob.MPIWrapperCOB(cob, mpi_debug_log=False) as mpi:
        if mpi.rank == 0:
            ctx = EvaluateContext()
            res = mpi({}, ctx)

            assert ctx.parameters == {}
            assert np.isclose(ctx.loss, res)

            assert "children" in ctx.meta
            assert len(ctx.meta["children"]) == N_TERMS

            child_losses = [child["loss"] for child in ctx.meta["children"]]
            assert np.allclose(child_losses, make_expected_child_losses())

            assert np.isclose(res, reduction(make_expected_terms()))
        else:
            mpi.worker_loop()


def test_combined_objective_exception_handlers_serial():
    def func1(params: dict) -> float:  # noqa: ARG001
        return 1.0

    def whoops(params: dict) -> float:  # noqa: ARG001
        msg = "Whoops"
        raise RuntimeError(msg)

    ob = combined_objective_function.CombinedObjectiveFunction([func1, whoops])

    ob.exception_handler = combined_objective_function.raising_exception_handler
    with pytest.raises(RuntimeError, match="Whoops"):
        ob({})

    ob.exception_handler = combined_objective_function.nan_exception_handler
    ctx = EvaluateContext()
    res = ob({}, ctx)
    assert math.isnan(res)
    assert math.isnan(ctx.loss)

    ob.exception_handler = combined_objective_function.skip_exception_handler
    ctx = EvaluateContext()
    res = ob({}, ctx)
    assert math.isclose(res, func1({}))
    assert math.isclose(ctx.loss, func1({}))


@pytest.mark.parametrize("executor", EXECUTORS)
def test_combined_objective_exception_handlers_with_executor(executor: ExecutorLike):
    def func1(params: dict) -> float:  # noqa: ARG001
        return 1.0

    def whoops(params: dict) -> float:  # noqa: ARG001
        msg = "Whoops"
        raise RuntimeError(msg)

    ob = combined_objective_function.CombinedObjectiveFunction([func1, whoops])
    wrapped = ExecutorWrapperCOB(ob, executor=executor)

    ob.exception_handler = combined_objective_function.raising_exception_handler
    with pytest.raises(RuntimeError, match="Whoops"):
        wrapped({})

    ob.exception_handler = combined_objective_function.nan_exception_handler
    ctx = EvaluateContext()
    res = wrapped({}, ctx)
    assert math.isnan(res)
    assert math.isnan(ctx.loss)

    ob.exception_handler = combined_objective_function.skip_exception_handler
    ctx = EvaluateContext()
    res = wrapped({}, ctx)
    assert math.isclose(res, func1({}))
    assert math.isclose(ctx.loss, func1({}))


def test_combined_objective_exception_handlers_with_mpi():
    mpi_wrapper_cob = pytest.importorskip(
        "chemfit.mpi_wrapper_cob", reason="Missing mpi4py"
    )

    def func1(params: dict) -> float:  # noqa: ARG001
        return 1.0

    def whoops(params: dict) -> float:  # noqa: ARG001
        msg = "Whoops"
        raise RuntimeError(msg)

    # raising
    ob = combined_objective_function.CombinedObjectiveFunction([func1, whoops])
    ob.exception_handler = combined_objective_function.raising_exception_handler

    with mpi_wrapper_cob.MPIWrapperCOB(ob, mpi_debug_log=False) as mpi:
        if mpi.rank == 0:
            with pytest.raises(RuntimeError, match="Whoops"):
                mpi({})
        else:
            mpi.worker_loop()

    # nan
    ob = combined_objective_function.CombinedObjectiveFunction([func1, whoops])
    ob.exception_handler = combined_objective_function.nan_exception_handler

    with mpi_wrapper_cob.MPIWrapperCOB(ob, mpi_debug_log=False) as mpi:
        if mpi.rank == 0:
            ctx = EvaluateContext()
            res = mpi({}, ctx)
            assert math.isnan(res)
            assert math.isnan(ctx.loss)
        else:
            mpi.worker_loop()

    # skip
    ob = combined_objective_function.CombinedObjectiveFunction([func1, whoops])
    ob.exception_handler = combined_objective_function.skip_exception_handler

    with mpi_wrapper_cob.MPIWrapperCOB(ob, mpi_debug_log=False) as mpi:
        if mpi.rank == 0:
            ctx = EvaluateContext()
            res = mpi({}, ctx)
            assert math.isclose(res, func1({}))
            assert math.isclose(ctx.loss, func1({}))
        else:
            mpi.worker_loop()


@pytest.mark.parametrize("executor", EXECUTORS)
def test_executor_wrapper_matches_serial_result(executor: ExecutorLike):
    cob = make_cob()
    wrapped = ExecutorWrapperCOB(cob, executor=executor)

    ctx_serial = EvaluateContext()
    ctx_exec = EvaluateContext()

    res_serial = cob({}, ctx_serial)
    res_exec = wrapped({}, ctx_exec)

    assert np.isclose(res_exec, res_serial)
    assert np.isclose(ctx_exec.loss, ctx_serial.loss)

    assert "children" in ctx_serial.meta
    assert "children" in ctx_exec.meta
    assert len(ctx_serial.meta["children"]) == len(ctx_exec.meta["children"])

    serial_child_losses = [child["loss"] for child in ctx_serial.meta["children"]]
    exec_child_losses = [child["loss"] for child in ctx_exec.meta["children"]]
    assert np.allclose(serial_child_losses, exec_child_losses)
