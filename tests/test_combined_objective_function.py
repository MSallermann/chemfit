import math
import random
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
from chemfit.abstract_objective_function import (
    EvaluateContext,
    ExecutorLike,
)
from chemfit.executor_utils import map_with_context
from chemfit.executor_wrapper_cob import ExecutorWrapperCOB
from chemfit.wrap_funcs import to_quantity_computer

N_TERMS = 10

PARAMS = {"x": 2.0, "y": 1.0}


def make_funcs(n_terms: int = N_TERMS) -> list[Callable[[dict], float]]:
    return [lambda p, i=i: p["x"] ** 2 - i * p["y"] for i in range(n_terms)]


def make_weights(n_terms: int = N_TERMS) -> list[float]:
    return list(range(n_terms))


def make_expected_child_losses(
    params: dict[str, float] = PARAMS, n_terms: int = N_TERMS
) -> list[float]:
    return [f(params) for f in make_funcs(n_terms)]


def make_expected_terms(
    params: dict[str, float] = PARAMS, n_terms: int = N_TERMS
) -> list[float]:
    return [
        w * f
        for w, f in zip(
            make_weights(n_terms), make_expected_child_losses(params, n_terms)
        )
    ]


def context_configurator(
    idx_child_ctx: int,
    child_ctx: EvaluateContext,
    num_children: int,
    parent_ctx: EvaluateContext,  # noqa: ARG001
):
    child_ctx.meta["configurator_number"] = idx_child_ctx + num_children


def make_expected_configurator_numbers(n_terms: int = N_TERMS):
    return [i + n_terms for i in range(n_terms)]


def make_cob(
    reduction: combined_objective_function.Reducer = combined_objective_function.sum_reducer,
) -> combined_objective_function.CombinedObjectiveFunction:
    return combined_objective_function.CombinedObjectiveFunction(
        make_funcs(),
        make_weights(),
        reduction=reduction,
        child_context_configurator=context_configurator,
    )


def std_reducer(terms: list[float]) -> float:
    return float(np.std(terms))


REDUCERS = [
    combined_objective_function.sum_reducer,
    std_reducer,
]


EXECUTORS: list[ExecutorLike] = [ThreadPoolExecutor(2)]

if loky is not None:
    EXECUTORS.append(loky.ProcessPoolExecutor(2))


def standard_asserts(
    res: float,
    ctx: EvaluateContext,
    reduction: combined_objective_function.Reducer,
    params: dict[str, float] = PARAMS,
    n_terms: int = N_TERMS,
):
    assert ctx.parameters == params
    assert np.isclose(ctx.loss, res)

    assert "children" in ctx.meta
    assert len(ctx.meta["children"]) == n_terms

    child_losses = [child["loss"] for child in ctx.meta["children"]]
    assert np.allclose(child_losses, make_expected_child_losses(params, n_terms))
    assert np.isclose(res, reduction(make_expected_terms(params, n_terms)))

    configured_numbers = [
        child["meta"]["configurator_number"] for child in ctx.meta["children"]
    ]
    assert all(
        np.isclose(configured_numbers, make_expected_configurator_numbers(n_terms))
    )


@pytest.mark.parametrize("reduction", REDUCERS)
def test_combined_objective_reduces_terms_serially(
    reduction: combined_objective_function.Reducer,
):
    cob = make_cob(reduction=reduction)

    ctx = EvaluateContext()
    res = cob(PARAMS, ctx)

    standard_asserts(res, ctx, reduction)


@pytest.mark.parametrize(("reduction", "executor"), list(product(REDUCERS, EXECUTORS)))
def test_combined_objective_reduces_terms_with_executor(
    reduction: combined_objective_function.Reducer, executor: ExecutorLike
):
    cob = make_cob(reduction=reduction)
    wrapped = ExecutorWrapperCOB(cob, executor=executor)

    ctx = EvaluateContext()
    res = wrapped(PARAMS, ctx)

    standard_asserts(res, ctx, reduction)


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
            res = mpi(PARAMS, ctx)

            standard_asserts(res, ctx, reduction)

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
        ob(PARAMS)

    ob.exception_handler = combined_objective_function.nan_exception_handler
    ctx = EvaluateContext()
    res = ob(PARAMS, ctx)
    assert math.isnan(res)
    assert math.isnan(ctx.loss)

    ob.exception_handler = combined_objective_function.skip_exception_handler
    ctx = EvaluateContext()
    res = ob(PARAMS, ctx)
    assert math.isclose(res, func1(PARAMS))
    assert math.isclose(ctx.loss, func1(PARAMS))
    assert ctx.meta["skipped_indices"] == [1]


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
        wrapped(PARAMS)

    ob.exception_handler = combined_objective_function.nan_exception_handler
    ctx = EvaluateContext()
    res = wrapped(PARAMS, ctx)
    assert math.isnan(res)
    assert math.isnan(ctx.loss)

    ob.exception_handler = combined_objective_function.skip_exception_handler
    ctx = EvaluateContext()
    res = wrapped(PARAMS, ctx)

    assert math.isclose(res, func1(PARAMS))
    assert math.isclose(ctx.loss, func1(PARAMS))
    assert ctx.meta["skipped_indices"] == [1]


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
                mpi(PARAMS)
        else:
            mpi.worker_loop()

    # nan
    ob = combined_objective_function.CombinedObjectiveFunction([func1, whoops])
    ob.exception_handler = combined_objective_function.nan_exception_handler

    with mpi_wrapper_cob.MPIWrapperCOB(ob, mpi_debug_log=False) as mpi:
        if mpi.rank == 0:
            ctx = EvaluateContext()
            res = mpi(PARAMS, ctx)
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
            res = mpi(PARAMS, ctx)
            assert math.isclose(res, func1(PARAMS))
            assert math.isclose(ctx.loss, func1(PARAMS))
            assert ctx.meta["skipped_indices"] == [1]

        else:
            mpi.worker_loop()


@pytest.mark.parametrize("executor", EXECUTORS)
def test_aggregator(executor: ExecutorLike):
    def custom_aggregator(
        terms: list[float],  # noqa: ARG001
        quantities: list[dict],
        ctx: EvaluateContext,
    ) -> float:
        ctx.meta["foo"] = "bar"
        return sum(q["test"] for q in quantities)

    @to_quantity_computer()
    def q1(parameters: dict[str, float], f: float) -> dict[str, float]:
        return {"test": f * parameters["x"] + parameters["y"]}

    cob = combined_objective_function.CombinedObjectiveFunction(
        [q1.bind(f=1).with_loss(lambda p: 0.0), q1.bind(f=2).with_loss(lambda p: 0.0)],  # noqa: ARG005
        reduction=custom_aggregator,
    )

    cob_wrapped = ExecutorWrapperCOB(cob, executor)

    ctx = EvaluateContext()
    res = cob_wrapped(PARAMS, ctx)

    assert math.isclose(res, 8.0)
    assert ctx.meta["foo"] == "bar"


@pytest.mark.parametrize("executor", EXECUTORS)
def test_executor_wrapper_matches_serial_result(executor: ExecutorLike):
    cob = make_cob()
    wrapped = ExecutorWrapperCOB(cob, executor=executor)

    ctx_serial = EvaluateContext()
    ctx_exec = EvaluateContext()

    res_serial = cob(PARAMS, ctx_serial)
    res_exec = wrapped(PARAMS, ctx_exec)

    assert np.isclose(res_exec, res_serial)
    assert np.isclose(ctx_exec.loss, ctx_serial.loss)

    assert "children" in ctx_serial.meta
    assert "children" in ctx_exec.meta
    assert len(ctx_serial.meta["children"]) == len(ctx_exec.meta["children"])

    serial_child_losses = [child["loss"] for child in ctx_serial.meta["children"]]
    exec_child_losses = [child["loss"] for child in ctx_exec.meta["children"]]
    assert np.allclose(serial_child_losses, exec_child_losses)


N_EVALS = 4
EXECUTORS_OUTER = [ThreadPoolExecutor(N_EVALS)]

if loky is not None:
    EXECUTORS_OUTER.append(loky.ProcessPoolExecutor(N_EVALS))


@pytest.mark.parametrize(
    ("executor_outer", "executor_inner"), list(product(EXECUTORS_OUTER, EXECUTORS))
)
def test_parallel_evaluation(
    executor_outer: ExecutorLike, executor_inner: ExecutorLike
):
    cob = make_cob()
    wrapped = ExecutorWrapperCOB(cob, executor=executor_inner)

    executor_outer = ThreadPoolExecutor(N_EVALS)

    params_list = [
        {"x": random.random(), "y": random.random()}  # noqa: S311
        for _ in range(N_EVALS)
    ]
    results_expected = [cob(p) for p in params_list]

    ctxs = [EvaluateContext() for _ in range(N_EVALS)]

    results = map_with_context(executor_outer, wrapped, params_list, ctxs=ctxs)

    assert results == results_expected

    for res, ctx, params in zip(results, ctxs, params_list):
        child_losses = [child["loss"] for child in ctx.meta["children"]]
        assert np.allclose(child_losses, make_expected_child_losses(params))

        assert isinstance(cob.reduction, combined_objective_function.WrappedReducer)

        standard_asserts(
            res=res,
            ctx=ctx,
            reduction=cob.reduction.to_reducer(),
            params=params,
            n_terms=cob.n_terms(),
        )
