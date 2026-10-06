from __future__ import annotations

import math
from concurrent.futures import Executor, ThreadPoolExecutor
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

try:
    import loky
except ImportError:
    loky = None

from chemfit import combined_objective_function
from chemfit.abstract_objective_function import EvaluateContext
from chemfit.executor_scheduler import ExecutorTreeScheduler
from chemfit.scheduling import EvaluationRequest
from chemfit.wrap_funcs import quantity

if TYPE_CHECKING:
    from collections.abc import Callable

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
            make_weights(n_terms),
            make_expected_child_losses(params, n_terms),
            strict=False,
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
    sum,
    std_reducer,
]


def test_constructor_rejects_empty_objective_list() -> None:
    with pytest.raises(ValueError, match="zero terms"):
        combined_objective_function.CombinedObjectiveFunction([])


def test_constructor_rejects_invalid_weights():
    funcs = make_funcs(2)

    with pytest.raises(ValueError, match=r"expected 2, got 1"):
        combined_objective_function.CombinedObjectiveFunction(funcs, weights=[1.0])

    with pytest.raises(ValueError, match="must be non-negative"):
        combined_objective_function.CombinedObjectiveFunction(
            funcs,
            weights=[1.0, -1.0],
        )


def test_fluent_weights_copy_configuration() -> None:
    def term_a(_params: dict[str, float]) -> float:
        return 1.0

    def term_b(_params: dict[str, float]) -> float:
        return 3.0

    def hook_a(ctx: EvaluateContext) -> None:
        del ctx

    def hook_b(ctx: EvaluateContext) -> None:
        del ctx

    source = combined_objective_function.CombinedObjectiveFunction(
        [term_a, term_b]
    ).with_meta(dataset="source")
    source.register_eval_hook(pre=hook_a, post=hook_a)
    supplied_weights = [1.0, 0.2]
    variant = source.with_weights(supplied_weights)
    supplied_weights[0] = 9.0
    variant.register_eval_hook(pre=hook_b, post=hook_b)

    assert variant is not source
    assert all(
        variant_term is source_term
        for variant_term, source_term in zip(
            variant.child_objectives(),
            source.child_objectives(),
            strict=True,
        )
    )
    assert source.pre_eval_hooks == [hook_a]
    assert variant.pre_eval_hooks == [hook_a, hook_b]
    assert source.post_eval_hooks == [hook_a]
    assert variant.post_eval_hooks == [hook_a, hook_b]
    assert math.isclose(source({}, EvaluateContext()), 4.0)
    assert math.isclose(variant({}, EvaluateContext()), 1.6)

    retagged = variant.with_meta(dataset="variant")
    assert retagged.static_meta_data == {"dataset": "variant"}
    assert variant.static_meta_data == {"dataset": "source"}
    assert math.isclose(retagged({}, EvaluateContext()), 1.6)


@pytest.mark.parametrize("weights", [[1.0], [1.0, -0.2]])
def test_with_weights_rejects_invalid_values(weights: list[float]) -> None:
    source = combined_objective_function.CombinedObjectiveFunction(make_funcs(2))

    with pytest.raises(ValueError, match="weights"):
        source.with_weights(weights)

    assert math.isclose(
        source(PARAMS, EvaluateContext()),
        sum(make_expected_child_losses(n_terms=2)),
    )


def test_fluent_reduction_and_aggregator_use_last_configuration() -> None:
    source = combined_objective_function.CombinedObjectiveFunction(
        [lambda _params: 1.0, lambda _params: 3.0]
    )

    def aggregator(
        terms: list[float],
        _quantities: list[dict[str, Any] | None],
        ctx: EvaluateContext,
    ) -> float:
        ctx.meta["aggregated"] = True
        return max(terms) + 10.0

    reduced = source.with_reduction(combined_objective_function.mean_reducer)
    aggregated = source.with_aggregator(aggregator)
    aggregator_last = reduced.with_aggregator(aggregator)
    reducer_last = aggregated.with_reduction(combined_objective_function.sum_reducer)

    assert reduced is not source
    assert aggregated is not source
    assert aggregator_last is not reduced
    assert reducer_last is not aggregated
    assert isinstance(source.reduction, combined_objective_function.WrappedReducer)
    assert source.reduction.to_reducer() is combined_objective_function.sum_reducer
    assert isinstance(reduced.reduction, combined_objective_function.WrappedReducer)
    assert reduced.reduction.to_reducer() is combined_objective_function.mean_reducer
    assert aggregated.reduction is aggregator
    assert aggregator_last.reduction is aggregator
    assert isinstance(
        reducer_last.reduction, combined_objective_function.WrappedReducer
    )
    assert (
        reducer_last.reduction.to_reducer() is combined_objective_function.sum_reducer
    )
    assert math.isclose(source({}, EvaluateContext()), 4.0)
    assert math.isclose(reduced({}, EvaluateContext()), 2.0)
    aggregated_ctx = EvaluateContext()
    assert math.isclose(aggregated({}, aggregated_ctx), 13.0)
    assert aggregated_ctx.meta["aggregated"] is True
    assert math.isclose(aggregator_last({}, EvaluateContext()), 13.0)
    assert math.isclose(reducer_last({}, EvaluateContext()), 4.0)


def test_fluent_exception_handler_uses_last_configuration() -> None:
    def ok(_params: dict[str, float]) -> float:
        return 2.0

    def broken(_params: dict[str, float]) -> float:
        msg = "broken term"
        raise RuntimeError(msg)

    source = combined_objective_function.CombinedObjectiveFunction([ok, broken])
    skipped = source.with_exception_handler(
        combined_objective_function.skip_exception_handler
    )
    replaced = skipped.with_exception_handler(
        combined_objective_function.nan_exception_handler
    )

    assert skipped is not source
    assert replaced is not skipped
    assert (
        source.exception_handler
        is combined_objective_function.raising_exception_handler
    )
    assert (
        skipped.exception_handler is combined_objective_function.skip_exception_handler
    )
    assert (
        replaced.exception_handler is combined_objective_function.nan_exception_handler
    )
    with pytest.raises(RuntimeError, match="broken term"):
        source({}, EvaluateContext())
    assert math.isclose(skipped({}, EvaluateContext()), 2.0)
    assert math.isnan(replaced({}, EvaluateContext()))


def test_terms_and_weights_are_not_public_attributes() -> None:
    cob = combined_objective_function.CombinedObjectiveFunction(make_funcs(1))

    assert not hasattr(cob, "add")
    assert not hasattr(cob, "objective_functions")
    assert not hasattr(cob, "weights")
    assert cob.n_terms() == 1
    assert len(cob.child_objectives()) == 1


EXECUTORS: list[Executor] = [ThreadPoolExecutor(2)]

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
    assert ctx.loss is not None
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
    reduction: combined_objective_function.Reducer, executor: Executor
):
    cob = make_cob(reduction=reduction)
    with ExecutorTreeScheduler(executor=executor).prepare(cob) as schedule:
        ctx = EvaluateContext()
        res = schedule.evaluate(PARAMS, ctx)

    standard_asserts(res, ctx, reduction)


@pytest.mark.parametrize(("reduction", "executor"), list(product(REDUCERS, EXECUTORS)))
def test_combined_objective_uses_executor_scheduler(
    reduction: combined_objective_function.Reducer, executor: Executor
):
    scheduler = ExecutorTreeScheduler(executor=executor)

    cob = combined_objective_function.CombinedObjectiveFunction(
        make_funcs(),
        make_weights(),
        reduction=reduction,
        child_context_configurator=context_configurator,
    )

    with scheduler.prepare(cob) as schedule:
        ctx = EvaluateContext()
        res = schedule.evaluate(PARAMS, ctx)

    standard_asserts(res, ctx, reduction)


@pytest.mark.parametrize("reduction", REDUCERS)
def test_combined_objective_reduces_terms_with_mpi(
    reduction: combined_objective_function.Reducer,
):
    mpi_scheduler = pytest.importorskip(
        "chemfit.mpi_scheduler", reason="Missing mpi4py"
    )

    cob = make_cob(reduction=reduction)
    scheduler = mpi_scheduler.MPITreeScheduler(mpi_debug_log=False)

    with scheduler.prepare(cob) as mpi:
        if mpi.rank == 0:
            ctx = EvaluateContext()
            res = mpi.evaluate(PARAMS, ctx)

            standard_asserts(res, ctx, reduction)

        else:
            mpi.worker_loop()


@pytest.mark.parametrize("reduction", REDUCERS)
def test_combined_objective_uses_mpi_scheduler(
    reduction: combined_objective_function.Reducer,
):
    mpi_scheduler = pytest.importorskip(
        "chemfit.mpi_scheduler", reason="Missing mpi4py"
    )

    cob = combined_objective_function.CombinedObjectiveFunction(
        make_funcs(),
        make_weights(),
        reduction=reduction,
        child_context_configurator=context_configurator,
    )

    with mpi_scheduler.MPITreeScheduler(mpi_debug_log=False).prepare(cob) as mpi:
        if mpi.rank == 0:
            ctx = EvaluateContext()
            res = mpi.evaluate(PARAMS, ctx)

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
    assert ctx.loss is not None
    assert math.isnan(ctx.loss)

    ob.exception_handler = combined_objective_function.skip_exception_handler
    ctx = EvaluateContext()
    res = ob(PARAMS, ctx)
    assert math.isclose(res, func1(PARAMS))
    assert ctx.loss is not None
    assert math.isclose(ctx.loss, func1(PARAMS))
    assert ctx.meta["skipped_indices"] == [1]


@pytest.mark.parametrize("executor", EXECUTORS)
def test_combined_objective_exception_handlers_with_executor(executor: Executor):
    def func1(params: dict) -> float:  # noqa: ARG001
        return 1.0

    def whoops(params: dict) -> float:  # noqa: ARG001
        msg = "Whoops"
        raise RuntimeError(msg)

    ob = combined_objective_function.CombinedObjectiveFunction([func1, whoops])
    with ExecutorTreeScheduler(executor=executor).prepare(ob) as schedule:
        ob.exception_handler = combined_objective_function.raising_exception_handler
        with pytest.raises(RuntimeError, match="Whoops"):
            schedule.evaluate(PARAMS, EvaluateContext())

        ob.exception_handler = combined_objective_function.nan_exception_handler
        ctx = EvaluateContext()
        res = schedule.evaluate(PARAMS, ctx)
        assert math.isnan(res)
        assert ctx.loss is not None
        assert math.isnan(ctx.loss)

        ob.exception_handler = combined_objective_function.skip_exception_handler
        ctx = EvaluateContext()
        res = schedule.evaluate(PARAMS, ctx)

        assert math.isclose(res, func1(PARAMS))
        assert ctx.loss is not None
        assert math.isclose(ctx.loss, func1(PARAMS))
        assert ctx.meta["skipped_indices"] == [1]


def test_combined_objective_exception_handlers_with_mpi():
    mpi_scheduler = pytest.importorskip(
        "chemfit.mpi_scheduler", reason="Missing mpi4py"
    )

    def func1(params: dict) -> float:  # noqa: ARG001
        return 1.0

    def whoops(params: dict) -> float:  # noqa: ARG001
        msg = "Whoops"
        raise RuntimeError(msg)

    # raising
    ob = combined_objective_function.CombinedObjectiveFunction([func1, whoops])
    ob.exception_handler = combined_objective_function.raising_exception_handler

    with mpi_scheduler.MPITreeScheduler(mpi_debug_log=False).prepare(ob) as mpi:
        if mpi.rank == 0:
            with pytest.raises(RuntimeError, match="Whoops"):
                mpi.evaluate(PARAMS, EvaluateContext())
        else:
            mpi.worker_loop()

    # nan
    ob = combined_objective_function.CombinedObjectiveFunction([func1, whoops])
    ob.exception_handler = combined_objective_function.nan_exception_handler

    with mpi_scheduler.MPITreeScheduler(mpi_debug_log=False).prepare(ob) as mpi:
        if mpi.rank == 0:
            ctx = EvaluateContext()
            res = mpi.evaluate(PARAMS, ctx)
            assert math.isnan(res)
            assert ctx.loss is not None
            assert math.isnan(ctx.loss)
        else:
            mpi.worker_loop()

    # skip
    ob = combined_objective_function.CombinedObjectiveFunction([func1, whoops])
    ob.exception_handler = combined_objective_function.skip_exception_handler

    with mpi_scheduler.MPITreeScheduler(mpi_debug_log=False).prepare(ob) as mpi:
        if mpi.rank == 0:
            ctx = EvaluateContext()
            res = mpi.evaluate(PARAMS, ctx)
            assert math.isclose(res, func1(PARAMS))
            assert ctx.loss is not None
            assert math.isclose(ctx.loss, func1(PARAMS))
            assert ctx.meta["skipped_indices"] == [1]
        else:
            mpi.worker_loop()


@pytest.mark.parametrize("executor", EXECUTORS)
def test_aggregator(executor: Executor):
    def custom_aggregator(
        terms: list[float],  # noqa: ARG001
        quantities: list[dict[str, Any] | None],
        ctx: EvaluateContext,
    ) -> float:
        ctx.meta["foo"] = "bar"
        assert all(q is not None for q in quantities)
        return sum(q["test"] for q in quantities if q is not None)

    @quantity()
    def q1(parameters: dict[str, float], f: float) -> dict[str, float]:
        return {"test": f * parameters["x"] + parameters["y"]}

    cob = combined_objective_function.CombinedObjectiveFunction(
        [
            q1.bind(f=1).with_loss(lambda _: 0.0),
            q1.bind(f=2).with_loss(lambda _: 0.0),
        ],
        aggregator=custom_aggregator,
    )

    with ExecutorTreeScheduler(executor=executor).prepare(cob) as schedule:
        ctx = EvaluateContext()
        res = schedule.evaluate(PARAMS, ctx)

    assert math.isclose(res, 8.0)
    assert ctx.meta["foo"] == "bar"


def test_reducer_and_aggregator_are_mutually_exclusive():
    def custom_aggregator(
        terms: list[float],
        _quantities: list[dict[str, Any] | None],
        _ctx: EvaluateContext,
    ) -> float:
        return sum(terms)

    with pytest.raises(ValueError, match="either `reduction` or `aggregator`"):
        combined_objective_function.CombinedObjectiveFunction(
            make_funcs(),
            reduction=sum,
            aggregator=custom_aggregator,
        )


@pytest.mark.parametrize("executor", EXECUTORS)
def test_executor_scheduler_matches_serial_result(executor: Executor):
    cob = make_cob()

    ctx_serial = EvaluateContext()
    res_serial = cob(PARAMS, ctx_serial)

    with ExecutorTreeScheduler(executor=executor).prepare(cob) as schedule:
        ctx_exec = EvaluateContext()
        res_exec = schedule.evaluate(PARAMS, ctx_exec)

    assert np.isclose(res_exec, res_serial)
    assert ctx_exec.loss is not None
    assert ctx_serial.loss is not None
    assert np.isclose(ctx_exec.loss, ctx_serial.loss)

    assert "children" in ctx_serial.meta
    assert "children" in ctx_exec.meta
    assert len(ctx_serial.meta["children"]) == len(ctx_exec.meta["children"])

    serial_child_losses = [child["loss"] for child in ctx_serial.meta["children"]]
    exec_child_losses = [child["loss"] for child in ctx_exec.meta["children"]]
    assert np.allclose(serial_child_losses, exec_child_losses)


N_EVALS = 4


@pytest.mark.parametrize("executor", EXECUTORS)
def test_executor_scheduler_evaluates_batch(executor: Executor):
    serial_cob = make_cob()
    cob = make_cob()
    schedule = ExecutorTreeScheduler(executor=executor).prepare(cob)

    params_list = [
        {"x": float(i) / N_EVALS, "y": float(N_EVALS - i) / N_EVALS}
        for i in range(N_EVALS)
    ]
    results_expected = [serial_cob(p) for p in params_list]

    ctxs = [EvaluateContext() for _ in range(N_EVALS)]
    requests = [
        EvaluationRequest(parameters=params, ctx=ctx)
        for params, ctx in zip(params_list, ctxs, strict=True)
    ]
    with schedule:
        completed = sorted(
            schedule.evaluate_many(requests), key=lambda result: result.index
        )
    results = []
    for result in completed:
        assert result.success
        assert isinstance(result.value, float)
        results.append(result.value)

    assert results == results_expected

    for res, ctx, params in zip(results, ctxs, params_list, strict=False):
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
