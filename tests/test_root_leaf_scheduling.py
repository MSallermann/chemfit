"""Tests for scheduling an ordinary objective as the call-tree root."""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from chemfit.abstract_objective_function import EvaluateContext
from chemfit.executor_scheduler import ExecutorTreeScheduler
from chemfit.scheduling import EvaluationRequest
from chemfit.tree_schedule import SerialTreeScheduler
from chemfit.wrap_funcs import to_objective_functor

Parameters = dict[str, Any]


class OrdinaryLeafError(RuntimeError):
    """Ordinary per-evaluation failure raised by a root leaf."""


@to_objective_functor(pass_ctx=True)
def root_leaf(parameters: Parameters, ctx: EvaluateContext):
    """Evaluate one root-leaf request."""
    delay = float(parameters.get("delay", 0.0))
    if delay:
        time.sleep(delay)

    ctx.meta["context_identity"] = id(ctx)
    ctx.meta["label"] = parameters.get("label")
    ctx.quantities = {"value": float(parameters["value"])}

    if parameters.get("record_mpi_rank"):
        from mpi4py import MPI  # noqa: PLC0415

        ctx.meta["worker_rank"] = MPI.COMM_WORLD.Get_rank()

    if parameters.get("fail"):
        msg = f"root leaf failed for {parameters['value']}"
        raise OrdinaryLeafError(msg)

    return float(parameters["value"])


def test_serial_tree_scheduler_evaluates_many_root_leaf_runs() -> None:
    """Each request becomes an independent task for the same root node."""

    objective = root_leaf
    parameters = [
        {"value": value, "label": f"run-{idx}"}
        for idx, value in enumerate((1.0, 2.0, 3.0))
    ]
    contexts = [EvaluateContext() for _ in parameters]
    requests = [
        EvaluationRequest(parameters=params, ctx=ctx)
        for params, ctx in zip(parameters, contexts, strict=True)
    ]

    with SerialTreeScheduler().prepare(objective) as schedule:
        results = list(schedule.evaluate_many(requests))

    assert [(result.index, result.value) for result in results] == [
        (0, 1.0),
        (1, 2.0),
        (2, 3.0),
    ]
    for idx, (params, ctx) in enumerate(zip(parameters, contexts, strict=True)):
        assert ctx.parameters is params
        assert ctx.meta == {
            "context_identity": id(ctx),
            "label": f"run-{idx}",
        }
        assert "children" not in ctx.meta


def test_root_leaf_exception_is_an_ordinary_evaluation_outcome() -> None:
    """An Exception from a root leaf does not poison its prepared schedule."""

    objective = root_leaf
    ctx = EvaluateContext()

    with SerialTreeScheduler().prepare(objective) as schedule:
        (result,) = schedule.evaluate_many(
            [EvaluationRequest(parameters={"value": 7.0, "fail": True}, ctx=ctx)]
        )

        assert result.index == 0
        assert isinstance(result.value, OrdinaryLeafError)
        assert str(result.value) == "root leaf failed for 7.0"
        assert ctx.loss is None
        assert not schedule.closed
        assert schedule.evaluate({"value": 8.0}, EvaluateContext()) == 8.0


def test_executor_yields_root_leaf_runs_in_completion_order() -> None:
    """Executor scheduling parallelizes runs even with one distinct leaf."""

    objective = root_leaf
    parameter_batch: list[Parameters] = [
        {"value": 10.0, "delay": 0.20},
        {"value": 20.0, "delay": 0.00},
        {"value": 30.0, "delay": 0.05},
    ]
    contexts = [EvaluateContext() for _ in parameter_batch]
    requests = [
        EvaluationRequest(parameters=parameters, ctx=ctx)
        for parameters, ctx in zip(parameter_batch, contexts, strict=True)
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        schedule = ExecutorTreeScheduler(executor=executor).prepare(objective)
        results = list(schedule.evaluate_many(requests))
        schedule.close()

    assert [result.index for result in results] == [1, 2, 0]
    assert {result.index: result.value for result in results} == {
        0: 10.0,
        1: 20.0,
        2: 30.0,
    }
    assert all("children" not in ctx.meta for ctx in contexts)


def test_mpi_distributes_root_leaf_runs_across_workers() -> None:
    """MPI assigns separate root-leaf runs across more than one worker."""

    mpi_scheduler = pytest.importorskip(
        "chemfit.mpi_scheduler",
        reason="Missing mpi4py",
    )
    mpi = pytest.importorskip("mpi4py.MPI", reason="Missing mpi4py")
    if mpi.COMM_WORLD.Get_size() < 4:
        pytest.skip("requires one coordinator and at least three workers")

    objective = root_leaf
    with mpi_scheduler.MPITreeScheduler().prepare(objective) as schedule:
        if schedule.rank != 0:
            schedule.worker_loop()
            return

        parameter_batch: list[Parameters] = [
            {"value": float(idx), "record_mpi_rank": True} for idx in range(6)
        ]
        contexts = [EvaluateContext() for _ in parameter_batch]
        requests = [
            EvaluationRequest(parameters=parameters, ctx=ctx)
            for parameters, ctx in zip(parameter_batch, contexts, strict=True)
        ]

        results = sorted(
            schedule.evaluate_many(requests), key=lambda result: result.index
        )

        assert [(result.index, result.value) for result in results] == [
            (idx, float(idx)) for idx in range(6)
        ]
        assert {ctx.meta["worker_rank"] for ctx in contexts} == {1, 2, 3}
        assert all("children" not in ctx.meta for ctx in contexts)
