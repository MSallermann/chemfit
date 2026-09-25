from __future__ import annotations

import math
import os
from typing import Any

import pytest

from chemfit.abstract_objective_function import EvaluateContext, ObjectiveFunctor
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.executor_wrapper_cob import ExecutorWrapperCOB

PARAMETERS = {"x": 2.0}
# With four MPI ranks, four terms ensure that every rank evaluates a term.
N_TERMS = 4


class HookedObjective(ObjectiveFunctor[dict[str, float]]):
    def __init__(self, offset: float) -> None:
        """Initialize the objective with its term-specific offset."""
        super().__init__()
        self.offset = offset

    def _evaluate(self, parameters: dict[str, float], ctx: EvaluateContext) -> float:
        ctx.meta["evaluation_pid"] = os.getpid()
        return parameters["x"] + self.offset


def record_pre_hook(ctx: EvaluateContext) -> None:
    # Hook observations belong to the evaluation context. The executor and MPI
    # wrappers explicitly propagate this metadata back from worker processes.
    assert ctx.parameters is not None
    ctx.meta["pre_parameters"] = dict(ctx.parameters)
    ctx.meta["pre_loss"] = ctx.loss


def record_post_hook(ctx: EvaluateContext) -> None:
    ctx.meta["post_loss"] = ctx.loss
    ctx.meta["post_exception"] = ctx.temp.exception


def failing_post_hook(ctx: EvaluateContext) -> None:
    msg = f"post hook failed after loss {ctx.loss}"
    raise ValueError(msg)


def make_hooked_cob(*, failing: bool = False) -> CombinedObjectiveFunction:
    terms = []
    for offset in range(N_TERMS):
        term = HookedObjective(float(offset))
        term.register_pre_eval_hook(record_pre_hook)
        term.register_post_eval_hook(record_post_hook)
        if failing and offset == N_TERMS - 1:
            term.register_post_eval_hook(failing_post_hook)
        terms.append(term)
    return CombinedObjectiveFunction(terms)


def register_wrapper_hooks(wrapper: ObjectiveFunctor) -> None:
    wrapper.register_pre_eval_hook(record_pre_hook)
    wrapper.register_post_eval_hook(record_post_hook)


def assert_hook_metadata(metadata: dict[str, Any], expected_loss: float) -> None:
    assert metadata["pre_parameters"] == PARAMETERS
    assert metadata["pre_loss"] is None
    assert metadata["post_loss"] == expected_loss
    assert metadata["post_exception"] is None


def assert_successful_parallel_evaluation(
    result: float,
    ctx: EvaluateContext,
    expected_worker_count: int | None,
) -> None:
    expected_losses = [PARAMETERS["x"] + offset for offset in range(N_TERMS)]

    # These fields were written by hooks on the outer executor/MPI wrapper.
    assert result == sum(expected_losses)
    assert ctx.loss == result
    assert_hook_metadata(ctx.meta, result)

    # These fields were written by hooks around the individual objective terms
    # and had to cross the worker-process boundary in their child contexts.
    children = ctx.meta["children"]
    assert len(children) == N_TERMS
    for child, expected_loss in zip(children, expected_losses, strict=True):
        assert child["parameters"] == PARAMETERS
        assert child["loss"] == expected_loss
        assert_hook_metadata(child["meta"], expected_loss)

    # Process IDs confirm that the terms were genuinely evaluated by workers.
    worker_pids = {child["meta"]["evaluation_pid"] for child in children}
    if expected_worker_count is not None:
        assert len(worker_pids) == expected_worker_count


def assert_post_hook_error(error: ObjectiveFunctor.PostEvalHookError) -> None:
    assert len(error.exceptions) == 1
    assert isinstance(error.exceptions[0], ValueError)
    assert str(error.exceptions[0]) == "post hook failed after loss 5.0"


def test_objective_hooks_with_loky_process_pool():
    loky = pytest.importorskip("loky", reason="Missing loky")
    cob = make_hooked_cob()

    with loky.ProcessPoolExecutor(2) as executor:
        wrapped = ExecutorWrapperCOB(cob, executor=executor)
        register_wrapper_hooks(wrapped)
        ctx = EvaluateContext()
        result = wrapped(PARAMETERS, ctx)

    assert_successful_parallel_evaluation(result, ctx, expected_worker_count=None)
    child_pids = {child["meta"]["evaluation_pid"] for child in ctx.meta["children"]}
    assert os.getpid() not in child_pids


def test_post_hook_error_crosses_loky_process_boundary():
    loky = pytest.importorskip("loky", reason="Missing loky")
    cob = make_hooked_cob(failing=True)

    with loky.ProcessPoolExecutor(2) as executor:
        wrapped = ExecutorWrapperCOB(cob, executor=executor)
        # This also exercises PostEvalHookError.__reduce__: loky must serialize
        # both the aggregate error and its nested ValueError back to this process.
        with pytest.raises(ObjectiveFunctor.PostEvalHookError) as exc_info:
            wrapped(PARAMETERS)

    assert_post_hook_error(exc_info.value)


def test_objective_hooks_with_mpi():
    mpi_wrapper_cob = pytest.importorskip(
        "chemfit.mpi_wrapper_cob", reason="Missing mpi4py"
    )
    cob = make_hooked_cob()

    with mpi_wrapper_cob.MPIWrapperCOB(cob, mpi_debug_log=False) as wrapped:
        register_wrapper_hooks(wrapped)
        if wrapped.rank == 0:
            ctx = EvaluateContext()
            result = wrapped(PARAMETERS, ctx)
            # Match MPIWrapperCOB's ceiling-based partitioning so this remains
            # valid for MPI runs with a rank count other than four as well.
            terms_per_rank = math.ceil(N_TERMS / wrapped.size)
            active_ranks = math.ceil(N_TERMS / terms_per_rank)
            assert_successful_parallel_evaluation(
                result,
                ctx,
                expected_worker_count=active_ranks,
            )
        else:
            wrapped.worker_loop()


def test_post_hook_error_crosses_mpi_process_boundary():
    mpi_wrapper_cob = pytest.importorskip(
        "chemfit.mpi_wrapper_cob", reason="Missing mpi4py"
    )
    cob = make_hooked_cob(failing=True)

    with mpi_wrapper_cob.MPIWrapperCOB(cob, mpi_debug_log=False) as wrapped:
        if wrapped.rank == 0:
            # The failing term is the last one, which runs on a worker rank in
            # the four-rank test. MPI must transport the aggregate hook error.
            with pytest.raises(ObjectiveFunctor.PostEvalHookError) as exc_info:
                wrapped(PARAMETERS)
            assert_post_hook_error(exc_info.value)
        else:
            wrapped.worker_loop()
