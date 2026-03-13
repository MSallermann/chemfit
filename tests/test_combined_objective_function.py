import logging
import math
from typing import Any

import numpy as np
import pytest

from chemfit import combined_objective_function
from chemfit.abstract_objective_function import EvaluateContext
from chemfit.executor_wrapper_cob import ExecutorWrapperCOB

logger = logging.getLogger(__name__)

n_terms = 10

funcs = [lambda p, i=i: float(i) for i in range(n_terms)]  # noqa: ARG005
weights = list(range(n_terms))
expected_terms = [float(i**2) for i in range(n_terms)]
cob = combined_objective_function.CombinedObjectiveFunction(
    funcs, weights, reduction=combined_objective_function.sum_reducer
)
reducers = [
    combined_objective_function.sum_reducer,
    # combined_objective_function.mean_reducer,
    # combined_objective_function.root_mean_reducer,
    lambda terms: np.std(terms),
]


def test_combined_objective_function():
    for red in reducers:
        ctx = EvaluateContext()
        cob.reduction = red
        res = cob({}, ctx)

        assert len(ctx.meta["children"]) == n_terms

        assert np.isclose(res, red(expected_terms))


def test_combined_objective_function_async():
    cob_async = ExecutorWrapperCOB(cob)

    for red in reducers:
        cob.reduction = red

        ctx = EvaluateContext()
        res = cob_async({}, ctx)
        assert len(ctx.meta["children"]) == n_terms

        assert np.isclose(res, red(expected_terms))


def test_combined_objective_function_mpi():
    mpi_wrapper_cob = pytest.importorskip(
        "chemfit.mpi_wrapper_cob", reason="Missing mpi4py"
    )

    # Use the MPI Wrapper to make the combined objective function "MPI aware"
    with mpi_wrapper_cob.MPIWrapperCOB(cob, mpi_debug_log=False) as mpi:
        if mpi.rank == 0:
            for red in reducers:
                cob.reduction = red
                ctx = EvaluateContext()
                res = mpi({}, ctx)
                assert len(ctx.meta["children"]) == n_terms
                assert np.isclose(res, red(expected_terms))
        else:
            mpi.worker_loop()


def test_exception_handlers():
    def func1(params: dict[str, Any]) -> float:  # noqa: ARG001
        return 1

    def whoops(params: dict[str, Any]) -> float:  # noqa: ARG001
        msg = "Whoops"
        raise RuntimeError(msg)

    ob = combined_objective_function.CombinedObjectiveFunction([func1, whoops])

    # default is to re-raise, so we should get a runtime error
    with pytest.raises(RuntimeError):
        ob({})

    ctx = EvaluateContext()
    # now we change to the nan exception handler, we should get nan
    ob.exception_handler = combined_objective_function.nan_exception_handler
    res = ob({}, ctx)
    assert math.isnan(res)

    # now we change to the skip exception handler, we should just get the result from func1
    ob.exception_handler = combined_objective_function.skip_exception_handler
    res = ob({})
    print(res)
    assert math.isclose(res, func1({}))

    ####### repeat for async wrapper #######
    ob.exception_handler = combined_objective_function.raising_exception_handler
    ob_async = ExecutorWrapperCOB(ob)
    with pytest.raises(RuntimeError):
        ob_async({})

    # now we change to the nan exception handler, we should get nan
    ob.exception_handler = combined_objective_function.nan_exception_handler
    res = ob_async({})
    print(res)
    assert math.isnan(res)

    # now we change to the skip exception handler, we should just get the result from func1
    ob.exception_handler = combined_objective_function.skip_exception_handler
    res = ob_async({})
    print(res)
    assert math.isclose(res, func1({}))


def test_exception_handlers_mpi():
    logging.basicConfig(filename="bla.long", level=logging.INFO)

    mpi_wrapper_cob = pytest.importorskip(
        "chemfit.mpi_wrapper_cob", reason="Missing mpi4py"
    )

    def func1(params: dict[str, Any]) -> float:  # noqa: ARG001
        return 1

    def whoops(params: dict[str, Any]) -> float:  # noqa: ARG001
        msg = "Whoops"
        raise RuntimeError(msg)

    ob = combined_objective_function.CombinedObjectiveFunction([func1, whoops])
    ob.exception_handler = combined_objective_function.raising_exception_handler
    with mpi_wrapper_cob.MPIWrapperCOB(ob, mpi_debug_log=True) as mpi:
        if mpi.rank == 0:
            with pytest.raises(RuntimeError):
                mpi({})
        else:
            mpi.worker_loop()

    ob.exception_handler = combined_objective_function.nan_exception_handler
    with mpi_wrapper_cob.MPIWrapperCOB(ob, mpi_debug_log=True) as mpi:
        if mpi.rank == 0:
            assert math.isnan(mpi({}))
        else:
            mpi.worker_loop()

    ob.exception_handler = combined_objective_function.skip_exception_handler
    with mpi_wrapper_cob.MPIWrapperCOB(ob, mpi_debug_log=True) as mpi:
        if mpi.rank == 0:
            assert mpi({}) == func1({})
        else:
            mpi.worker_loop()
