import numpy as np
import pytest

from chemfit import combined_objective_function
from chemfit.abstract_objective_function import EvaluateContext
from chemfit.async_wrapper_cob import AsyncWrapperCOB

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
    cob_async = AsyncWrapperCOB(cob)

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
