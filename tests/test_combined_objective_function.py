import numpy as np
import pytest

from chemfit import combined_objective_function
from chemfit.async_wrapper_cob import AsyncWrapperCOB

n_terms = 10

funcs = [lambda p, i=i: float(i) for i in range(n_terms)]  # noqa: ARG005
weights = list(range(n_terms))
expected_terms = [float(i**2) for i in range(n_terms)]
cob = combined_objective_function.CombinedObjectiveFunction(
    funcs, weights, reduction=combined_objective_function.sum_reducer
)


def test_combined_objective_function():
    reducers = [
        combined_objective_function.sum_reducer,
        combined_objective_function.mean_reducer,
        combined_objective_function.root_mean_reducer,
        lambda terms: np.std(terms),
    ]

    for red in reducers:
        cob.reduction = red
        res = cob({})

        assert np.isclose(res, red(expected_terms))


def test_combined_objective_function_async():
    reducers = [
        combined_objective_function.sum_reducer,
        combined_objective_function.mean_reducer,
        combined_objective_function.root_mean_reducer,
        lambda terms: np.std(terms),
    ]

    cob_async = AsyncWrapperCOB(cob)

    for red in reducers:
        cob.reduction = red
        res = cob_async({})

        assert np.isclose(res, red(expected_terms))


def test_combined_objective_function_mpi():
    mpi_wrapper_cob = pytest.importorskip(
        "chemfit.mpi_wrapper_cob", reason="Missing mpi4py"
    )

    reducers = [
        combined_objective_function.sum_reducer,
        combined_objective_function.mean_reducer,
        combined_objective_function.root_mean_reducer,
        lambda terms: np.std(terms),
    ]

    # Use the MPI Wrapper to make the combined objective function "MPI aware"
    with mpi_wrapper_cob.MPIWrapperCOB(cob) as mpi:
        for red in reducers:
            if mpi.rank == 0:
                cob.reduction = red
                res = mpi({})
                assert np.isclose(res, red(expected_terms))
            else:
                mpi.worker_loop()
