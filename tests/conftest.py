try:
    import mpi4py
except ImportError:
    mpi4py = None

import pytest


@pytest.fixture(scope="session", autouse=True)
def mpi_finalize_fixture():
    yield

    if mpi4py is not None:
        # This teardown part runs after the session
        if not mpi4py.MPI.Is_finalized():
            mpi4py.MPI.Finalize()
