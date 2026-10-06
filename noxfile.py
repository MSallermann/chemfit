import nox
from pathlib import Path


# mpi tests
@nox.session(python=["3.10"])
def tests_mpi(session):  # noqa: ANN001
    session.install(".[test,mpi]")

    # Make 100% sure that mpiexec used the correct python interpreter
    python = str(Path(session.virtualenv.location).resolve() / "bin" / "python")

    # Smoketest for mpi import:
    # -u disables Python output buffering,
    # external=True removes the harmless Nox warning
    # about mpiexec being outside the virtual environment.
    session.run(
        "mpiexec",
        "-n",
        "4",
        python,
        "-u",
        "-c",
        "from mpi4py import MPI; print(MPI.COMM_WORLD.rank, flush=True)",
        external=True,
    )

    # -s disables pytest output capture.
    session.run(
        "mpiexec",
        "-n",
        "4",
        python,
        "-u",
        "-m",
        "pytest",
        "-vv",
        "-s",
        "tests",
        "-k",
        "mpi",
        external=True,
    )


# full set of tests for all python versions
@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests_all_versions(session):  # noqa: ANN001
    session.install(".[test]")
    session.run("pytest", "tests/")


# strict static API checks
@nox.session
def typing(session):  # noqa: ANN001
    session.install(".[test]", "pyright")
    session.run("pyright", "tests/static_typing.py")
