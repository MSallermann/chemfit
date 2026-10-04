import nox


# full set of tests for all python versions
@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests_all_versions(session):  # noqa: ANN001
    session.install(".[test,mpi]")
    session.run("pytest", "tests/")


# mpi tests
@nox.session(python=["3.10"])
def tests_mpi(session):  # noqa: ANN001
    session.install(".[test,mpi]")
    session.run("mpiexec", "-n", "4", "pytest", "tests", "-k", "mpi")


# strict static API checks
@nox.session
def typing(session):  # noqa: ANN001
    session.install(".[test]", "pyright")
    session.run("pyright", "tests/static_typing.py")
