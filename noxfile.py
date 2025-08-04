import nox


# reduced set of tests for all python versions
@nox.session(python=["3.9", "3.10", "3.11", "3.12", "3.13"])
def tests_all_versions(session):  # noqa: ANN001
    session.install("pytest")
    session.install(".")
    session.run("pytest", "tests/test_fitter.py::test_with_complicated_dict")
    session.run("pytest", "-k", "lj")


# full set of tests for 3.12
@nox.session(python=["3.12"])
def tests(session):  # noqa: ANN001
    session.install("pytest")
    session.install(".")
    session.run("pytest")


# mpi tests for 3.12
@nox.session(python=["3.12"])
def tests_mpi(session):  # noqa: ANN001
    session.install("pytest")
    session.install(".[mpi]")
    session.run("mpiexec", "-n", "2", "pytest", "-k", "mpi")
