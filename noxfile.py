import nox


@nox.session(python=["3.9", "3.10", "3.11", "3.12", "3.13"])
def tests(session):
    session.install("pytest")
    session.install(".")
    session.run("pytest")


@nox.session(python=["3.12"])
def tests_mpi(session):
    session.install("pytest")
    session.install(".[mpi]")
    session.run("mpiexec", "-n", "2", "pytest", "-k", "mpi")
