"""Tests for catastrophic MPI worker failure handling."""

import pickle
from typing import Any

import pytest

from chemfit.abstract_objective_function import EvaluateContext, ObjectiveFunctor
from chemfit.combined_objective_function import CombinedObjectiveFunction

MPI = pytest.importorskip("mpi4py.MPI", reason="Missing mpi4py")
mpi_scheduler = pytest.importorskip(
    "chemfit.mpi_scheduler",
    reason="Missing mpi4py",
)

Parameters = dict[str, float]


class UnpickleableInterrupt(KeyboardInterrupt):
    """A catastrophic failure that cannot itself cross an MPI boundary."""

    def __reduce__(self) -> Any:
        """Reject attempts to pickle the original exception."""

        msg = "exception cannot be pickled"
        raise TypeError(msg)


class CatastrophicLeaf(ObjectiveFunctor[Parameters]):
    """Raise an unpickleable BaseException on an MPI worker."""

    def _evaluate(self, parameters: Parameters, ctx: EvaluateContext) -> float:
        """Interrupt worker evaluation."""

        del parameters, ctx
        msg = "remote catastrophe"
        raise UnpickleableInterrupt(msg)


class PendingRequest:
    """Record cancellation of a send that never completes."""

    def __init__(self) -> None:
        """Initialize an incomplete request."""

        self.cancelled = False
        self.waited = False

    def Test(self) -> bool:  # noqa: N802
        """Report that the simulated large send is still pending."""

        return False

    def Cancel(self) -> None:  # noqa: N802
        """Record best-effort cancellation."""

        self.cancelled = True

    def Wait(self) -> None:  # noqa: N802
        """Record completion of the local cancellation."""

        self.waited = True


class ShutdownComm:
    """Simulate shutdown arriving while a worker result send is pending."""

    def __init__(self) -> None:
        """Initialize the fake communicator and pending request."""

        self.request = PendingRequest()

    def isend(self, result: object, *, dest: int, tag: int) -> PendingRequest:
        """Start a send that remains pending."""

        del result, dest, tag
        return self.request

    def Iprobe(self, *, source: int, tag: int) -> bool:  # noqa: N802
        """Report a waiting coordinator command."""

        del source, tag
        return True

    def recv(self, *, source: int, tag: int) -> object:
        """Return the queued shutdown command."""

        del source, tag
        return mpi_scheduler._Shutdown()  # noqa: SLF001


def test_worker_failure_payload_does_not_pickle_original_exception() -> None:
    """Only primitive failure details cross the communicator."""

    exception = UnpickleableInterrupt("worker interrupted")
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(exception)

    failure = mpi_scheduler._describe_worker_failure(3, exception)  # noqa: SLF001

    assert pickle.loads(pickle.dumps(failure)) == failure  # noqa: S301
    assert failure.rank == 3
    assert failure.exception_type == "UnpickleableInterrupt"
    assert failure.message == "worker interrupted"


def test_pending_worker_send_is_interrupted_by_shutdown() -> None:
    """A worker can consume shutdown without completing its result send."""

    schedule = object.__new__(mpi_scheduler.MPITreeSchedule)
    schedule.comm = ShutdownComm()
    schedule.rank = 2
    failure = mpi_scheduler._WorkerFailure(  # noqa: SLF001
        rank=2,
        exception_type="KeyboardInterrupt",
        message="stop",
    )

    assert not schedule._send_worker_result(failure)  # noqa: SLF001
    assert schedule.comm.request.cancelled
    assert schedule.comm.request.waited


@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() < 2, reason="requires MPI workers")
def test_unpickleable_worker_failure_reaches_rank_zero_as_mpi_error() -> None:
    """Reconstruct a serializable worker failure as MPIWorkerError."""

    objective = CombinedObjectiveFunction([CatastrophicLeaf()])
    with objective.set_scheduler(mpi_scheduler.MPITreeScheduler()) as schedule:
        if schedule.rank != 0:
            schedule.worker_loop()
            return

        expected = "Worker rank 1 failed with UnpickleableInterrupt: remote catastrophe"
        with pytest.raises(mpi_scheduler.MPIWorkerError, match=expected):
            objective({"x": 1.0}, EvaluateContext())

        assert schedule.closed
