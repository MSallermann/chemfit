"""Tests for catastrophic MPI worker failure handling."""

import pickle
from typing import Any

import pytest

from chemfit.abstract_objective_function import Component, EvaluateContext
from chemfit.combined_objective_function import CombinedObjectiveFunction

MPI = pytest.importorskip("mpi4py.MPI", reason="Missing mpi4py")
mpi_schedule = pytest.importorskip(
    "chemfit.mpi_graph_schedule",
    reason="Missing mpi4py",
)

Parameters = dict[str, float]


class UnpickleableInterrupt(KeyboardInterrupt):
    """A catastrophic failure that cannot itself cross an MPI boundary."""

    def __reduce__(self) -> Any:
        """Reject attempts to pickle the original exception."""

        msg = "exception cannot be pickled"
        raise TypeError(msg)


class CatastrophicLeaf(Component[Parameters, float]):
    """Raise an unpickleable BaseException on an MPI worker."""

    def _run(self, parameters: Parameters, /, ctx: EvaluateContext) -> float:
        """Interrupt worker evaluation."""

        del parameters, ctx
        msg = "remote catastrophe"
        raise UnpickleableInterrupt(msg)


class PendingRequest:
    """Complete a simulated result send after shutdown has been observed."""

    def __init__(self) -> None:
        """Initialize a result request that needs one progress iteration."""

        self.test_count = 0

    def Test(self) -> bool:  # noqa: N802
        """Report completion after the worker has handled one message."""

        self.test_count += 1
        return self.test_count > 1


class ShutdownComm:
    """Simulate shutdown arriving while a worker result send is pending."""

    def __init__(self) -> None:
        """Initialize the fake communicator and pending request."""

        self.request = PendingRequest()
        self.shutdown_received = False
        self.sent: list[tuple[object, int, int]] = []

    def isend(self, result: object, *, dest: int, tag: int) -> PendingRequest:
        """Start a send that remains pending."""

        del result, dest, tag
        return self.request

    def Iprobe(self, *, source: int, tag: int) -> bool:  # noqa: N802
        """Report a waiting coordinator command."""

        del source
        return tag == mpi_schedule._CONTROL_TAG and not self.shutdown_received  # noqa: SLF001

    def recv(self, *, source: int, tag: int) -> object:
        """Return the queued shutdown command."""

        del source, tag
        self.shutdown_received = True
        return mpi_schedule._Shutdown()  # noqa: SLF001

    def send(self, message: object, *, dest: int, tag: int) -> None:
        """Record the worker's shutdown acknowledgement."""

        self.sent.append((message, dest, tag))


class CommandRequest:
    """Record whether a retained nonblocking task send was completed."""

    def __init__(self) -> None:
        """Initialize an incomplete request."""

        self.waited = False

    def Wait(self) -> None:  # noqa: N802
        """Complete the simulated command send."""

        self.waited = True


class CommandComm:
    """Reject blocking task sends and retain nonblocking requests."""

    def __init__(self) -> None:
        """Initialize the command-send log."""

        self.sent: list[tuple[object, int, int, CommandRequest]] = []

    def send(self, message: object, *, dest: int, tag: int) -> None:
        """Fail if task dispatch uses a potentially blocking send."""

        del message, dest, tag
        msg = "task dispatch must be nonblocking"
        raise AssertionError(msg)

    def isend(self, message: object, *, dest: int, tag: int) -> CommandRequest:
        """Return and retain a simulated nonblocking send request."""

        request = CommandRequest()
        self.sent.append((message, dest, tag, request))
        return request


def test_worker_failure_payload_does_not_pickle_original_exception() -> None:
    """Only primitive failure details cross the communicator."""

    exception = UnpickleableInterrupt("worker interrupted")
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(exception)

    failure = mpi_schedule._describe_worker_failure(3, exception)  # noqa: SLF001

    assert pickle.loads(pickle.dumps(failure)) == failure  # noqa: S301
    assert failure.rank == 3
    assert failure.exception_type == "UnpickleableInterrupt"
    assert failure.message == "worker interrupted"


def test_shutdown_does_not_cancel_pending_worker_batch_result() -> None:
    """Regression: shutdown completes an in-transit batch without MPI_Cancel."""

    schedule = object.__new__(mpi_schedule.MPIGraphSchedule)
    schedule.comm = ShutdownComm()
    schedule.rank = 2
    result = mpi_schedule._BatchCompletion(  # noqa: SLF001
        (
            mpi_schedule.NodeCompletion(
                run_id=0,
                node_id=1,
                outcome=2,
            ),
        )
    )

    assert not schedule._send_worker_result(result)  # noqa: SLF001
    schedule._acknowledge_shutdown()  # noqa: SLF001
    assert schedule.comm.sent == [
        (mpi_schedule._ShutdownAcknowledgement(), 0, mpi_schedule._CONTROL_TAG)  # noqa: SLF001
    ]


def test_task_dispatch_retains_nonblocking_command_send() -> None:
    """Large task dispatch must not block against a worker result send."""

    schedule = object.__new__(mpi_schedule.MPIGraphSchedule)
    schedule.comm = CommandComm()
    schedule.size = 2
    schedule.max_tasks_per_message = 8
    schedule.target_batches_per_worker = 8
    task: Any = object()
    batch_state: Any = mpi_schedule.BatchState()
    batch_state.pending_tasks = mpi_schedule.deque([task])
    batch_state.in_flight = {}
    batch_state.command_sends = {}
    batch_state.idle_workers = mpi_schedule.deque([1])

    schedule._dispatch_available(batch_state)  # noqa: SLF001

    assert batch_state.pending_tasks == mpi_schedule.deque()
    assert batch_state.in_flight == {1: (task,)}
    assert batch_state.command_sends == {1: schedule.comm.sent[0][3]}
    message, destination, tag, request = schedule.comm.sent[0]
    assert message == mpi_schedule._EvaluateBatch((task,))  # noqa: SLF001
    assert destination == 1
    assert tag == mpi_schedule._COMMAND_TAG  # noqa: SLF001
    assert not request.waited


def test_worker_loop_consumes_task_prefetched_during_result_send() -> None:
    """A rendezvous result send can receive the coordinator's next task."""

    schedule = object.__new__(mpi_schedule.MPIGraphSchedule)
    schedule.rank = 2
    first_task: Any = object()
    second_task: Any = object()
    commands = [mpi_schedule._EvaluateBatch((first_task,))]  # noqa: SLF001
    evaluated: list[object] = []
    acknowledgements: list[None] = []

    def next_command() -> object:
        return commands.pop(0)

    def evaluate_task(task: object, *, capture_context_state: bool) -> object:
        assert capture_context_state
        evaluated.append(task)
        return task

    def send_result(result: object, prefetched: Any = None) -> bool:
        assert prefetched is not None
        if result == mpi_schedule._BatchCompletion((first_task,)):  # noqa: SLF001
            prefetched.append(mpi_schedule._EvaluateBatch((second_task,)))  # noqa: SLF001
            return True
        return False

    def acknowledge_shutdown() -> None:
        acknowledgements.append(None)

    schedule._next_worker_command = next_command  # type: ignore[method-assign]  # noqa: SLF001
    schedule._evaluate_worker_task = evaluate_task  # type: ignore[method-assign]  # noqa: SLF001
    schedule._send_worker_result = send_result  # type: ignore[method-assign]  # noqa: SLF001
    schedule._acknowledge_shutdown = acknowledge_shutdown  # type: ignore[method-assign]  # noqa: SLF001

    schedule.worker_loop()

    assert evaluated == [first_task, second_task]
    assert commands == []
    assert acknowledgements == [None]


def test_worker_interrupt_is_reported_before_shutdown_acknowledgement() -> None:
    """An interrupt while receiving work follows the normal failure protocol."""

    schedule = object.__new__(mpi_schedule.MPIGraphSchedule)
    schedule.rank = 2
    commands: list[object] = [KeyboardInterrupt("stop"), mpi_schedule._Shutdown()]  # noqa: SLF001
    failures: list[object] = []
    acknowledgements: list[None] = []

    def next_command() -> object:
        command = commands.pop(0)
        if isinstance(command, BaseException):
            raise command
        return command

    def report_failure(exception: BaseException) -> bool:
        failures.append(exception)
        return True

    def acknowledge_shutdown() -> None:
        acknowledgements.append(None)

    schedule._next_worker_command = next_command  # type: ignore[method-assign]  # noqa: SLF001
    schedule._report_worker_failure = report_failure  # type: ignore[method-assign]  # noqa: SLF001
    schedule._acknowledge_shutdown = acknowledge_shutdown  # type: ignore[method-assign]  # noqa: SLF001

    schedule.worker_loop()

    assert len(failures) == 1
    assert isinstance(failures[0], KeyboardInterrupt)
    assert acknowledgements == [None]


@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() < 2, reason="requires MPI workers")
def test_unpickleable_worker_failure_reaches_rank_zero_as_mpi_error() -> None:
    """Reconstruct a serializable worker failure as MPIWorkerError."""

    objective = CombinedObjectiveFunction([CatastrophicLeaf()])
    with mpi_schedule.MPIGraphScheduler().prepare(objective) as schedule:
        if schedule.rank != 0:
            schedule.worker_loop()
            return

        expected = "Worker rank 1 failed with UnpickleableInterrupt: remote catastrophe"
        with pytest.raises(mpi_schedule.MPIWorkerError, match=expected):
            schedule.evaluate({"x": 1.0}, EvaluateContext())

        assert schedule.closed
