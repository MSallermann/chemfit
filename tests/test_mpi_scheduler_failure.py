"""Tests for catastrophic MPI worker failure handling."""

import pickle
from time import sleep
from typing import Any

import pytest

from chemfit.abstract_objective_function import Component, EvaluateContext, component
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.scheduling import EvaluationRequest

MPI = pytest.importorskip("mpi4py.MPI", reason="Missing mpi4py")
mpi_schedule = pytest.importorskip(
    "chemfit.mpi_graph_schedule",
    reason="Missing mpi4py",
)

Parameters = dict[str, float]
_LARGE_PAYLOAD_SIZE = 16 * 1024 * 1024


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
        return mpi_schedule._Shutdown(0)  # noqa: SLF001

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

    state = mpi_schedule._WorkerProtocolState()  # noqa: SLF001
    schedule._send_worker_result(result, state)  # noqa: SLF001
    schedule._finish_worker_shutdown(state)  # noqa: SLF001
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
    schedule._last_command_sequence = {1: 0}  # noqa: SLF001
    task = mpi_schedule.NodeTask(
        run_id=0,
        node_id=0,
        input=None,
        ctx=EvaluateContext(),
    )
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
    assert message == mpi_schedule._EvaluateBatch((task,), 1)  # noqa: SLF001
    assert destination == 1
    assert tag == mpi_schedule._COMMAND_TAG  # noqa: SLF001
    assert not request.waited


def test_worker_loop_consumes_task_prefetched_during_result_send() -> None:
    """A rendezvous result send can receive the coordinator's next task."""

    schedule = object.__new__(mpi_schedule.MPIGraphSchedule)
    schedule.rank = 2
    first_task: Any = object()
    second_task: Any = object()
    commands = [mpi_schedule._EvaluateBatch((first_task,), 1)]  # noqa: SLF001
    evaluated: list[object] = []
    acknowledgements: list[None] = []

    def next_command() -> object:
        return commands.pop(0)

    def evaluate_task(task: object, *, capture_context_state: bool) -> object:
        assert capture_context_state
        evaluated.append(task)
        return task

    def send_result(result: object, state: Any) -> None:
        if result == mpi_schedule._BatchCompletion((first_task,)):  # noqa: SLF001
            schedule._record_worker_batch(  # noqa: SLF001
                state,
                mpi_schedule._EvaluateBatch((second_task,), 2),  # noqa: SLF001
                prefetch=True,
            )
        else:
            schedule._record_worker_shutdown(  # noqa: SLF001
                state,
                mpi_schedule._Shutdown(2),  # noqa: SLF001
            )

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
    commands: list[object] = [KeyboardInterrupt("stop"), mpi_schedule._Shutdown(0)]  # noqa: SLF001
    failures: list[object] = []
    acknowledgements: list[None] = []

    def next_command() -> object:
        command = commands.pop(0)
        if isinstance(command, BaseException):
            raise command
        return command

    def report_failure(exception: BaseException, _state: Any) -> None:
        failures.append(exception)

    def acknowledge_shutdown() -> None:
        acknowledgements.append(None)

    schedule._next_worker_command = next_command  # type: ignore[method-assign]  # noqa: SLF001
    schedule._report_worker_failure = report_failure  # type: ignore[method-assign]  # noqa: SLF001
    schedule._acknowledge_shutdown = acknowledge_shutdown  # type: ignore[method-assign]  # noqa: SLF001

    schedule.worker_loop()

    assert len(failures) == 1
    assert isinstance(failures[0], KeyboardInterrupt)
    assert acknowledgements == [None]


def test_shutdown_fence_retires_eager_command_after_protocol_error() -> None:
    """Shutdown consumes a queued command and acknowledges once after failure."""

    class ResultRequest:
        def __init__(self, pending_tests: int) -> None:
            self.pending_tests = pending_tests

        def Test(self) -> bool:  # noqa: N802
            self.pending_tests -= 1
            return self.pending_tests < 0

    class ProtocolComm:
        def __init__(self) -> None:
            self.controls = [object(), mpi_schedule._Shutdown(2)]  # noqa: SLF001
            self.commands = [
                mpi_schedule._EvaluateBatch((second_task,), 2),  # noqa: SLF001
            ]
            self.results: list[object] = []
            self.acknowledgements = 0

        def isend(self, result: object, *, dest: int, tag: int) -> ResultRequest:
            del dest, tag
            self.results.append(result)
            return ResultRequest(2 if len(self.results) == 1 else -1)

        def Iprobe(self, *, source: int, tag: int) -> bool:  # noqa: N802
            del source
            if tag == mpi_schedule._CONTROL_TAG:  # noqa: SLF001
                return bool(self.controls)
            return bool(self.commands)

        def recv(self, *, source: int, tag: int) -> object:
            del source
            if tag == mpi_schedule._CONTROL_TAG:  # noqa: SLF001
                return self.controls.pop(0)
            return self.commands.pop(0)

        def send(self, message: object, *, dest: int, tag: int) -> None:
            del dest, tag
            assert isinstance(message, mpi_schedule._ShutdownAcknowledgement)  # noqa: SLF001
            self.acknowledgements += 1

    schedule = object.__new__(mpi_schedule.MPIGraphSchedule)
    schedule.rank = 1
    first_task: Any = object()
    second_task: Any = object()
    evaluated: list[object] = []

    def evaluate_task(task: object, *, capture_context_state: bool) -> object:
        assert capture_context_state
        evaluated.append(task)
        return task

    schedule.comm = ProtocolComm()
    schedule._next_worker_command = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: mpi_schedule._EvaluateBatch((first_task,), 1)  # noqa: SLF001
    )
    schedule._evaluate_worker_task = evaluate_task  # type: ignore[method-assign]  # noqa: SLF001

    schedule.worker_loop()

    assert evaluated == [first_task]
    assert schedule.comm.commands == []
    assert isinstance(schedule.comm.results[0], mpi_schedule._BatchCompletion)  # noqa: SLF001
    assert isinstance(schedule.comm.results[1], mpi_schedule._WorkerFailure)  # noqa: SLF001
    assert schedule.comm.acknowledgements == 1


def test_shutdown_cleanup_preserves_primary_failure() -> None:
    """Unexpected shutdown traffic is recorded only after workers acknowledge."""

    class ShutdownDrainComm:
        def __init__(self) -> None:
            self.messages = [
                (1, 999, object()),
                (
                    1,
                    mpi_schedule._CONTROL_TAG,  # noqa: SLF001
                    mpi_schedule._ShutdownAcknowledgement(),  # noqa: SLF001
                ),
            ]

        def isend(self, message: object, *, dest: int, tag: int) -> Any:
            del message, dest, tag
            return MPI.REQUEST_NULL

        def recv(self, *, source: int, tag: int, status: Any) -> object:
            del source, tag
            message_source, message_tag, message = self.messages.pop(0)
            status.Set_source(message_source)
            status.Set_tag(message_tag)
            return message

    schedule = object.__new__(mpi_schedule.MPIGraphSchedule)
    schedule._closed = False  # noqa: SLF001
    schedule.rank = 0
    schedule.size = 2
    schedule.comm = ShutdownDrainComm()
    schedule._workers_released = False  # noqa: SLF001
    schedule._last_command_sequence = {1: 0}  # noqa: SLF001
    primary = RuntimeError("primary scheduling failure")

    schedule._close_after_catastrophic_failure(primary)  # noqa: SLF001

    assert schedule.closed
    assert schedule._workers_released  # noqa: SLF001
    assert schedule.comm.messages == []
    if hasattr(primary, "__notes__"):
        assert "Unexpected MPI message" in "\n".join(primary.__notes__)


def test_concurrent_mpi_evaluation_is_rejected() -> None:
    """One prepared MPI schedule owns at most one active batch."""

    class SingleRankComm:
        def Get_rank(self) -> int:  # noqa: N802
            return 0

        def Get_size(self) -> int:  # noqa: N802
            return 1

    @component()
    def identity(value: int) -> int:
        return value

    schedule = mpi_schedule.MPIGraphScheduler(comm=SingleRankComm()).prepare(identity)
    first = schedule.evaluate_many(
        [
            EvaluationRequest(1, EvaluateContext()),
            EvaluationRequest(2, EvaluateContext()),
        ]
    )
    assert next(first).value == 1

    concurrent = schedule.evaluate_many([EvaluationRequest(3, EvaluateContext())])
    with pytest.raises(RuntimeError, match="Concurrent evaluation"):
        next(concurrent)

    assert [result.value for result in first] == [2]


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


@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() < 4, reason="requires four MPI ranks")
def test_shutdown_drains_large_worker_results_after_fatal_failure() -> None:
    """Complete rendezvous result sends before workers acknowledge shutdown."""

    @component()
    def fail_or_return(item: tuple[str, bytes]) -> bytes:
        mode, payload = item
        if mode == "fail":
            msg = "live large-payload failure"
            raise KeyboardInterrupt(msg)
        sleep(0.25)
        return payload

    with mpi_schedule.MPIGraphScheduler(max_tasks_per_message=1).prepare(
        fail_or_return
    ) as schedule:
        if schedule.rank != 0:
            schedule.worker_loop()
            return

        payload = b"x" * _LARGE_PAYLOAD_SIZE
        requests = [
            # Deliver the fatal command eagerly. The other command sends still
            # have to complete before abort can initiate shutdown, ensuring the
            # slow workers are evaluating when the control messages arrive.
            EvaluationRequest(("fail", b""), EvaluateContext()),
            EvaluationRequest(("slow", payload), EvaluateContext()),
            EvaluationRequest(("slow", payload), EvaluateContext()),
        ]
        expected = "live large-payload failure"
        with pytest.raises(mpi_schedule.MPIWorkerError, match=expected):
            list(schedule.evaluate_many(requests))

        assert schedule.closed
