"""Focused tests for adaptive MPI task batching."""

from collections import deque
from time import sleep
from typing import Any

import pytest

from chemfit.abstract_objective_function import (
    EvaluateContext,
    PipelineComponent,
    component,
)
from chemfit.api import join
from chemfit.graph_schedule_base import BatchState, NodeCompletion
from chemfit.scheduling import EvaluationRequest

MPI = pytest.importorskip("mpi4py.MPI", reason="Missing mpi4py")
mpi_schedule = pytest.importorskip(
    "chemfit.mpi_graph_schedule",
    reason="Missing mpi4py",
)

_LARGE_PAYLOAD_SIZE = 2 * 1024 * 1024


class CompletedRequest:
    """Record when a retained command send is released."""

    def __init__(self) -> None:
        """Create an incomplete request."""
        self.waited = False

    def Wait(self) -> None:  # noqa: N802
        self.waited = True


class RecordingComm:
    """Record coordinator sends and serve queued worker results."""

    def __init__(
        self,
        *,
        size: int = 4,
        results: list[tuple[int, object]] | None = None,
    ) -> None:
        """Initialize the communicator logs and queued results."""
        self.size = size
        self.results = deque(results or ())
        self.sent: list[tuple[object, int, int, CompletedRequest]] = []
        self.events: list[tuple[str, int]] = []

    def Get_rank(self) -> int:  # noqa: N802
        return 0

    def Get_size(self) -> int:  # noqa: N802
        return self.size

    def isend(self, message: object, *, dest: int, tag: int) -> CompletedRequest:
        request = CompletedRequest()
        self.sent.append((message, dest, tag, request))
        self.events.append(("send", dest))
        return request

    def recv(self, *, source: int, tag: int, status: Any) -> object:
        del source, tag
        result_source, message = self.results.popleft()
        self.events.append(("recv", result_source))
        status.Set_source(result_source)
        return message

    def Iprobe(self, *, source: int, tag: int) -> bool:  # noqa: N802
        del source, tag
        return bool(self.results)


class LiveRecordingComm:
    """Delegate to a live communicator while recording dispatched batch sizes."""

    def __init__(self, comm: Any) -> None:
        """Wrap a live MPI communicator."""
        self.comm = comm
        self.batch_sizes: list[int] = []

    def __getattr__(self, name: str) -> Any:
        """Delegate communicator operations not customized by the test."""
        return getattr(self.comm, name)

    def isend(self, message: object, *, dest: int, tag: int) -> Any:
        """Record task-batch sizes before starting the real nonblocking send."""
        if isinstance(message, mpi_schedule._EvaluateBatch):  # noqa: SLF001
            self.batch_sizes.append(len(message.tasks))
        return self.comm.isend(message, dest=dest, tag=tag)


def make_schedule(
    comm: RecordingComm,
    *,
    max_tasks_per_message: int = 8,
    target_batches_per_worker: int = 8,
) -> Any:
    """Make a protocol-only schedule without requiring a component graph."""
    schedule = object.__new__(mpi_schedule.MPIGraphSchedule)
    schedule._closed = False  # noqa: SLF001
    schedule.comm = comm
    schedule.rank = 0
    schedule.size = comm.size
    schedule.max_tasks_per_message = max_tasks_per_message
    schedule.target_batches_per_worker = target_batches_per_worker
    schedule._poisoned = False  # noqa: SLF001
    schedule._evaluation_active = False  # noqa: SLF001
    schedule._workers_released = False  # noqa: SLF001
    schedule._last_command_sequence = dict.fromkeys(  # noqa: SLF001
        range(1, schedule.size), 0
    )
    return schedule


def make_task(run_id: int, node_id: int = 0) -> Any:
    """Use a minimal task-shaped object for coordinator-only tests."""
    return mpi_schedule.NodeTask(
        run_id=run_id,
        node_id=node_id,
        input=run_id,
        ctx=EvaluateContext(),
    )


def make_completion(task: Any, outcome: object | None = None) -> NodeCompletion:
    return NodeCompletion(
        run_id=task.run_id,
        node_id=task.node_id,
        outcome=task.input if outcome is None else outcome,
    )


def make_batch_state(
    *,
    pending: tuple[Any, ...] = (),
    in_flight: dict[int, tuple[Any, ...]] | None = None,
    command_sends: dict[int, CompletedRequest] | None = None,
    idle_workers: tuple[int, ...] = (),
    dispatch_batch_size: int | None = None,
    dispatch_tail_started: bool = False,
) -> Any:
    state = BatchState()
    state.pending_tasks = deque(pending)
    state.in_flight = {} if in_flight is None else in_flight
    state.command_sends = {} if command_sends is None else command_sends
    state.idle_workers = deque(idle_workers)
    state.dispatch_batch_size = dispatch_batch_size
    state.dispatch_tail_started = dispatch_tail_started
    return state


def test_dispatch_builds_exact_batches_and_short_tail() -> None:
    comm = RecordingComm(size=4)
    schedule = make_schedule(
        comm,
        max_tasks_per_message=4,
        target_batches_per_worker=1,
    )
    tasks = tuple(make_task(index) for index in range(10))
    state = make_batch_state(pending=tasks, idle_workers=(1, 2, 3))

    schedule._dispatch_available(state)  # noqa: SLF001

    batches = [message.tasks for message, _, _, _ in comm.sent]
    assert batches == [tasks[:4], tasks[4:8], tasks[8:]]
    assert state.in_flight == {1: tasks[:4], 2: tasks[4:8], 3: tasks[8:]}
    assert not state.pending_tasks
    assert state.dispatch_batch_size is None


def test_refills_retain_original_cohort_batch_size() -> None:
    comm = RecordingComm(size=3)
    schedule = make_schedule(
        comm,
        max_tasks_per_message=16,
        target_batches_per_worker=4,
    )
    tasks = tuple(make_task(index) for index in range(40))
    state = make_batch_state(pending=tasks, idle_workers=(1, 2))

    schedule._dispatch_available(state)  # noqa: SLF001
    first_batch = state.in_flight[1]
    assert len(first_batch) == 5
    assert state.dispatch_batch_size == 5

    comm.results.append(
        (
            1,
            mpi_schedule._BatchCompletion(  # noqa: SLF001
                tuple(make_completion(task) for task in first_batch)
            ),
        )
    )
    schedule.wait_for_completions(state)

    assert [len(message.tasks) for message, _, _, _ in comm.sent] == [5, 5, 5]
    assert state.dispatch_batch_size == 5


def test_appending_to_pending_cohort_retains_batch_size() -> None:
    comm = RecordingComm(size=2)
    schedule = make_schedule(
        comm,
        max_tasks_per_message=16,
        target_batches_per_worker=1,
    )
    original_tasks = tuple(make_task(index) for index in range(20))
    state = make_batch_state(pending=original_tasks, idle_workers=(1,))

    schedule._dispatch_available(state)  # noqa: SLF001
    assert state.dispatch_batch_size == 16

    added_tasks = tuple(make_task(index) for index in range(20, 40))
    schedule.submit_tasks(added_tasks, state)

    assert state.dispatch_batch_size == 16


def test_new_cohort_recalculates_batch_size() -> None:
    comm = RecordingComm(size=3)
    schedule = make_schedule(comm)
    first_cohort = tuple(make_task(index) for index in range(2))
    state = make_batch_state(idle_workers=(1, 2))

    schedule.submit_tasks(first_cohort, state)
    assert [len(message.tasks) for message, _, _, _ in comm.sent] == [1, 1]
    assert state.dispatch_batch_size is None

    state.in_flight.clear()
    state.command_sends.clear()
    state.idle_workers.extend((1, 2))
    second_cohort = tuple(make_task(index) for index in range(2, 34))
    schedule.submit_tasks(second_cohort, state)

    assert [len(message.tasks) for message, _, _, _ in comm.sent[-2:]] == [2, 2]
    assert state.dispatch_batch_size == 2


def test_large_cohort_uses_expected_stable_quantum() -> None:
    comm = RecordingComm(size=1152)
    schedule = make_schedule(
        comm,
        max_tasks_per_message=16,
        target_batches_per_worker=4,
    )

    assert schedule._chunk_size(32_768) == 8  # noqa: SLF001


def test_final_worker_wave_shrinks_to_spread_queue_tail() -> None:
    comm = RecordingComm(size=5)
    schedule = make_schedule(
        comm,
        max_tasks_per_message=8,
        target_batches_per_worker=1,
    )
    tasks = tuple(make_task(index) for index in range(10))
    state = make_batch_state(
        pending=tasks,
        in_flight={
            2: (make_task(20),),
            3: (make_task(21),),
            4: (make_task(22),),
        },
        command_sends={
            2: CompletedRequest(),
            3: CompletedRequest(),
            4: CompletedRequest(),
        },
        idle_workers=(1,),
        dispatch_batch_size=8,
    )

    schedule._dispatch_available(state)  # noqa: SLF001

    assert len(comm.sent[0][0].tasks) == 3
    assert state.dispatch_batch_size == 3
    assert state.dispatch_tail_started
    assert state.pending_tasks == deque(tasks[3:])


def test_adaptive_policy_reduces_narrow_queue_to_single_tasks() -> None:
    comm = RecordingComm(size=4)
    schedule = make_schedule(comm)
    tasks = tuple(make_task(index) for index in range(4))
    state = make_batch_state(pending=tasks, idle_workers=(1, 2, 3))

    schedule._dispatch_available(state)  # noqa: SLF001

    assert [message.tasks for message, _, _, _ in comm.sent] == [
        (tasks[0],),
        (tasks[1],),
        (tasks[2],),
    ]
    assert state.pending_tasks == deque((tasks[3],))


def test_max_tasks_per_message_one_is_compatibility_mode() -> None:
    comm = RecordingComm(size=3)
    schedule = make_schedule(
        comm,
        max_tasks_per_message=1,
        target_batches_per_worker=1,
    )
    tasks = tuple(make_task(index) for index in range(5))
    state = make_batch_state(pending=tasks, idle_workers=(1, 2))

    schedule._dispatch_available(state)  # noqa: SLF001

    assert [message.tasks for message, _, _, _ in comm.sent] == [
        (tasks[0],),
        (tasks[1],),
    ]


def test_one_worker_response_returns_multiple_completions() -> None:
    tasks = (make_task(0), make_task(1))
    completions = tuple(make_completion(task) for task in tasks)
    comm = RecordingComm(
        size=2,
        results=[(1, mpi_schedule._BatchCompletion(completions))],  # noqa: SLF001
    )
    request = CompletedRequest()
    schedule = make_schedule(comm)
    state = make_batch_state(
        in_flight={1: tasks},
        command_sends={1: request},
    )

    assert tuple(schedule.wait_for_completions(state)) == completions
    assert request.waited
    assert state.in_flight == {}
    assert state.command_sends == {}
    assert state.idle_workers == deque((1,))


def test_each_available_worker_is_refilled_before_draining_next_result() -> None:
    first_batch = (make_task(0), make_task(1))
    second_batch = (make_task(2), make_task(3))
    pending = tuple(make_task(index) for index in range(4, 8))
    comm = RecordingComm(
        size=3,
        results=[
            (
                2,
                mpi_schedule._BatchCompletion(  # noqa: SLF001
                    tuple(make_completion(task) for task in second_batch)
                ),
            ),
            (
                1,
                mpi_schedule._BatchCompletion(  # noqa: SLF001
                    tuple(make_completion(task) for task in first_batch)
                ),
            ),
        ],
    )
    schedule = make_schedule(comm)
    state = make_batch_state(
        pending=pending,
        in_flight={1: first_batch, 2: second_batch},
        command_sends={1: CompletedRequest(), 2: CompletedRequest()},
        dispatch_batch_size=2,
    )

    completions = schedule.wait_for_completions(state)

    assert len(completions) == 4
    assert comm.events == [
        ("recv", 2),
        ("send", 2),
        ("recv", 1),
        ("send", 1),
    ]
    assert [destination for _, destination, _, _ in comm.sent] == [2, 1]
    assert not state.pending_tasks


def test_unavailable_worker_does_not_delay_refilling_completed_worker() -> None:
    completed_batch = (make_task(0), make_task(1))
    unavailable_batch = (make_task(2), make_task(3))
    pending = tuple(make_task(index) for index in range(4, 8))
    comm = RecordingComm(
        size=3,
        results=[
            (
                1,
                mpi_schedule._BatchCompletion(  # noqa: SLF001
                    tuple(make_completion(task) for task in completed_batch)
                ),
            )
        ],
    )
    schedule = make_schedule(comm)
    state = make_batch_state(
        pending=pending,
        in_flight={1: completed_batch, 2: unavailable_batch},
        command_sends={1: CompletedRequest(), 2: CompletedRequest()},
        dispatch_batch_size=2,
    )

    completions = schedule.wait_for_completions(state)

    assert len(completions) == 2
    assert comm.events == [("recv", 1), ("send", 1)]
    assert state.in_flight[2] == unavailable_batch
    assert state.in_flight[1] == pending[:2]
    assert state.pending_tasks == deque(pending[2:])


@pytest.mark.parametrize(
    ("returned", "error"),
    [
        (lambda tasks: (make_completion(tasks[0]),) * 2, "duplicate completion"),
        (lambda tasks: (make_completion(tasks[0]),), "omitted completion"),
        (
            lambda tasks: (
                make_completion(tasks[0]),
                NodeCompletion(run_id=99, node_id=0, outcome=None),
            ),
            "unexpected task",
        ),
    ],
)
def test_invalid_completion_batches_are_rejected(returned, error) -> None:
    tasks = (make_task(0), make_task(1))
    comm = RecordingComm(
        size=2,
        results=[
            (1, mpi_schedule._BatchCompletion(returned(tasks))),  # noqa: SLF001
        ],
    )
    request = CompletedRequest()
    schedule = make_schedule(comm)
    state = make_batch_state(
        in_flight={1: tasks},
        command_sends={1: request},
    )

    with pytest.raises(mpi_schedule.MPIWorkerError, match=error):
        schedule._receive_worker_completion(state)  # noqa: SLF001

    assert not request.waited
    assert state.in_flight == {1: tasks}


def test_worker_returns_ordinary_exception_alongside_other_results() -> None:
    schedule = object.__new__(mpi_schedule.MPIGraphSchedule)
    schedule.rank = 1
    tasks = (make_task(0), make_task(1), make_task(2))
    failure = ValueError("ordinary failure")
    commands = deque(
        (
            mpi_schedule._EvaluateBatch(tasks, 1),  # noqa: SLF001
            mpi_schedule._Shutdown(1),  # noqa: SLF001
        )
    )
    sent: list[object] = []
    acknowledgements: list[None] = []

    def evaluate(task: Any, *, capture_context_state: bool) -> NodeCompletion:
        assert capture_context_state
        return make_completion(
            task,
            failure if task is tasks[1] else task.input,
        )

    def send(result: object, _state: object) -> None:
        sent.append(result)

    schedule._next_worker_command = commands.popleft  # type: ignore[method-assign]  # noqa: SLF001
    schedule._evaluate_worker_task = evaluate  # type: ignore[method-assign]  # noqa: SLF001
    schedule._send_worker_result = send  # type: ignore[method-assign]  # noqa: SLF001
    schedule._acknowledge_shutdown = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: acknowledgements.append(None)
    )

    schedule.worker_loop()

    assert sent == [
        mpi_schedule._BatchCompletion(  # noqa: SLF001
            tuple(
                make_completion(
                    task,
                    failure if task is tasks[1] else task.input,
                )
                for task in tasks
            )
        )
    ]
    assert acknowledgements == [None]


def test_fatal_failure_aborts_rest_of_worker_batch() -> None:
    schedule = object.__new__(mpi_schedule.MPIGraphSchedule)
    schedule.rank = 1
    tasks = (make_task(0), make_task(1), make_task(2))
    commands = deque(
        (
            mpi_schedule._EvaluateBatch(tasks, 1),  # noqa: SLF001
            mpi_schedule._Shutdown(1),  # noqa: SLF001
        )
    )
    evaluated: list[Any] = []
    failures: list[BaseException] = []
    acknowledgements: list[None] = []

    def evaluate(task: Any, *, capture_context_state: bool) -> NodeCompletion:
        assert capture_context_state
        evaluated.append(task)
        if task is tasks[1]:
            msg = "fatal"
            raise KeyboardInterrupt(msg)
        return make_completion(task)

    schedule._next_worker_command = commands.popleft  # type: ignore[method-assign]  # noqa: SLF001
    schedule._evaluate_worker_task = evaluate  # type: ignore[method-assign]  # noqa: SLF001
    schedule._report_worker_failure = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda exception, _state: failures.append(exception)
    )
    schedule._acknowledge_shutdown = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: acknowledgements.append(None)
    )

    schedule.worker_loop()

    assert evaluated == list(tasks[:2])
    assert len(failures) == 1
    assert isinstance(failures[0], KeyboardInterrupt)
    assert acknowledgements == [None]


@pytest.mark.parametrize("option", [0, -1, 1.5, "8", True])
@pytest.mark.parametrize(
    "name",
    ["max_tasks_per_message", "target_batches_per_worker"],
)
def test_invalid_batching_options_are_rejected(name: str, option: object) -> None:
    options = {name: option}
    with pytest.raises(ValueError, match="positive integer"):
        mpi_schedule.MPIGraphScheduler(**options)


def test_numerical_results_match_across_batch_sizes() -> None:
    @component()
    def delayed_offset(value: int, *, increment: int, delay: float) -> int:
        sleep(delay)
        return value + increment

    @component()
    def total(values: tuple[int, ...]) -> int:
        return sum(values)

    @component()
    def double(value: int) -> int:
        return 2 * value

    uneven_branches = tuple(
        delayed_offset.bind(increment=index, delay=(index % 4) * 0.001)
        for index in range(17)
    )
    objective = PipelineComponent(join(*uneven_branches), total).then(double)
    results: list[int] = []

    for batch_size in (1, 2, 4, 8):
        with mpi_schedule.MPIGraphScheduler(
            max_tasks_per_message=batch_size,
            target_batches_per_worker=1,
        ).prepare(objective) as schedule:
            if schedule.rank == 0:
                results.append(schedule(4, EvaluateContext()))
            else:
                schedule.worker_loop()

    if MPI.COMM_WORLD.Get_rank() == 0:
        assert results == [408] * 4


@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() < 4, reason="requires four MPI ranks")
def test_large_payload_batches_refill_workers_and_preserve_results() -> None:
    """Exercise live multi-task refills beyond typical MPI eager limits."""

    @component()
    def round_trip(item: tuple[int, bytes]) -> tuple[int, bytes]:
        return item

    comm = LiveRecordingComm(MPI.COMM_WORLD)
    with mpi_schedule.MPIGraphScheduler(
        comm=comm,
        max_tasks_per_message=2,
        target_batches_per_worker=2,
    ).prepare(round_trip) as schedule:
        if schedule.rank != 0:
            schedule.worker_loop()
            return

        payloads = [bytes([index]) * _LARGE_PAYLOAD_SIZE for index in range(12)]
        requests = [
            EvaluationRequest((index, payload), EvaluateContext())
            for index, payload in enumerate(payloads)
        ]
        results = list(schedule.evaluate_many(requests))

    by_index = {result.index: result.value for result in results}
    assert by_index == {
        index: (index, payload) for index, payload in enumerate(payloads)
    }
    assert comm.batch_sizes == [2] * 6
