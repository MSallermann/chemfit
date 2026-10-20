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

MPI = pytest.importorskip("mpi4py.MPI", reason="Missing mpi4py")
mpi_schedule = pytest.importorskip(
    "chemfit.mpi_graph_schedule",
    reason="Missing mpi4py",
)


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

    def Get_rank(self) -> int:  # noqa: N802
        return 0

    def Get_size(self) -> int:  # noqa: N802
        return self.size

    def isend(self, message: object, *, dest: int, tag: int) -> CompletedRequest:
        request = CompletedRequest()
        self.sent.append((message, dest, tag, request))
        return request

    def recv(self, *, source: int, tag: int, status: Any) -> object:
        del source, tag
        result_source, message = self.results.popleft()
        status.Set_source(result_source)
        return message

    def Iprobe(self, *, source: int, tag: int) -> bool:  # noqa: N802
        del source, tag
        return bool(self.results)


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
) -> Any:
    state = BatchState()
    state.pending_tasks = deque(pending)
    state.in_flight = {} if in_flight is None else in_flight
    state.command_sends = {} if command_sends is None else command_sends
    state.idle_workers = deque(idle_workers)
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
            mpi_schedule._EvaluateBatch(tasks),  # noqa: SLF001
            mpi_schedule._Shutdown(),  # noqa: SLF001
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

    def send(result: object, prefetched: object = None) -> bool:
        assert prefetched is not None
        sent.append(result)
        return True

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
            mpi_schedule._EvaluateBatch(tasks),  # noqa: SLF001
            mpi_schedule._Shutdown(),  # noqa: SLF001
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
        lambda exception: failures.append(exception) is None
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
        with mpi_schedule.MPIGraphScheduler(max_tasks_per_message=batch_size).prepare(
            objective
        ) as schedule:
            if schedule.rank == 0:
                results.append(schedule(4, EvaluateContext()))
            else:
                schedule.worker_loop()

    if MPI.COMM_WORLD.Get_rank() == 0:
        assert results == [408] * 4
