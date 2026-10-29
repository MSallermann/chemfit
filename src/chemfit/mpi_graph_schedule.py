"""
MPI backend for ChemFit's graph scheduler.

Rank 0 owns graph traversal and structural-component lifecycles. Other ranks
run a persistent worker loop and evaluate only AtomicNode tasks. Tasks are
queued on rank 0 and dispatched to idle workers in adaptive batches, allowing
new work to become available as earlier graph nodes finish without requiring a
message round trip for every task in broad graphs.

Ordinary component exceptions are returned in NodeCompletion. MPI/protocol
failures are fatal to the prepared schedule and trigger worker shutdown.

All ranks must prepare the same component graph; only rank 0 evaluates it.
Nonzero ranks call ``schedule.worker_loop()`` and rank 0 calls ``schedule.close()``
after the final evaluation to release the workers.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

try:
    from mpi4py import MPI
except ImportError as e:
    if hasattr(e, "add_note"):
        e.add_note(
            "Install ChemFit's MPI dependencies with `pip install chemfit[mpi]`."
        )
    raise

from chemfit.abstract_objective_function import (
    Component,
    InputT_contra,
    OutputT_co,
)
from chemfit.callgraph import AtomicNode, CallGraph, objective_to_call_graph
from chemfit.debug_utils import log_all_methods
from chemfit.graph_schedule_base import (
    BatchState,
    GraphScheduleBase,
    NodeCompletion,
    NodeOutcome,
    NodeTask,
)
from chemfit.scheduling import Scheduler

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger(__name__)

_COMMAND_TAG = 41001
_RESULT_TAG = 41002
_CONTROL_TAG = 41003


@dataclass(frozen=True, slots=True)
class _EvaluateBatch:
    """
    Carry one coordinator command to a worker.

    Attributes:
        tasks:
            Atomic graph tasks owned by the destination worker until it
            returns the matching completion batch.
        sequence:
            Per-worker command number. It increases by one for every dispatch
            and lets shutdown name the final command a worker must consume.

    """

    tasks: tuple[NodeTask, ...]
    sequence: int


@dataclass(frozen=True, slots=True)
class _BatchCompletion:
    """
    Return all outcomes for one evaluation command.

    Attributes:
        completions:
            One completion for each task in the corresponding
            :class:`_EvaluateBatch`; the coordinator validates their task
            identities before releasing the worker.

    """

    completions: tuple[NodeCompletion, ...]


@dataclass(frozen=True, slots=True)
class _Shutdown:
    """
    Tell a worker to stop after consuming every dispatched command.

    Attributes:
        last_command_sequence:
            Per-worker command fence captured by the coordinator. A worker
            must consume or retire every command through this number before
            acknowledging shutdown.

    """

    last_command_sequence: int


@dataclass(frozen=True, slots=True)
class _ShutdownAcknowledgement:
    """Confirm that a worker has stopped receiving and evaluating tasks."""


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    """
    Describe a fatal worker or protocol failure without serializing it.

    Attributes:
        rank:
            Rank reporting the failure. The coordinator verifies that it
            matches the MPI message source.
        exception_type:
            Qualified name of the original exception type.
        message:
            Best-effort string form of the original exception.

    """

    rank: int
    exception_type: str
    message: str


@dataclass(slots=True)
class _WorkerProtocolState:
    """
    Track command consumption and the single worker shutdown transition.

    Attributes:
        prefetched:
            A received evaluation batch waiting to run. Protocol progress may
            receive at most one batch while a result send is pending.
        last_command_sequence:
            Sequence number of the most recently consumed evaluation command,
            whether that command was evaluated, prefetched, or retired.
        shutdown_sequence:
            Final command sequence announced by the coordinator at shutdown.
            The worker cannot acknowledge shutdown until it has consumed every
            command through this sequence. ``None`` means shutdown has not
            started.
        failed:
            Whether a fatal task or protocol error has poisoned this worker.
            A failed worker consumes later commands but does not evaluate them.
        failure_reported:
            Whether the worker has already attempted its single failure report.
        shutdown_acknowledged:
            Whether the worker has sent its single shutdown acknowledgement.

    """

    prefetched: deque[_EvaluateBatch] = field(default_factory=deque)
    last_command_sequence: int = 0
    shutdown_sequence: int | None = None
    failed: bool = False
    failure_reported: bool = False
    shutdown_acknowledged: bool = False


class MPIWorkerError(RuntimeError):
    """Fatal MPI worker or scheduling protocol failure."""


def _validate_batching_options(
    *,
    max_tasks_per_message: int,
    target_batches_per_worker: int,
) -> None:
    """Reject batching settings that cannot produce a nonempty batch."""
    if type(max_tasks_per_message) is not int or max_tasks_per_message < 1:
        msg = "max_tasks_per_message must be a positive integer"
        raise ValueError(msg)
    if type(target_batches_per_worker) is not int or target_batches_per_worker < 1:
        msg = "target_batches_per_worker must be a positive integer"
        raise ValueError(msg)


def _describe_worker_failure(rank: int, exception: BaseException) -> _WorkerFailure:
    try:
        message = str(exception)
    except BaseException:
        message = "<exception message could not be formatted>"

    return _WorkerFailure(
        rank=rank,
        exception_type=type(exception).__qualname__,
        message=message,
    )


class MPIGraphSchedule(GraphScheduleBase):
    """
    Prepared graph schedule using persistent MPI workers.

    The communicator must be dedicated to one live ``MPIGraphSchedule``. The
    protocol uses fixed tags and deliberately does not multiplex independent
    schedules on one communicator.

    Attributes:
        comm:
            Communicator used exclusively by this schedule's protocol.
        rank:
            This process's rank in ``comm``. Rank zero owns graph traversal.
        size:
            Number of ranks in ``comm``, including the coordinator.
        max_tasks_per_message:
            Hard upper bound on tasks placed in one evaluation command.
        target_batches_per_worker:
            Number of command batches the adaptive chunker aims to preserve
            per worker when enough work is available.
        _workers_released:
            Whether every worker has acknowledged shutdown. Once true, no
            worker can still evaluate or wait for protocol messages.
        _evaluation_active:
            Reentrancy guard set while one ``evaluate_many()`` iterator is
            active on the coordinator.
        _poisoned:
            Whether an aborted evaluation made this schedule permanently
            unusable for further evaluation.
        _last_command_sequence:
            Latest sequence dispatched to each worker rank. Shutdown snapshots
            these values as the command-consumption fences.

    """

    def __init__(
        self,
        graph: CallGraph,
        comm: Any,
        *,
        max_tasks_per_message: int = 8,
        target_batches_per_worker: int = 8,
        mpi_debug_log: bool = False,
    ) -> None:
        """Bind a call graph to an MPI communicator."""
        super().__init__(graph)
        _validate_batching_options(
            max_tasks_per_message=max_tasks_per_message,
            target_batches_per_worker=target_batches_per_worker,
        )
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.max_tasks_per_message = max_tasks_per_message
        self.target_batches_per_worker = target_batches_per_worker
        self._workers_released = False
        self._evaluation_active = False
        self._poisoned = False
        self._last_command_sequence = dict.fromkeys(range(1, self.size), 0)

        if mpi_debug_log:
            self.comm = log_all_methods(
                self.comm,
                log_func=self._log_func,
                log_args=True,
                log_res=True,
            )

    def _log_func(self, message: str) -> None:
        logger.warning("[Rank %s] %s", self.rank, message)

    def _ensure_coordinator(self) -> None:
        """Raises unless we are on rank 0."""

        if self.rank != 0:
            msg = "Graph evaluation is only supported on MPI rank 0"
            raise RuntimeError(msg)
        if self.closed:
            msg = "Prepared MPI graph schedule is closed"
            raise RuntimeError(msg)
        if self._poisoned:
            msg = "Prepared MPI graph schedule is poisoned"
            raise RuntimeError(msg)

    def _ensure_batch_state(self, batch_state: BatchState) -> None:
        """Construct default fields on batch_state if needed and assert their properties."""
        if not hasattr(batch_state, "pending_tasks"):
            # Tasks ready for dispatch but not yet owned by a worker.
            batch_state.pending_tasks = deque()
            # A ready-work cohort keeps one bulk dispatch quantum, then chooses
            # one smaller quantum for its final worker wave. Recomputing after
            # every refill would create progressively smaller batches and
            # increase the result-message rate during a completion burst.
            # None means there is no current ready-work cohort.
            batch_state.dispatch_batch_size = None
            # True after the current cohort has made its single tail-size
            # adjustment; it prevents repeated shrinking on later refills.
            batch_state.dispatch_tail_started = False
            # Worker rank -> exact tasks assigned by its outstanding command.
            # These tasks stay here until their validated result is received.
            batch_state.in_flight = {}
            # Keep nonblocking command sends alive until the corresponding
            # worker result proves that the command was received.  A blocking
            # send here can deadlock with a worker's rendezvous result send
            # when both task and result exceed the MPI eager limit.
            # Its keys must always equal the keys of in_flight.
            batch_state.command_sends = {}
            # Rank availability changes only on dispatch and completion, so an
            # explicit queue avoids scanning every rank on every refill. It is
            # disjoint from in_flight, and together they contain every worker.
            batch_state.idle_workers = deque(range(1, self.size))
        self._assert_batch_state(batch_state)

    def _assert_batch_state(self, batch_state: BatchState) -> None:
        """Check the one-command-per-worker ownership invariant."""
        idle_workers = tuple(batch_state.idle_workers)
        idle = set(idle_workers)
        in_flight = set(batch_state.in_flight)
        command_sends = set(batch_state.command_sends)
        workers = set(range(1, self.size))

        assert len(idle) == len(idle_workers), "idle worker ranks must be unique"
        assert idle.isdisjoint(in_flight), "workers cannot be idle and in flight"
        assert idle | in_flight == workers, "every worker must have one owner state"
        assert command_sends == in_flight, "command sends must match in-flight work"

        task_ids = [(task.run_id, task.node_id) for task in batch_state.pending_tasks]
        task_ids.extend(
            (task.run_id, task.node_id)
            for tasks in batch_state.in_flight.values()
            for task in tasks
        )
        assert len(task_ids) == len(set(task_ids)), "each task must have one owner"

    def _evaluate_worker_task(
        self, task: NodeTask, *, capture_context_state: bool
    ) -> NodeCompletion:
        """Evaluate an AtomicNode task and return a NodeCompletion."""

        node = self.graph.nodes[task.node_id]
        if not isinstance(node, AtomicNode):
            msg = f"MPI worker received non-atomic node {task.node_id}"
            raise MPIWorkerError(msg)

        try:
            outcome: NodeOutcome = node.objective(task.input, task.ctx)
        except Exception as exception:
            outcome = exception

        context_state = task.ctx.to_result_state() if capture_context_state else None
        return NodeCompletion(
            run_id=task.run_id,
            node_id=task.node_id,
            outcome=outcome,
            ctx_result_state=context_state,
        )

    def _dispatch_available(self, batch_state: BatchState) -> None:
        """Assign one adaptive batch to each idle worker while work remains."""
        self._assert_batch_state(batch_state)
        queue: deque[NodeTask] = batch_state.pending_tasks
        in_flight: dict[int, tuple[NodeTask, ...]] = batch_state.in_flight
        command_sends: dict[int, Any] = batch_state.command_sends
        idle_workers: deque[int] = batch_state.idle_workers

        if not queue:
            batch_state.dispatch_batch_size = None
            batch_state.dispatch_tail_started = False
            return

        chunk_size: int | None = getattr(
            batch_state,
            "dispatch_batch_size",
            None,
        )
        if chunk_size is None:
            chunk_size = self._chunk_size(len(queue))
            batch_state.dispatch_batch_size = chunk_size

        # Keep the cohort quantum stable while there is more than one full
        # worker wave queued. For the tail, spread the remaining tasks across
        # the workers that are idle or still in flight instead of assigning
        # full batches to only a subset and leaving the rest unused.
        worker_slots = len(idle_workers) + len(in_flight)
        tail_started = getattr(batch_state, "dispatch_tail_started", False)
        if (
            not tail_started
            and worker_slots
            and len(queue) <= chunk_size * worker_slots
        ):
            chunk_size = (len(queue) + worker_slots - 1) // worker_slots
            batch_state.dispatch_batch_size = chunk_size
            batch_state.dispatch_tail_started = True

        while queue and idle_workers:
            rank = idle_workers.popleft()
            tasks = tuple(queue.popleft() for _ in range(min(chunk_size, len(queue))))
            sequence = self._last_command_sequence[rank] + 1
            command_send = self.comm.isend(
                _EvaluateBatch(tasks, sequence), dest=rank, tag=_COMMAND_TAG
            )
            self._last_command_sequence[rank] = sequence
            command_sends[rank] = command_send
            in_flight[rank] = tasks

        if not queue:
            batch_state.dispatch_batch_size = None
            batch_state.dispatch_tail_started = False
        self._assert_batch_state(batch_state)

    def _chunk_size(self, queued_tasks: int) -> int:
        """Choose a message size that preserves several batches per worker."""
        worker_count = max(1, self.size - 1)
        target_batch_count = worker_count * self.target_batches_per_worker
        adaptive_size = (queued_tasks + target_batch_count - 1) // target_batch_count
        return min(self.max_tasks_per_message, max(1, adaptive_size))

    def submit_tasks(
        self,
        tasks: Sequence[NodeTask],
        batch_state: BatchState,
    ) -> None:
        """Queue tasks and immediately give available workers more work."""
        self._ensure_coordinator()
        self._ensure_batch_state(batch_state)
        queue_was_empty = not batch_state.pending_tasks
        batch_state.pending_tasks.extend(tasks)

        if queue_was_empty and batch_state.pending_tasks:
            batch_state.dispatch_batch_size = self._chunk_size(
                len(batch_state.pending_tasks)
            )
            batch_state.dispatch_tail_started = False

        if self.size > 1:
            self._dispatch_available(batch_state)
        else:
            self._assert_batch_state(batch_state)

    def _receive_worker_completion(  # noqa: PLR0915
        self, batch_state: BatchState
    ) -> tuple[NodeCompletion, ...]:
        """Receive and validate one worker batch, then reuse or idle its rank."""
        self._assert_batch_state(batch_state)
        in_flight: dict[int, tuple[NodeTask, ...]] = batch_state.in_flight
        command_sends: dict[int, Any] = batch_state.command_sends
        status = MPI.Status()
        message = self.comm.recv(
            source=MPI.ANY_SOURCE,
            tag=_RESULT_TAG,
            status=status,
        )
        source = status.Get_source()

        if isinstance(message, _WorkerFailure):
            if message.rank != source:
                msg = (
                    f"Worker failure rank {message.rank} does not match "
                    f"message source {source}"
                )
                raise MPIWorkerError(msg)
            msg = (
                f"Worker rank {message.rank} failed with "
                f"{message.exception_type}: {message.message}"
            )
            raise MPIWorkerError(msg)

        expected_tasks = in_flight.get(source)
        if expected_tasks is None:
            msg = f"Unexpected completion from idle worker {source}"
            raise MPIWorkerError(msg)
        if not isinstance(message, _BatchCompletion):
            msg = f"Unexpected MPI worker result: {message!r}"
            raise MPIWorkerError(msg)
        if not all(
            isinstance(completion, NodeCompletion) for completion in message.completions
        ):
            msg = f"Worker {source} returned a malformed completion batch"
            raise MPIWorkerError(msg)

        expected = Counter((task.run_id, task.node_id) for task in expected_tasks)
        returned = Counter(
            (completion.run_id, completion.node_id)
            for completion in message.completions
        )
        duplicates = [identity for identity, count in returned.items() if count > 1]
        if duplicates:
            msg = (
                f"Worker {source} returned duplicate completion for task "
                f"{duplicates[0]}"
            )
            raise MPIWorkerError(msg)
        unexpected = list(returned.keys() - expected.keys())
        if unexpected:
            msg = (
                f"Worker {source} returned completion for unexpected task "
                f"{unexpected[0]}"
            )
            raise MPIWorkerError(msg)
        missing = list(expected.keys() - returned.keys())
        if missing:
            msg = f"Worker {source} omitted completion for task {missing[0]}"
            raise MPIWorkerError(msg)
        if returned != expected:
            msg = f"Worker {source} returned an invalid completion batch"
            raise MPIWorkerError(msg)

        # The matching result proves that the worker fully received this
        # command, so completing the retained send cannot wait on remote
        # progress. Do this before replacing the per-rank request.
        command_send = command_sends.get(source)
        if command_send is None:
            msg = f"Missing command send for worker {source}"
            raise MPIWorkerError(msg)
        command_send.Wait()
        del command_sends[source]
        del in_flight[source]

        queue: deque[NodeTask] = batch_state.pending_tasks
        if queue:
            # Refill the completed worker before receiving another result. At
            # large rank counts, deferring all refills until the completion
            # burst has been drained leaves every worker in that burst idle
            # for the coordinator's aggregate receive/validation time.
            batch_state.idle_workers.appendleft(source)
            self._dispatch_available(batch_state)
        else:
            batch_state.idle_workers.append(source)

        self._assert_batch_state(batch_state)
        return message.completions

    def wait_for_completions(
        self,
        batch_state: BatchState,
    ) -> Sequence[NodeCompletion]:
        """Block for one MPI result and drain other results already available."""
        self._ensure_coordinator()
        self._ensure_batch_state(batch_state)

        if self.size == 1:
            if not batch_state.pending_tasks:
                msg = "No queued MPI tasks to complete"
                raise MPIWorkerError(msg)
            task = batch_state.pending_tasks.popleft()
            if not batch_state.pending_tasks:
                batch_state.dispatch_batch_size = None
                batch_state.dispatch_tail_started = False
            completion = self._evaluate_worker_task(task, capture_context_state=False)
            self._assert_batch_state(batch_state)
            return (completion,)

        in_flight: dict[int, tuple[NodeTask, ...]] = batch_state.in_flight
        if not in_flight:
            msg = "No MPI tasks are in flight"
            raise MPIWorkerError(msg)

        # Bound draining by the number in flight at entry. Refilled work can
        # finish quickly, but consuming those new results in this call could
        # otherwise keep the coordinator in an unbounded receive loop and
        # delay graph dependency release.
        drain_limit = len(in_flight)
        completions = list(self._receive_worker_completion(batch_state))
        received_batches = 1
        while received_batches < drain_limit and self.comm.Iprobe(
            source=MPI.ANY_SOURCE,
            tag=_RESULT_TAG,
        ):
            completions.extend(self._receive_worker_completion(batch_state))
            received_batches += 1

        return completions

    def next_completion(self, batch_state: BatchState) -> NodeCompletion:
        """Return one task completion through the compatibility API."""
        if not hasattr(batch_state, "completion_buffer"):
            # A worker command may return several completions at once. Keep
            # those not yet exposed through this one-at-a-time compatibility
            # method in arrival order.
            batch_state.completion_buffer = deque()

        if not batch_state.completion_buffer:
            completions = self.wait_for_completions(batch_state)
            batch_state.completion_buffer.extend(completions)

        return batch_state.completion_buffer.popleft()

    def evaluate_many(self, requests: Sequence[Any]) -> Iterator[Any]:
        """Evaluate one batch, rejecting overlapping use of this schedule."""
        self._ensure_coordinator()
        if self._evaluation_active:
            msg = "Concurrent evaluation on one MPI schedule is not supported"
            raise RuntimeError(msg)

        self._evaluation_active = True
        try:
            yield from super().evaluate_many(requests)
        finally:
            self._evaluation_active = False

    def _send_worker_result(
        self,
        result: _BatchCompletion | _WorkerFailure,
        state: _WorkerProtocolState,
    ) -> None:
        """
        Send a result while progressing command receipt and shutdown control.

        MPI result sends must always finish. Cancelling a send that may already
        be matched by rank 0 is fragile across MPI transports, and Open MPI's
        OFI transport can crash in that path. During shutdown rank 0 continues
        receiving result messages until every worker acknowledges the request.
        """
        request = self.comm.isend(result, dest=0, tag=_RESULT_TAG)
        protocol_error: MPIWorkerError | None = None
        while not request.Test():
            if self.comm.Iprobe(source=0, tag=_CONTROL_TAG):
                command = self.comm.recv(source=0, tag=_CONTROL_TAG)
                try:
                    self._record_worker_shutdown(state, command)
                except MPIWorkerError as exception:
                    protocol_error = protocol_error or exception

            if (
                (
                    state.shutdown_sequence is not None
                    and state.last_command_sequence < state.shutdown_sequence
                )
                or (
                    state.shutdown_sequence is None
                    and protocol_error is None
                    and not state.failed
                    and not state.prefetched
                )
            ) and self.comm.Iprobe(source=0, tag=_COMMAND_TAG):
                command = self.comm.recv(source=0, tag=_COMMAND_TAG)
                try:
                    self._record_worker_batch(
                        state,
                        command,
                        prefetch=state.shutdown_sequence is None,
                    )
                except MPIWorkerError as exception:
                    protocol_error = protocol_error or exception

        if protocol_error is not None:
            raise protocol_error

    def _record_worker_batch(
        self,
        state: _WorkerProtocolState,
        command: object,
        *,
        prefetch: bool,
    ) -> _EvaluateBatch:
        """Validate one command sequence and optionally retain it for execution."""
        if not isinstance(command, _EvaluateBatch):
            msg = f"Worker {self.rank} received unexpected task command: {command!r}"
            raise MPIWorkerError(msg)

        expected = state.last_command_sequence + 1
        if command.sequence != expected:
            msg = (
                f"Worker {self.rank} received command sequence {command.sequence}; "
                f"expected {expected}"
            )
            raise MPIWorkerError(msg)

        state.last_command_sequence = command.sequence
        if prefetch and not state.failed:
            state.prefetched.append(command)
        return command

    def _record_worker_shutdown(
        self,
        state: _WorkerProtocolState,
        command: object,
    ) -> None:
        """Enter shutdown while retaining the command-consumption fence."""
        if not isinstance(command, _Shutdown):
            msg = f"Worker {self.rank} received unknown control: {command!r}"
            raise MPIWorkerError(msg)
        if state.shutdown_sequence is not None:
            msg = f"Worker {self.rank} received duplicate shutdown control"
            raise MPIWorkerError(msg)

        state.shutdown_sequence = command.last_command_sequence
        if state.shutdown_sequence < state.last_command_sequence:
            msg = (
                f"Worker {self.rank} received stale shutdown fence "
                f"{state.shutdown_sequence} after command "
                f"{state.last_command_sequence}"
            )
            state.shutdown_sequence = state.last_command_sequence
            raise MPIWorkerError(msg)

    def _retire_commands_before_shutdown(self, state: _WorkerProtocolState) -> None:
        """Consume every command covered by shutdown without evaluating it."""
        assert state.shutdown_sequence is not None
        state.prefetched.clear()
        while state.last_command_sequence < state.shutdown_sequence:
            command = self.comm.recv(source=0, tag=_COMMAND_TAG)
            self._record_worker_batch(state, command, prefetch=False)

    def _finish_worker_shutdown(self, state: _WorkerProtocolState) -> None:
        """Send the worker's sole shutdown acknowledgement."""
        if state.shutdown_acknowledged:
            return
        state.shutdown_acknowledged = True
        self._acknowledge_shutdown()

    def _acknowledge_shutdown(self) -> None:
        """Tell rank 0 it may finish closing after all result sends complete."""
        self.comm.send(_ShutdownAcknowledgement(), dest=0, tag=_CONTROL_TAG)

    def _report_worker_failure(
        self,
        exception: BaseException,
        state: _WorkerProtocolState,
    ) -> None:
        """
        Report a fatal worker failure and keep serving shutdown control.

        This is intentionally used for every ``BaseException`` that escapes
        worker protocol progress, not only exceptions raised by user tasks.
        In particular, a signal delivered while a worker waits for a command
        must not make it disappear before rank 0 can complete shutdown.
        """
        if state.failure_reported:
            return
        state.failure_reported = True
        self._send_worker_result(_describe_worker_failure(self.rank, exception), state)

    def _next_worker_command(self) -> object:
        """Receive either an evaluation request or shutdown control."""
        status = MPI.Status()
        command = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag not in {_COMMAND_TAG, _CONTROL_TAG}:
            msg = f"Worker {self.rank} received message with unknown tag"
            raise MPIWorkerError(msg)
        if tag == _COMMAND_TAG and not isinstance(command, _EvaluateBatch):
            msg = f"Worker {self.rank} received unknown task command: {command!r}"
            raise MPIWorkerError(msg)
        if tag == _CONTROL_TAG and not isinstance(command, _Shutdown):
            msg = f"Worker {self.rank} received unknown control: {command!r}"
            raise MPIWorkerError(msg)
        return command

    def worker_loop(self) -> None:
        """Process tasks on a nonzero MPI rank until shutdown."""
        if self.rank == 0:
            msg = "worker_loop() cannot be called on MPI rank 0"
            raise RuntimeError(msg)

        state = _WorkerProtocolState()
        while state.shutdown_sequence is None:
            try:
                if state.prefetched:
                    command = state.prefetched.popleft()
                else:
                    command = self._next_worker_command()
                    if isinstance(command, _Shutdown):
                        self._record_worker_shutdown(state, command)
                        continue
                    command = self._record_worker_batch(
                        state,
                        command,
                        prefetch=False,
                    )

                if state.failed:
                    continue

                result = _BatchCompletion(
                    tuple(
                        self._evaluate_worker_task(
                            task,
                            capture_context_state=True,
                        )
                        for task in command.tasks
                    )
                )
                self._send_worker_result(result, state)
            except BaseException as exception:
                state.failed = True
                try:
                    self._report_worker_failure(exception, state)
                except BaseException as report_exception:
                    # Do not recursively report a failure of the failure path.
                    # The worker still honors any shutdown already consumed.
                    logger.error(
                        "Worker %s could not report its protocol failure",
                        self.rank,
                        exc_info=(
                            type(report_exception),
                            report_exception,
                            report_exception.__traceback__,
                        ),
                    )

        self._retire_commands_before_shutdown(state)
        self._finish_worker_shutdown(state)

    def release_workers(self) -> None:
        """Request shutdown and wait until every worker has stopped safely."""
        if self.rank != 0 or self.size <= 1 or self._workers_released:
            return

        requests = [
            self.comm.isend(
                _Shutdown(self._last_command_sequence[rank]),
                dest=rank,
                tag=_CONTROL_TAG,
            )
            for rank in range(1, self.size)
        ]

        # A worker may see shutdown while its previous result send is still
        # active. Keep draining results until it confirms shutdown so that send
        # can complete without MPI_Cancel. The acknowledgment also means close
        # does not return while a worker can still evaluate a prefetched task.
        waiting_for_ack = set(range(1, self.size))
        shutdown_error: MPIWorkerError | None = None
        while waiting_for_ack:
            status = MPI.Status()
            message = self.comm.recv(
                source=MPI.ANY_SOURCE,
                tag=MPI.ANY_TAG,
                status=status,
            )
            source = status.Get_source()
            tag = status.Get_tag()
            if tag == _RESULT_TAG:
                valid_result = (
                    isinstance(message, _BatchCompletion)
                    and all(
                        isinstance(completion, NodeCompletion)
                        for completion in message.completions
                    )
                ) or (isinstance(message, _WorkerFailure) and message.rank == source)
                if source not in waiting_for_ack or not valid_result:
                    msg = (
                        f"Unexpected result from worker {source} while shutting "
                        f"down: {message!r}"
                    )
                    shutdown_error = shutdown_error or MPIWorkerError(msg)
                continue
            if tag != _CONTROL_TAG or not isinstance(message, _ShutdownAcknowledgement):
                msg = f"Unexpected MPI message while shutting down: {message!r}"
                shutdown_error = shutdown_error or MPIWorkerError(msg)
                continue
            if source not in waiting_for_ack:
                msg = f"Unexpected shutdown acknowledgement from worker {source}"
                shutdown_error = shutdown_error or MPIWorkerError(msg)
                continue
            waiting_for_ack.remove(source)

        self._workers_released = True
        if requests:
            MPI.Request.Waitall(requests)
        if shutdown_error is not None:
            raise shutdown_error

    def abort_batch(self, batch_state: BatchState) -> None:
        """Discard queued work and poison the schedule before common cleanup."""
        self._poisoned = True
        if hasattr(batch_state, "pending_tasks"):
            batch_state.pending_tasks.clear()
        # Release coordinator send buffers before shutdown. Completion is not
        # treated as proof of delivery: the per-worker shutdown sequence fence
        # separately requires every dispatched command to be consumed.
        command_sends = getattr(batch_state, "command_sends", {})
        if command_sends:
            MPI.Request.Waitall(list(command_sends.values()))
            command_sends.clear()

    def close(self) -> None:
        """Release workers and permanently close the prepared schedule."""
        if self.closed:
            return
        try:
            self.release_workers()
        except BaseException:
            if self._workers_released:
                super().close()
            raise
        else:
            super().close()


class MPIGraphScheduler(Scheduler[MPIGraphSchedule]):
    """Prepare MPI-backed graph schedules."""

    def __init__(
        self,
        comm: Any | None = None,
        *,
        max_tasks_per_message: int = 8,
        target_batches_per_worker: int = 8,
        mpi_debug_log: bool = False,
    ) -> None:
        """Configure the communicator, batching, and optional protocol logging."""
        super().__init__()
        _validate_batching_options(
            max_tasks_per_message=max_tasks_per_message,
            target_batches_per_worker=target_batches_per_worker,
        )
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.max_tasks_per_message = max_tasks_per_message
        self.target_batches_per_worker = target_batches_per_worker
        self.mpi_debug_log = mpi_debug_log

    def prepare(
        self,
        objective: Component[InputT_contra, OutputT_co],
        /,
    ) -> MPIGraphSchedule:
        """Compile and bind the component graph to the MPI communicator."""
        return MPIGraphSchedule(
            graph=objective_to_call_graph(objective),
            comm=self.comm,
            max_tasks_per_message=self.max_tasks_per_message,
            target_batches_per_worker=self.target_batches_per_worker,
            mpi_debug_log=self.mpi_debug_log,
        )
