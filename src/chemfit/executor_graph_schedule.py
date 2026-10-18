from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from queue import SimpleQueue

from chemfit.abstract_objective_function import (
    Component,
    InputT_contra,
    OutputT_co,
)
from chemfit.callgraph import CallGraph, objective_to_call_graph
from chemfit.graph_schedule_base import (
    BatchState,
    GraphScheduleBase,
    NodeCompletion,
    NodeOutcome,
    NodeTask,
)
from chemfit.scheduling import Scheduler


def evaluate_leaf_task(
    objective: Component[InputT_contra, OutputT_co],
    task: NodeTask[InputT_contra],
    capture_context_state: bool = True,
) -> NodeCompletion:
    """Evaluate one task and convert ordinary exceptions into node outcomes."""

    try:
        outcome: NodeOutcome = objective(task.input, task.ctx)
    except Exception as exception:
        outcome = exception

    ctx_result_state = task.ctx.to_result_state() if capture_context_state else None
    return NodeCompletion(
        run_id=task.run_id,
        node_id=task.node_id,
        outcome=outcome,
        ctx_result_state=ctx_result_state,
    )


class ExecutorGraphSchedule(GraphScheduleBase):
    def __init__(self, graph: CallGraph, executor: Executor, owns_executor: bool):
        """Bind a call graph to an executor and its ownership policy."""
        super().__init__(graph)
        self.executor = executor
        self.owns_executor = owns_executor
        # This is an explicit whitelist, not a heuristic based on an executor
        # name or private implementation detail. Other Executor subclasses keep
        # the serialization-safe context transfer path.
        self._shares_context_objects = isinstance(executor, ThreadPoolExecutor)

    def close(self) -> None:
        """
        Close the schedule and shut down an owned executor.

        A caller-provided executor is left open. Repeated calls are safe and do
        not attempt to shut down the executor again.

        """

        if self.closed:
            return

        try:
            if self.owns_executor:
                self.executor.shutdown()
        finally:
            super().close()

    def submit_tasks(
        self,
        tasks: Sequence[NodeTask],
        batch_state: BatchState,
    ) -> None:
        """Submit tasks for execution without waiting for completion."""
        if not hasattr(batch_state, "task_futures"):
            batch_state.task_futures = set()
            batch_state.completed_futures = SimpleQueue()

        for t in tasks:
            component = self.graph.nodes[t.node_id].objective
            future = self.executor.submit(
                evaluate_leaf_task,
                component,
                t,
                not self._shares_context_objects,
            )
            # Register ownership before the callback: add_done_callback() may
            # invoke synchronously when an inexpensive task already finished.
            batch_state.task_futures.add(future)
            # Workers only enqueue their Future. The coordinator can now block
            # on one O(1) queue operation instead of attaching a new waiter to
            # the entire outstanding-future set after every completion.
            future.add_done_callback(batch_state.completed_futures.put)

    def wait_for_completions(
        self,
        batch_state: BatchState,
    ) -> Sequence[NodeCompletion]:
        """
        Block for one completed future and drain others already available.

        Future callbacks perform constant-time queue insertion. This avoids
        repeatedly constructing a waiter over the entire outstanding future
        set after every atomic completion.

        """
        task_futures: set[Future[NodeCompletion]] = batch_state.task_futures
        completed_futures: SimpleQueue[Future[NodeCompletion]] = (
            batch_state.completed_futures
        )
        # Snapshot the outstanding count so a drain is finite even if backend
        # behavior changes to overlap submission with completion processing.
        drain_limit = len(task_futures)
        futures = [completed_futures.get()]

        # This coordinator is the queue's only consumer, so observing a queued
        # item cannot race with another consumer removing it.
        while len(futures) < drain_limit and not completed_futures.empty():
            futures.append(completed_futures.get())

        completions = []
        for future in futures:
            task_futures.remove(future)
            completions.append(future.result())

        return completions

    def next_completion(
        self,
        batch_state: BatchState,
    ) -> NodeCompletion:
        """Return one task completion through the compatibility API."""
        if not hasattr(batch_state, "completion_buffer"):
            batch_state.completion_buffer = deque()

        if not batch_state.completion_buffer:
            completions = self.wait_for_completions(batch_state)
            batch_state.completion_buffer.extend(completions)

        return batch_state.completion_buffer.popleft()

    def abort_batch(self, batch_state: BatchState) -> None:
        """Cancel pending futures."""

        for future in getattr(batch_state, "task_futures", ()):
            future.cancel()


class ExecutorGraphScheduler(Scheduler[ExecutorGraphSchedule]):
    """Prepare graph schedules backed by a caller or factory-owned executor."""

    def __init__(
        self,
        /,
        executor_factory: Callable[[], Executor] | None = None,
        executor: Executor | None = None,
    ) -> None:
        """
        Initialize the executor-backed scheduler configuration.

        Args:
            executor_factory: Zero-argument callable that creates an executor
                for each prepared schedule.
            executor: Existing executor to reuse without taking ownership.

        Raises:
            ValueError: If both arguments or neither argument are provided.

        """

        if (executor is None) == (executor_factory is None):
            msg = "Specify exactly one of executor or executor_factory."
            raise ValueError(msg)

        self.executor = executor
        self.executor_factory = executor_factory
        super().__init__()

    def prepare(
        self, objective: Component[InputT_contra, OutputT_co]
    ) -> ExecutorGraphSchedule:
        """
        Compile a component into an executor-backed graph schedule.

        Args:
            objective: Root atomic or structural component to schedule.

        Returns:
            Prepared executor-backed schedule for the component graph.

        """

        graph = objective_to_call_graph(objective)

        if self.executor is not None:
            executor = self.executor
            owns_executor = False
        else:
            assert self.executor_factory is not None
            executor = self.executor_factory()
            owns_executor = True

        return ExecutorGraphSchedule(graph, executor, owns_executor=owns_executor)
