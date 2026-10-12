from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Executor, Future, wait

from chemfit.abstract_objective_function import (
    InputT_contra,
    ObjectiveComponent,
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
    objective: ObjectiveComponent[InputT_contra, OutputT_co],
    task: NodeTask[InputT_contra],
) -> NodeCompletion:
    """Evaluate one task and convert ordinary exceptions into node outcomes."""

    try:
        outcome: NodeOutcome = objective(task.input, task.ctx)
    except Exception as exception:
        outcome = exception

    ctx_result_state = task.ctx.to_result_state()
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

        for t in tasks:
            component = self.graph.nodes[t.node_id].objective

            batch_state.task_futures.add(
                self.executor.submit(evaluate_leaf_task, component, t)
            )

    def next_completion(
        self,
        batch_state: BatchState,
    ) -> NodeCompletion:
        """Return a completed task."""

        task_futures: set[Future[NodeCompletion]] = batch_state.task_futures

        done, _ = wait(task_futures, return_when=FIRST_COMPLETED)

        fut = done.pop()
        task_futures.remove(fut)

        return fut.result()

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
        self, objective: ObjectiveComponent[InputT_contra, OutputT_co]
    ) -> ExecutorGraphSchedule:
        """
        Compile an objective component into an executor-backed graph schedule.

        Args:
            objective: Root ordinary or composite objective to schedule.

        Returns:
            Prepared executor-backed schedule for the objective graph.

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
