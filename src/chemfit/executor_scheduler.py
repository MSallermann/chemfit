"""
Executor-backed tree scheduling for objective-functor evaluations.

ExecutorTreeScheduler prepares an objective call tree and binds it to
either a caller-provided executor or an executor created by a factory. Leaf
objectives are submitted as independent futures, while TreeScheduleBase
coordinates composite activation, completion propagation, and result ordering.
Each composite objective retains its own evaluation semantics.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Executor, Future, as_completed
from typing import Any, Generic, TypeVar, cast

from chemfit.abstract_objective_function import ObjectiveFunctor
from chemfit.callgraph import CallTree, LeafNode, objective_to_call_tree
from chemfit.scheduling import Scheduler
from chemfit.tree_schedule import (
    BatchState,
    LeafCompletion,
    LeafTask,
    TreeScheduleBase,
    evaluate_leaf_task,
)

ParametersT_contra = TypeVar(
    "ParametersT_contra", contravariant=True, bound=Mapping[str, Any]
)


class ExecutorTreeSchedule(
    TreeScheduleBase[ParametersT_contra], Generic[ParametersT_contra]
):
    """
    Evaluate tree leaves concurrently through an Executor.

    Args:
        tree: Compiled ordinary or composite objective call tree.
        executor: Executor used to submit leaf evaluations.
        owns_executor: Whether closing this schedule should shut down the
            executor.

    Attributes:
        executor: Executor used by this prepared schedule.
        owns_executor: Whether the executor was created for this schedule.

    """

    def __init__(self, tree: CallTree, executor: Executor, owns_executor: bool) -> None:
        """Initialize the executor schedule."""

        super().__init__(tree)
        self.executor = executor
        self.owns_executor = owns_executor

    def close(self):
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

    def execute_leaf_tasks(
        self, tasks: Sequence[LeafTask[ParametersT_contra]], batch_state: BatchState
    ) -> Iterator[LeafCompletion]:
        """
        Submit leaf tasks and yield their completions in completion order.

        Args:
            tasks: Backend-neutral pending leaf tasks.

        Yields:
            Backend-neutral leaf completions.

        """
        batch_state.futures = []
        for task in tasks:
            node = self.tree.nodes[task.node_id]
            assert isinstance(node, LeafNode)
            batch_state.futures.append(
                self.executor.submit(
                    evaluate_leaf_task,
                    node.objective,
                    task,
                    capture_context_state=True,
                )
            )

        for future in as_completed(batch_state.futures):
            yield future.result()

    def abort_batch(self, batch_state: BatchState) -> None:
        """Cancel pending futures."""
        futures = cast(
            "Sequence[Future[LeafCompletion]]",
            getattr(batch_state, "futures", ()),
        )
        for future in futures:
            future.cancel()


class ExecutorTreeScheduler(Scheduler[ExecutorTreeSchedule[Any]]):
    """
    Prepare tree schedules backed by a reusable or per-schedule executor.

    Exactly one executor source must be provided. A supplied executor remains
    owned by the caller and can be shared across schedules. An executor factory
    creates a new executor for each prepared schedule, which then owns and
    shuts down that executor.

    Args:
        executor_factory: Zero-argument callable that creates an executor for
            each prepared schedule.
        executor: Existing executor to reuse without taking ownership.

    Raises:
        ValueError: If both executor sources or neither source are provided.

    """

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
        self,
        objective: ObjectiveFunctor[ParametersT_contra],
        /,
    ) -> ExecutorTreeSchedule[ParametersT_contra]:
        """
        Compile an objective functor and bind it to an executor.

        Args:
            objective: Root ordinary or composite objective to schedule.

        Returns:
            Prepared executor-backed tree schedule. It owns the executor only
            when that executor was created by executor_factory.

        """

        if self.executor is not None:
            executor = self.executor
            owns_executor = False
        else:
            assert self.executor_factory is not None
            executor = self.executor_factory()
            owns_executor = True

        return ExecutorTreeSchedule(
            tree=objective_to_call_tree(objective),
            executor=executor,
            owns_executor=owns_executor,
        )
