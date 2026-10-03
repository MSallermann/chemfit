"""Tests for the backend-neutral tree-schedule execution protocol."""

from collections.abc import Iterator, Sequence

import pytest

from chemfit.abstract_objective_function import EvaluateContext, ObjectiveFunctor
from chemfit.callgraph import CallTree, cob_to_call_tree
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.scheduling import EvaluationRequest
from chemfit.tree_schedule import (
    LeafCompletion,
    LeafTask,
    SerialTreeSchedule,
    SerialTreeScheduler,
    TreeScheduleBase,
)

Parameters = dict[str, float]


class RecordingTaskSchedule(TreeScheduleBase[Parameters]):
    """Record neutral tasks and complete them in reverse order."""

    def __init__(self, tree: CallTree) -> None:
        """Initialize the schedule and its task-batch log."""

        super().__init__(tree)
        self.task_batches: list[tuple[LeafTask[Parameters], ...]] = []

    def execute_leaf_tasks(
        self,
        tasks: Sequence[LeafTask[Parameters]],
    ) -> Iterator[LeafCompletion]:
        """Return synthetic worker completions for a task batch."""

        self.task_batches.append(tuple(tasks))
        for task in reversed(tasks):
            value = task.parameters["x"]
            yield LeafCompletion(
                run_id=task.run_id,
                node_id=task.node_id,
                outcome=value,
                ctx_result_state={
                    "parameters": task.parameters,
                    "loss": value,
                    "quantities": {"x": value},
                    "meta": {"completed_by": "recording-backend"},
                },
            )


class CatastrophicLeaf(ObjectiveFunctor[Parameters]):
    """Raise a BaseException that must poison its prepared schedule."""

    def _evaluate(self, parameters: Parameters, ctx: EvaluateContext) -> float:
        """Interrupt leaf evaluation."""

        del parameters, ctx
        msg = "catastrophic leaf failure"
        raise KeyboardInterrupt(msg)


class CleanupFailure(RuntimeError):
    """Failure raised while closing a poisoned test schedule."""


class FailingCloseSchedule(SerialTreeSchedule[Parameters]):
    """Simulate a backend whose cleanup also fails."""

    def close(self) -> None:
        """Fail before delegating to the base close implementation."""

        msg = "cleanup failed"
        raise CleanupFailure(msg)


def make_objective(*, catastrophic: bool = False) -> CombinedObjectiveFunction:
    """Build a two-leaf objective used by the protocol tests."""

    objectives = (
        [CatastrophicLeaf(), lambda parameters: parameters["x"]]
        if catastrophic
        else [lambda parameters: parameters["x"]] * 2
    )
    return CombinedObjectiveFunction(objectives, weights=[2.0, 3.0])


def test_base_expands_tasks_and_restores_completion_contexts() -> None:
    """TreeScheduleBase owns task expansion and result-state restoration."""

    objective = make_objective()
    schedule = RecordingTaskSchedule(cob_to_call_tree(objective))
    contexts = [EvaluateContext(), EvaluateContext()]
    requests = [
        EvaluationRequest(parameters={"x": value}, ctx=ctx)
        for value, ctx in zip((2.0, 4.0), contexts, strict=True)
    ]

    results = sorted(schedule.evaluate_many(requests), key=lambda result: result.index)

    assert [result.value for result in results] == [10.0, 20.0]
    assert len(schedule.task_batches) == 1
    tasks = schedule.task_batches[0]
    assert [task.run_id for task in tasks] == [0, 0, 1, 1]
    assert {task.node_id for task in tasks} == set(schedule.leaf_ids)
    assert [task.parameters["x"] for task in tasks] == [2.0, 2.0, 4.0, 4.0]

    for ctx, value in zip(contexts, (2.0, 4.0), strict=True):
        children = ctx.meta["children"]
        assert [child["loss"] for child in children] == [value, value]
        assert all(
            child["meta"]["completed_by"] == "recording-backend" for child in children
        )


def test_catastrophic_leaf_failure_closes_schedule_and_prevents_reuse() -> None:
    """An escaping BaseException permanently poisons the schedule."""

    schedule = SerialTreeScheduler().prepare(make_objective(catastrophic=True))

    with pytest.raises(KeyboardInterrupt, match="catastrophic leaf failure"):
        schedule.evaluate({"x": 2.0}, EvaluateContext())

    assert schedule.closed
    with pytest.raises(RuntimeError, match="Prepared schedule is closed"):
        schedule.evaluate({"x": 2.0}, EvaluateContext())


def test_cleanup_failure_is_not_allowed_to_replace_catastrophic_failure() -> None:
    """Preserve the original failure and attach failed cleanup as a note."""

    objective = make_objective(catastrophic=True)
    schedule = FailingCloseSchedule(cob_to_call_tree(objective))

    with pytest.raises(KeyboardInterrupt):
        schedule.evaluate({"x": 2.0}, EvaluateContext())

    assert schedule.closed
