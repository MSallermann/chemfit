"""Tests for the backend-neutral tree-schedule execution protocol."""

from collections.abc import Iterator, Sequence

import pytest

from chemfit.abstract_objective_function import EvaluateContext, ObjectiveFunctor
from chemfit.callgraph import CallTree, CombineNode, cob_to_call_tree
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.scheduling import EvaluationRequest, SerialScheduler
from chemfit.tree_schedule import (
    BatchState,
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
        batch_state: BatchState,  # noqa: ARG002
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


class ParameterLeaf(ObjectiveFunctor[Parameters]):
    """Return the parameter selected at construction time."""

    def __init__(self, name: str) -> None:
        """Store the parameter name read during evaluation."""

        super().__init__()
        self.name = name

    def _evaluate(self, parameters: Parameters, ctx: EvaluateContext) -> float:
        """Return one parameter value."""

        del ctx
        return parameters[self.name]


class StructuralComposite(ObjectiveFunctor[Parameters]):
    """Minimal composite used to verify structural protocol scheduling."""

    def __init__(self, failure_phase: str | None = None) -> None:
        """Create two children and optionally fail one composite phase."""

        super().__init__()
        self.children = (ParameterLeaf("x"), ParameterLeaf("y"))
        self.failure_phase = failure_phase

    def child_objectives(self) -> Sequence[ObjectiveFunctor[Parameters]]:
        """Return the immediate children."""

        return self.children

    def begin_composite_evaluation(
        self,
        parameters: Parameters,
        ctx: EvaluateContext,
    ) -> Sequence[EvaluateContext]:
        """Create one context per child or fail setup."""

        del parameters
        if self.failure_phase == "begin":
            msg = "structural composite begin failed"
            raise RuntimeError(msg)
        ctx.meta["began"] = True
        return ctx.spawn_children(len(self.children))

    def finish_composite_evaluation(
        self,
        child_outcomes: Sequence[float | Exception],
        ctx: EvaluateContext,
    ) -> float:
        """Add child results or fail completion."""

        if self.failure_phase == "finish":
            msg = "structural composite finish failed"
            raise RuntimeError(msg)
        assert all(isinstance(outcome, float) for outcome in child_outcomes)
        value = sum(outcome for outcome in child_outcomes if isinstance(outcome, float))
        ctx.loss = value
        ctx.meta["finished"] = True
        return value

    def _evaluate(self, parameters: Parameters, ctx: EvaluateContext) -> float:
        """Reject direct leaf-style evaluation of this composite."""

        del parameters, ctx
        msg = "composite must not execute as a leaf"
        raise AssertionError(msg)


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


def test_structural_composite_is_scheduled_as_a_combine_node() -> None:
    """Recognize and evaluate a composite without a concrete COB type check."""

    objective = StructuralComposite()
    schedule = SerialTreeScheduler().prepare(objective)
    ctx = EvaluateContext()

    assert isinstance(schedule.tree.nodes[schedule.tree.root], CombineNode)
    assert schedule.evaluate({"x": 2.0, "y": 3.0}, ctx) == 5.0
    assert ctx.meta == {"began": True, "finished": True}


def test_post_hook_loss_changes_propagate_through_tree_schedule() -> None:
    """Propagate final leaf and composite losses after successful post hooks."""

    first = ParameterLeaf("x")
    first.register_eval_hook(post=lambda ctx: setattr(ctx, "loss", 5.0))
    objective = CombinedObjectiveFunction([first, ParameterLeaf("y")])

    def replace_root_loss(ctx: EvaluateContext) -> None:
        ctx.meta["loss_before_root_post"] = ctx.loss
        ctx.loss = 11.0

    objective.register_eval_hook(post=replace_root_loss)
    ctx = EvaluateContext()

    with SerialTreeScheduler().prepare(objective) as schedule:
        result = schedule.evaluate({"x": 1.0, "y": 2.0}, ctx)

    assert ctx.meta["loss_before_root_post"] == 7.0
    assert result == 11.0
    assert ctx.loss == 11.0


@pytest.mark.parametrize("failure_phase", ["begin", "finish"])
def test_structural_composite_phase_failure_is_an_evaluation_outcome(
    failure_phase: str,
) -> None:
    """Return ordinary composite phase failures without poisoning a schedule."""

    schedule = SerialTreeScheduler().prepare(StructuralComposite(failure_phase))

    with pytest.raises(RuntimeError, match=f"{failure_phase} failed"):
        schedule.evaluate({"x": 2.0, "y": 3.0}, EvaluateContext())

    assert not schedule.closed


def test_catastrophic_leaf_failure_closes_schedule_and_prevents_reuse() -> None:
    """An escaping BaseException permanently poisons the schedule."""

    schedule = SerialTreeScheduler().prepare(make_objective(catastrophic=True))

    with pytest.raises(KeyboardInterrupt, match="catastrophic leaf failure"):
        schedule.evaluate({"x": 2.0}, EvaluateContext())

    assert schedule.closed
    with pytest.raises(RuntimeError, match="Prepared schedule is closed"):
        schedule.evaluate({"x": 2.0}, EvaluateContext())


def test_serial_scheduler_closes_after_catastrophic_failure() -> None:
    """The simple serial schedule has the same catastrophic semantics."""

    schedule = SerialScheduler().prepare(CatastrophicLeaf())

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
