from __future__ import annotations

from uuid import UUID

import pytest

from chemfit.abstract_objective_function import EvaluateContext, ObjectiveFunctor
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.objective_hooks import TimingHook, UUIDHook


class ConstantObjective(ObjectiveFunctor[dict[str, float]]):
    def _evaluate(
        self,
        parameters: dict[str, float],
        ctx: EvaluateContext,  # noqa: ARG002
    ) -> float:
        return parameters["value"]


class FailingObjective(ObjectiveFunctor[dict[str, float]]):
    def _evaluate(
        self,
        parameters: dict[str, float],  # noqa: ARG002
        ctx: EvaluateContext,  # noqa: ARG002
    ) -> float:
        msg = "evaluation failed"
        raise ValueError(msg)


def test_hook_object_and_direct_callbacks_are_mutually_exclusive():
    objective = ConstantObjective()

    with pytest.raises(TypeError, match="either hook or pre/post callbacks"):
        objective.register_eval_hook(
            hook=UUIDHook(),
            pre=lambda _ctx: None,
        )


def test_direct_pre_and_post_callbacks_are_registered():
    objective = ConstantObjective()
    calls: list[str] = []
    objective.register_eval_hook(
        pre=lambda _ctx: calls.append("pre"),
        post=lambda _ctx: calls.append("post"),
    )

    objective({"value": 3.0})

    assert calls == ["pre", "post"]


def test_post_callback_can_replace_successful_objective_result() -> None:
    objective = ConstantObjective()
    objective.register_eval_hook(post=lambda ctx: setattr(ctx, "loss", 7.0))
    ctx = EvaluateContext()

    result = objective({"value": 1.0}, ctx)

    assert result == 7.0
    assert ctx.loss == 7.0


def test_uuid_hook_assigns_new_id_when_context_is_reused(
    monkeypatch: pytest.MonkeyPatch,
):
    evaluation_ids = iter(
        [
            UUID("6f1c7f52-1a14-4fc8-8d17-047e62bcd4e0"),
            UUID("756f144c-b5cd-43c6-80b2-c266c025fe26"),
        ]
    )
    monkeypatch.setattr(
        "chemfit.objective_hooks.uuid.uuid4", lambda: next(evaluation_ids)
    )

    objective = ConstantObjective()
    objective.register_eval_hook(hook=UUIDHook())
    ctx = EvaluateContext()

    objective({"value": 3.0}, ctx)
    assert ctx.meta["evaluation_id"] == "6f1c7f52-1a14-4fc8-8d17-047e62bcd4e0"

    objective({"value": 4.0}, ctx)
    assert ctx.meta["evaluation_id"] == "756f144c-b5cd-43c6-80b2-c266c025fe26"


def test_timing_hook_records_elapsed_seconds(monkeypatch: pytest.MonkeyPatch):
    ticks = iter([1_000_000_000, 2_250_000_000])
    monkeypatch.setattr(
        "chemfit.objective_hooks.time.perf_counter_ns", lambda: next(ticks)
    )

    objective = ConstantObjective()
    objective.register_eval_hook(hook=TimingHook())
    ctx = EvaluateContext()

    assert objective({"value": 3.0}, ctx) == 3.0
    assert ctx.meta["timing"] == {"elapsed_seconds": 1.25}


def test_timing_hook_records_failed_evaluation(monkeypatch: pytest.MonkeyPatch):
    ticks = iter([4_000_000_000, 4_500_000_000])
    monkeypatch.setattr(
        "chemfit.objective_hooks.time.perf_counter_ns", lambda: next(ticks)
    )

    objective = FailingObjective()
    objective.register_eval_hook(hook=TimingHook(meta_key="runtime"))
    ctx = EvaluateContext()

    with pytest.raises(ValueError, match="evaluation failed"):
        objective({}, ctx)

    assert ctx.meta["runtime"] == {"elapsed_seconds": 0.5}
    assert isinstance(ctx.temp.exception, ValueError)


@pytest.mark.parametrize("recursive", [False, True])
def test_recursive_timing_visits_nested_terms(recursive: bool):
    leaf = ConstantObjective()
    inner = CombinedObjectiveFunction([leaf, ConstantObjective()])
    objective = CombinedObjectiveFunction([inner, ConstantObjective()])
    assert (
        objective.register_eval_hook(hook=TimingHook(), recursive=recursive)
        is objective
    )
    ctx = EvaluateContext()
    assert objective({"value": 3.0}, ctx) == 9.0
    assert ctx.meta["timing"]["elapsed_seconds"] >= 0
    children = ctx.children
    for child in [*children, *children[0].children]:
        assert ("timing" in child.meta) is recursive
    assert len(leaf.pre_eval_hooks) == int(recursive)
    assert len(leaf.post_eval_hooks) == int(recursive)
