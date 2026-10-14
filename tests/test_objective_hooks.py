from uuid import UUID

import pytest

from chemfit.abstract_objective_function import PipelineComponent, component
from chemfit.evaluate_context import EvaluateContext
from chemfit.objective_hooks import TimingHook, UUIDHook


@component()
def identity(value):
    return value


def test_uuid_hook_replaces_the_id_when_a_context_is_reused(monkeypatch):
    evaluation_ids = iter(
        (
            UUID("6f1c7f52-1a14-4fc8-8d17-047e62bcd4e0"),
            UUID("756f144c-b5cd-43c6-80b2-c266c025fe26"),
        )
    )
    monkeypatch.setattr(
        "chemfit.objective_hooks.uuid.uuid4", lambda: next(evaluation_ids)
    )
    objective = identity.copy().register_eval_hook(hook=UUIDHook())
    ctx = EvaluateContext()

    objective(3, ctx)
    first = ctx.meta["evaluation_id"]
    objective(4, ctx)

    assert first == "6f1c7f52-1a14-4fc8-8d17-047e62bcd4e0"
    assert ctx.meta["evaluation_id"] == "756f144c-b5cd-43c6-80b2-c266c025fe26"


def test_timing_hook_records_success_and_failure(monkeypatch):
    ticks = iter((1_000_000_000, 2_250_000_000, 4_000_000_000, 4_500_000_000))
    monkeypatch.setattr(
        "chemfit.objective_hooks.time.perf_counter_ns", lambda: next(ticks)
    )
    successful = identity.copy().register_eval_hook(hook=TimingHook())

    @component()
    def failing(_value):
        msg = "evaluation failed"
        raise ValueError(msg)

    failing.register_eval_hook(hook=TimingHook(meta_key="runtime"))
    success_ctx = EvaluateContext()
    failure_ctx = EvaluateContext()

    assert successful(3, success_ctx) == 3
    with pytest.raises(ValueError, match="evaluation failed"):
        failing(3, failure_ctx)

    assert success_ctx.meta["timing"] == {"elapsed_seconds": 1.25}
    assert failure_ctx.meta["runtime"] == {"elapsed_seconds": 0.5}
    assert isinstance(failure_ctx.output, ValueError)


def test_recursive_registration_updates_shared_descendant_instances():
    @component()
    def shared(value):
        return value + 1

    @component()
    def first_tail(value):
        return 2 * value

    @component()
    def second_tail(value):
        return value - 3

    first_composition = PipelineComponent(shared, first_tail)
    second_composition = PipelineComponent(shared, second_tail)

    def mark_shared(ctx):
        ctx.meta["registered_recursively"] = True

    first_composition.register_eval_hook(pre=mark_shared, recursive=True)

    ctx = EvaluateContext()
    assert second_composition(5, ctx) == 3
    assert set(ctx.meta) == {"chemfit.structure"}
    assert ctx.children[0].meta["registered_recursively"] is True
