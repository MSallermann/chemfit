"""Direct ObjectiveComponent lifecycle tests (no schedulers)."""

import pytest

from chemfit.abstract_objective_function import (
    NO_OUTPUT,
    ObjectiveComponent,
    OpaquePipeline,
    component,
    quantity,
)
from chemfit.evaluate_context import EvaluateContext


class Square(ObjectiveComponent[int, int]):
    def _run(self, value: int, /, ctx: EvaluateContext) -> int:
        ctx.meta["ran"] = True
        return value * value


def test_hooks_run_in_registration_order():
    obj = Square().with_meta(tag="square")
    ctx = EvaluateContext()
    events = []

    def pre(label):
        def hook(c):
            events.append((label, c.input, c.output is NO_OUTPUT))

        return hook

    def post(label):
        def hook(c):
            events.append((label, c.output))

        return hook

    obj.register_eval_hook(pre=pre("pre-1"))
    obj.register_eval_hook(pre=pre("pre-2"))
    obj.register_eval_hook(post=post("post-1"))
    obj.register_eval_hook(post=post("post-2"))

    assert obj(3, ctx) == 9
    assert events == [
        ("pre-1", 3, True),
        ("pre-2", 3, True),
        ("post-1", 9),
        ("post-2", 9),
    ]
    assert ctx.meta == {"ran": True, "tag": "square"}


def test_post_hook_failure_replaces_the_outcome_and_stops_hooks():
    obj = Square()
    observed = []

    def fail(message):
        def hook(_ctx):
            observed.append(message)
            raise ValueError(message)

        return hook

    obj.register_eval_hook(post=fail("first"))
    obj.register_eval_hook(post=lambda _ctx: observed.append("middle"))
    obj.register_eval_hook(post=fail("last"))

    ctx = EvaluateContext()
    with pytest.raises(ValueError, match="first"):
        obj(2, ctx)

    assert observed == ["first"]
    assert isinstance(ctx.output, ValueError)


def test_post_hook_error_replaces_an_evaluation_error():
    error = ValueError("evaluation failed")
    observed = []

    @component(pass_ctx=True)
    def failing(_value, *, ctx):
        ctx.meta["ran"] = True
        raise error

    def bad_post(ctx):
        observed.append(("bad", ctx.output is error))
        msg = "post failed"
        raise RuntimeError(msg)

    def good_post(ctx):
        observed.append(("good", ctx.output is error))

    failing.register_eval_hook(post=bad_post)
    failing.register_eval_hook(post=good_post)
    ctx = EvaluateContext()

    with pytest.raises(RuntimeError, match="post failed"):
        failing(3, ctx)

    assert observed == [("bad", True)]
    assert isinstance(ctx.output, RuntimeError)
    assert ctx.meta["ran"] is True


def test_with_loss_preserves_type_errors():
    @quantity()
    def compute(value):
        return {"value": value}

    def invalid_loss(_quantities):
        msg = "loss calculation failed"
        raise TypeError(msg)

    objective = compute.with_loss(invalid_loss)
    assert isinstance(objective, OpaquePipeline)

    ctx = EvaluateContext()
    with pytest.raises(TypeError, match="loss calculation failed"):
        objective(3, ctx)

    assert isinstance(ctx.output, TypeError)
    assert len(ctx.children) == 2
    assert ctx.children[0].output == {"value": 3}
    assert isinstance(ctx.children[1].output, TypeError)


def test_bound_arguments_and_optional_loss_argument():
    @quantity()
    def compute(value, *, factor):
        return {"value": value * factor}

    def loss(quantities, offset=0):
        return quantities["value"] + offset

    obj = compute.bind(factor=2).with_loss(loss)
    ctx = EvaluateContext()

    assert obj(3, ctx) == 6
    assert [stage.output for stage in ctx.children] == [{"value": 6}, 6]
    assert compute.bind(factor=3).with_loss(loss, offset=4)(3) == 13
