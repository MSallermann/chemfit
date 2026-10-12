"""Direct evaluation tests for CombinedObjectiveFunction (no schedulers)."""

import math

import pytest

from chemfit.abstract_objective_function import EvaluateContext, quantity
from chemfit.combined_objective_function import (
    CombinedObjectiveFunction,
    mean_reducer,
    nan_exception_handler,
    skip_exception_handler,
)


def constant(value):
    return lambda _params: value


def broken(_params):
    raise RuntimeError("broken term")


def test_empty_join_is_rejected():
    with pytest.raises(ValueError):
        CombinedObjectiveFunction([])


@pytest.mark.parametrize(
    "weights", [[1], [1, -1], [1, float("nan")], [1, float("inf")]]
)
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        CombinedObjectiveFunction([constant(1), constant(2)], weights=weights)


def test_weighted_join_and_contexts():
    ctx = EvaluateContext()
    cob = CombinedObjectiveFunction([constant(2), constant(3)], weights=[4, 5])

    assert cob({"x": 1}, ctx) == 23
    assert ctx.input == {"x": 1}
    assert ctx.output == 23
    assert ctx.exception is None
    assert [c.output for c in ctx.incoming] == [2, 3]
    assert all(c.outgoing == (ctx,) for c in ctx.incoming)
    assert ctx.meta["skipped_indices"] == []


def test_custom_reduction_with_and_without_context():
    source = CombinedObjectiveFunction([constant(2), constant(4)])
    mean = source.with_reduction(mean_reducer)

    def inspect_context(terms, *, ctx):
        assert [c.output for c in ctx.incoming] == [2, 4]
        ctx.meta["inspected"] = True
        return max(terms)

    custom = mean.with_reduction(inspect_context)
    ctx = EvaluateContext()
    assert source({}) == 6
    assert mean({}) == 3
    assert custom({}, ctx) == 4
    assert ctx.meta["inspected"] is True
    assert source({}) == 6  # Fluent changes do not mutate the source.


def test_fluent_configuration_is_independent():
    source = CombinedObjectiveFunction([constant(2), constant(3)]).with_meta(
        tag="original"
    )
    weights = [2, 1]
    variant = source.with_weights(weights).with_meta(tag="variant")
    weights[0] = 100

    assert source({}) == 5
    assert variant({}) == 7
    assert source.static_meta_data == {"tag": "original"}
    assert variant.static_meta_data == {"tag": "variant"}
    assert all(
        a is b for a, b in zip(source.components(), variant.components(), strict=True)
    )


def test_default_exception_handler_raises():
    ctx = EvaluateContext()
    cob = CombinedObjectiveFunction([constant(2), broken])

    with pytest.raises(RuntimeError, match="broken term"):
        cob({}, ctx)

    assert isinstance(ctx.exception, RuntimeError)
    assert isinstance(ctx.incoming[1].exception, RuntimeError)


@pytest.mark.parametrize(
    ("handler", "expected", "skipped"),
    [
        (skip_exception_handler, 2.0, [1]),
        (nan_exception_handler, math.nan, []),
    ],
)
def test_exception_replacement(handler, expected, skipped):
    cob = CombinedObjectiveFunction([constant(2), broken]).with_exception_handler(
        handler
    )
    ctx = EvaluateContext()
    result = cob({}, ctx)

    if math.isnan(expected):
        assert math.isnan(result)
    else:
        assert result == expected
    assert ctx.meta["skipped_indices"] == skipped
    assert isinstance(ctx.incoming[1].exception, RuntimeError)


def test_nested_join():
    inner = CombinedObjectiveFunction([constant(1), constant(2)])
    outer = CombinedObjectiveFunction([inner, constant(4)], weights=[2, 1])
    ctx = EvaluateContext()

    assert outer({}, ctx) == 10
    assert [c.output for c in ctx.incoming] == [3, 4]
    assert [c.output for c in ctx.incoming[0].incoming] == [1, 2]


def test_opaque_pipeline_is_internal_to_a_join_term():
    @quantity()
    def quantities(_params, scale):
        return {"scale": scale}

    term = quantities.bind(scale=3).with_loss(lambda q: 2 * q["scale"])
    cob = CombinedObjectiveFunction([term, constant(1)])
    ctx = EvaluateContext()

    assert cob({}, ctx) == 7
    first, second = ctx.incoming[0].pipeline
    assert first.output == {"scale": 3}
    assert second.input == {"scale": 3}
    assert second.output == 6
    assert first not in ctx.incoming  # Pipeline stages aren't DAG nodes.


def test_recursive_hooks_stop_at_opaque_components():
    @quantity()
    def quantities(_params):
        return {"value": 2}

    term = quantities.with_loss(lambda q: q["value"])
    cob = CombinedObjectiveFunction([term, constant(1)])
    seen = []
    cob.register_eval_hook(pre=lambda ctx: seen.append(ctx), recursive=True)

    ctx = EvaluateContext()
    assert cob({}, ctx) == 3
    assert seen == [ctx, *ctx.incoming]
    assert not any(stage in seen for stage in ctx.incoming[0].pipeline)
