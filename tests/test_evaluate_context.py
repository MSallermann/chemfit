"""Evaluation context semantics, independent of scheduling."""

import pickle
from types import SimpleNamespace

import pytest

from chemfit.evaluate_context import EvaluateContext, NO_OUTPUT


def test_defaults():
    ctx = EvaluateContext()

    assert ctx.input is None
    assert ctx.output is NO_OUTPUT
    assert ctx.exception is None
    assert ctx.meta == {}
    assert ctx.incoming == ()
    assert ctx.outgoing == ()
    assert ctx.pipeline == []
    assert vars(ctx.temp) == {}
    assert vars(ctx.config) == {}
    assert vars(ctx.shared) == {}


def test_spawn_ctx_copies_config_but_shares_resources():
    shared = SimpleNamespace(resource=[])
    ctx = EvaluateContext(
        config=SimpleNamespace(options={"threshold": 3}), shared=shared
    )
    child = ctx.spawn_ctx()

    assert child is not ctx
    assert child.config is not ctx.config
    assert child.config.options == ctx.config.options
    assert child.shared is shared

    child.config.options["threshold"] = 9
    child.shared.resource.append("used")
    assert ctx.config.options["threshold"] == 3
    assert ctx.shared.resource == ["used"]
    assert child.incoming == ()
    assert child.pipeline == []


def test_dependency_edges_are_bidirectional_and_deduplicated():
    a, b, join = (EvaluateContext() for _ in range(3))
    join.add_incoming(a)
    join.add_incoming(b)
    join.add_incoming(a)

    assert join.incoming == (a, b)
    assert a.outgoing == (join,)
    assert b.outgoing == (join,)
    assert a.incoming == ()

    with pytest.raises(ValueError, match="itself"):
        join.add_incoming(join)


def test_shared_dependency_and_clear_incoming():
    source, left, right = (EvaluateContext() for _ in range(3))
    left.add_incoming(source)
    right.add_incoming(source)

    assert source.outgoing == (left, right)
    left._clear_incoming()
    assert left.incoming == ()
    assert source.outgoing == (right,)
    assert right.incoming == (source,)


def test_recursive_summary_includes_dependencies_and_pipeline():
    root, dependency, stage = (EvaluateContext() for _ in range(3))
    root.input = {"x": 1}
    root.output = 5
    dependency.output = 2
    stage.output = 3
    root.add_incoming(dependency)
    root.pipeline.append(stage)

    shallow = root.to_summary()
    recursive = root.to_summary(recursive=True)

    assert shallow["input"] == {"x": 1}
    assert shallow["output"] == 5
    assert "incoming" not in shallow
    assert "pipeline" not in shallow
    assert recursive["incoming"][0]["output"] == 2
    assert recursive["pipeline"][0]["output"] == 3


def test_worker_input_pickle_does_not_transfer_graph_or_pipeline():
    ctx = EvaluateContext(
        config=SimpleNamespace(value=2), shared=SimpleNamespace(value=3)
    )
    ctx.input = {"x": 4}
    ctx.meta["tag"] = "before"
    ctx.output = 10
    ctx.add_incoming(EvaluateContext())
    ctx.pipeline.append(EvaluateContext())

    restored = pickle.loads(pickle.dumps(ctx))

    assert restored.input == {"x": 4}
    assert restored.config.value == 2
    assert restored.shared.value == 3
    assert restored.meta == {"tag": "before"}
    assert restored.output is NO_OUTPUT
    assert restored.incoming == ()
    assert restored.outgoing == ()
    assert restored.pipeline == []


def test_result_state_round_trip_preserves_pipeline_not_graph():
    source = EvaluateContext()
    source.input = {"x": 1}
    source.output = 7
    source.meta["done"] = True
    source.pipeline = [source.spawn_ctx(), source.spawn_ctx()]
    source.pipeline[0].input = {"x": 1}
    source.pipeline[0].output = {"quantity": 3}
    source.pipeline[1].input = {"quantity": 3}
    source.pipeline[1].output = 7
    source.pipeline[1].meta["stage"] = "loss"

    target = EvaluateContext()
    predecessor = EvaluateContext()
    target.add_incoming(predecessor)
    target.apply_result_state(source.to_result_state())

    assert target.input == source.input
    assert target.output == 7
    assert target.meta == {"done": True}
    assert [stage.output for stage in target.pipeline] == [{"quantity": 3}, 7]
    assert target.pipeline[1].input == {"quantity": 3}
    assert target.pipeline[1].meta == {"stage": "loss"}
    assert target.incoming == (predecessor,)
    assert predecessor.outgoing == (target,)
    assert all(stage.incoming == () for stage in target.pipeline)


def test_result_state_preserves_exceptions_in_nested_pipeline():
    source = EvaluateContext()
    source.exception = RuntimeError("outer")
    stage = source.spawn_ctx()
    stage.exception = ValueError("inner")
    source.pipeline.append(stage)

    target = EvaluateContext()
    target.apply_result_state(source.to_result_state())

    assert isinstance(target.exception, RuntimeError)
    assert str(target.exception) == "outer"
    assert isinstance(target.pipeline[0].exception, ValueError)
    assert str(target.pipeline[0].exception) == "inner"
