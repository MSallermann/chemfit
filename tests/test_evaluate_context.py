"""Evaluation context semantics, independent of scheduling."""

import copy
import pickle
from types import SimpleNamespace

from chemfit.evaluate_context import NO_OUTPUT, EvaluateContext


def test_defaults():
    ctx = EvaluateContext()

    assert ctx.input is None
    assert ctx.output is NO_OUTPUT
    assert ctx.meta == {}
    assert ctx.children == []
    assert not hasattr(ctx, "incoming")
    assert not hasattr(ctx, "outgoing")
    assert not hasattr(ctx, "pipeline")
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
    assert child.children == []


def test_recursive_summary_contains_one_child_hierarchy():
    root, child, grandchild = (EvaluateContext() for _ in range(3))
    root.input = {"x": 1}
    root.output = 5
    child.output = 2
    grandchild.output = 3
    root.children.append(child)
    child.children.append(grandchild)

    shallow = root.to_summary()
    recursive = root.to_summary(recursive=True)

    assert shallow["input"] == {"x": 1}
    assert shallow["output"] == 5
    assert "children" not in shallow
    assert recursive["children"][0]["output"] == 2
    assert recursive["children"][0]["children"][0]["output"] == 3
    assert recursive["children"][0]["children"][0]["children"] == []


def test_deepcopy_preserves_the_complete_evaluation_tree():
    root = EvaluateContext()
    root.input = {"x": [1]}
    root.output = 5
    root.meta["root"] = {"complete": True}
    child = root.spawn_ctx()
    child.meta["child"] = {"complete": True}
    root.children.append(child)

    copied = copy.deepcopy(root)
    copied.input["x"].append(2)
    copied.meta["root"]["complete"] = False
    copied.children[0].meta["child"]["complete"] = False

    assert copied is not root
    assert copied.input == {"x": [1, 2]}
    assert copied.output == 5
    assert copied.meta == {"root": {"complete": False}}
    assert copied.children[0].meta == {"child": {"complete": False}}
    assert copied.children[0] is not child
    assert root.input == {"x": [1]}
    assert root.meta == {"root": {"complete": True}}
    assert child.meta == {"child": {"complete": True}}


def test_worker_input_pickle_does_not_transfer_children():
    ctx = EvaluateContext(
        config=SimpleNamespace(value=2), shared=SimpleNamespace(value=3)
    )
    ctx.input = {"x": 4}
    ctx.meta["tag"] = "before"
    ctx.temp.worker_only = True
    ctx.output = 10
    ctx.children.append(EvaluateContext())

    restored = pickle.loads(pickle.dumps(ctx))  # noqa: S301 - trusted test value

    assert restored.input == {"x": 4}
    assert restored.config.value == 2
    assert restored.shared.value == 3
    assert restored.meta == {"tag": "before"}
    assert restored.output is NO_OUTPUT
    assert restored.children == []
    assert vars(restored.temp) == {}


def test_result_state_round_trip_preserves_nested_children():
    source = EvaluateContext()
    source.input = {"x": 1}
    source.output = 7
    source.meta["done"] = True
    source.config.source_only = True
    source.shared.source_only = True
    source.temp.source_only = True
    source.children = [source.spawn_ctx(), source.spawn_ctx()]
    source.children[0].input = {"x": 1}
    source.children[0].output = {"quantity": 3}
    source.children[1].input = {"quantity": 3}
    source.children[1].output = 7
    source.children[1].meta["stage"] = "loss"
    nested = source.children[0].spawn_ctx()
    nested.output = 3
    source.children[0].children.append(nested)

    target_shared = SimpleNamespace(target_only=True)
    target = EvaluateContext(
        config=SimpleNamespace(target_only=True), shared=target_shared
    )
    target.temp.target_only = True
    state = source.to_result_state()
    assert set(state) == {"input", "output", "meta", "children"}
    target.apply_result_state(state)

    assert target.input == source.input
    assert target.output == 7
    assert target.meta == {"done": True}
    assert [child.output for child in target.children] == [{"quantity": 3}, 7]
    assert target.children[0].children[0].output == 3
    assert target.children[1].input == {"quantity": 3}
    assert target.children[1].meta == {"stage": "loss"}
    assert vars(target.config) == {"target_only": True}
    assert target.shared is target_shared
    assert vars(target.temp) == {"target_only": True}


def test_result_state_preserves_exception_outcomes_in_nested_children():
    source = EvaluateContext()
    source.output = RuntimeError("outer")
    stage = source.spawn_ctx()
    stage.output = ValueError("inner")
    source.children.append(stage)

    target = EvaluateContext()
    target.apply_result_state(source.to_result_state())

    assert isinstance(target.output, RuntimeError)
    assert str(target.output) == "outer"
    assert isinstance(target.children[0].output, ValueError)
    assert str(target.children[0].output) == "inner"
