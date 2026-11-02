"""Tests for persistent evaluation-context records."""

import json
from importlib.metadata import version

import numpy as np
import pytest

from chemfit.abstract_objective_function import Skipped
from chemfit.context_serialization import ContextSerializationError
from chemfit.evaluate_context import NO_OUTPUT, EvaluateContext
from chemfit.fitter import FitContext


class UnsupportedValue:
    def __repr__(self) -> str:
        return f"<UnsupportedValue {'x' * 300}>"


class CustomContext(EvaluateContext):
    """Context with serializer-independent custom persistent state."""

    def record_sections(self):
        sections = dict(super().record_sections())
        sections["custom"] = {"label": "example"}
        return sections


class InvalidCustomContext(EvaluateContext):
    """Context that collides with a serializer-owned record key."""

    def record_sections(self):
        return {"status": "custom"}


def test_successful_context_produces_versioned_json_record():
    ctx = EvaluateContext()
    ctx.input = {"epsilon": 0.5}
    ctx.output = -12.3
    ctx.meta = {"runtime": 1.25}

    record = ctx.to_record()

    assert record == {
        "format": "chemfit.evaluate_context",
        "schema_version": 1,
        "chemfit_version": version("chemfit"),
        "root": {
            "input": {"epsilon": 0.5},
            "output": -12.3,
            "meta": {"runtime": 1.25},
            "status": "normal",
            "path": "root",
            "children": [],
        },
    }
    json.dumps(record, allow_nan=False)


def test_nested_contexts_have_stable_path_ids_and_preserve_order():
    root = EvaluateContext()
    first = root.spawn_ctx()
    second = root.spawn_ctx()
    grandchild = second.spawn_ctx()
    root.children = [first, second]
    second.children = [grandchild]
    root.meta["chemfit.structure"] = {
        "children": [{"index": 0}, {"index": 1}],
        "dependencies": [[], [0]],
    }

    first_record = root.to_record()
    second_record = root.to_record()

    assert first_record == second_record
    assert [child["path"] for child in first_record["root"]["children"]] == [
        "root.0",
        "root.1",
    ]
    assert first_record["root"]["children"][1]["children"][0]["path"] == "root.1.0"
    assert first_record["root"]["meta"]["chemfit.structure"] == {
        "children": [{"index": 0}, {"index": 1}],
        "dependencies": [[], [0]],
    }


def test_status_distinguishes_failure_none_and_no_output():
    root = EvaluateContext()
    successful_none = root.spawn_ctx()
    not_executed = root.spawn_ctx()
    root.output = ValueError("invalid parameter")
    root.meta["error"] = RuntimeError("details")
    successful_none.output = None
    root.children = [successful_none, not_executed]

    record = root.to_record()["root"]

    assert record["status"] == "exception"
    assert record["output"] == {
        "__type__": "exception",
        "module": "builtins",
        "class": "ValueError",
        "message": "invalid parameter",
    }
    assert record["meta"]["error"]["class"] == "RuntimeError"
    assert record["children"][0]["status"] == "normal"
    assert record["children"][0]["output"] is None
    assert record["children"][1]["status"] == "no_output"
    assert record["children"][1]["output"] == {"__type__": "no_output"}
    assert not_executed.output is NO_OUTPUT


def test_numpy_values_and_nonfinite_floats_are_json_compatible():
    ctx = EvaluateContext()
    ctx.output = {
        "array": np.array([1, 2, 3]),
        "scalar": np.float64(1.5),
        "integer": np.int64(4),
        "tuple": (1, 2, 3),
        "nan": float("nan"),
        "positive": float("inf"),
        "negative": float("-inf"),
    }

    output = ctx.to_record()["root"]["output"]

    assert output == {
        "array": [1, 2, 3],
        "scalar": 1.5,
        "integer": 4,
        "tuple": [1, 2, 3],
        "nan": "NaN",
        "positive": "Infinity",
        "negative": "-Infinity",
    }
    json.dumps(output, allow_nan=False)


def test_skipped_values_preserve_optional_reasons():
    ctx = EvaluateContext()
    ctx.output = [Skipped(), Skipped(reason=ValueError("unavailable"))]

    assert ctx.to_record()["root"]["output"] == [
        {"__type__": "skipped", "reason": None},
        {
            "__type__": "skipped",
            "reason": {
                "__type__": "exception",
                "module": "builtins",
                "class": "ValueError",
                "message": "unavailable",
            },
        },
    ]

    ctx.output = Skipped(reason=ValueError("not applicable"))
    assert ctx.to_record()["root"]["status"] == "skipped"


def test_strict_mode_reports_nested_unsupported_value_path():
    root = EvaluateContext()
    root.children = [root.spawn_ctx() for _ in range(3)]
    root.children[2].meta["custom_object"] = UnsupportedValue()

    with pytest.raises(
        ContextSerializationError,
        match=(
            r"root\.2\.meta\.custom_object: unsupported type "
            r"test_context_serialization\.UnsupportedValue"
        ),
    ):
        root.to_record(strict=True)


def test_permissive_mode_tags_unsupported_values_with_bounded_repr():
    ctx = EvaluateContext()
    ctx.output = UnsupportedValue()

    output = ctx.to_record()["root"]["output"]

    assert output["__type__"] == "unsupported"
    assert output["python_type"].endswith(".UnsupportedValue")
    assert len(output["repr"]) == 200


def test_reserved_type_keys_use_mapping_encoding():
    ctx = EvaluateContext()
    ctx.output = {"__type__": "exception", "message": "some data"}

    output = ctx.to_record()["root"]["output"]

    assert output == {
        "__type__": "mapping",
        "value": {"__type__": "exception", "message": "some data"},
    }


def test_nonstring_mapping_keys_are_rejected():
    ctx = EvaluateContext()
    ctx.output = {1: "one"}

    with pytest.raises(
        ContextSerializationError,
        match=r"Non-string dictionary key at \['root', 'output'\]: 1",
    ):
        ctx.to_record()


def test_fit_context_adds_scalar_fit_state_but_omits_best_snapshot():
    ctx = FitContext()
    ctx.input = {"x": 2.0}
    ctx.output = 4.0
    ctx.loss = 4.0
    ctx.n_evals = 12
    optimum = FitContext()
    optimum.input = {"x": 1.0}
    optimum.output = 1.0
    optimum.loss = 1.0
    optimum.n_evals = 5
    optimum.children = [EvaluateContext()]
    ctx.opt_context = optimum

    root = ctx.to_record()["root"]

    assert root["children"] == []
    assert root["fit"] == {"loss": 4.0, "n_evals": 12}
    assert ctx.opt_context is optimum


def test_context_subclass_declares_record_sections_without_serializer_changes():
    ctx = CustomContext()

    assert ctx.to_record()["root"]["custom"] == {"label": "example"}


def test_context_subclass_cannot_overwrite_reserved_sections():
    ctx = InvalidCustomContext()

    with pytest.raises(
        ContextSerializationError,
        match="Custom context record section contains reserved key: status",
    ):
        ctx.to_record()


def test_serialization_does_not_mutate_the_context():
    ctx = EvaluateContext()
    array = np.array([1.0, 2.0])
    metadata = {"values": [1, 2]}
    ctx.input = array
    ctx.meta = metadata

    ctx.to_record()

    assert ctx.input is array
    assert np.array_equal(ctx.input, [1.0, 2.0])
    assert ctx.meta is metadata
    assert ctx.meta == {"values": [1, 2]}
