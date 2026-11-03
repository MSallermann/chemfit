import copy
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from chemfit.abstract_objective_function import Skipped
from chemfit.fitter import FitContext
from chemfit.fitter_callbacks import (
    CheckpointBestParameters,
    SaveMetaData,
    SaveTrajectory,
)


def make_context(loss: float, x: float) -> FitContext:
    ctx = FitContext()
    ctx.loss = loss
    ctx.input = {"x": x}
    ctx.opt_context = copy.deepcopy(ctx)
    return ctx


def test_save_metadata_includes_nested_summaries(tmp_path: Path):
    ctx = FitContext()
    child = ctx.spawn_ctx()
    ctx.children.append(child)
    child.meta["marker"] = "child"

    SaveMetaData(tmp_path)(0, [ctx])

    saved = json.loads((tmp_path / "step_0_ctx_0.json").read_text())
    assert saved["format"] == "chemfit.evaluate_context"
    assert saved["root"]["children"][0]["meta"] == {"marker": "child"}


def test_checkpoint_includes_nested_metadata_from_the_best_evaluation(
    tmp_path: Path,
):
    ctx = make_context(loss=1.0, x=1.0)
    child = ctx.spawn_ctx()
    grandchild = child.spawn_ctx()
    child.meta["stage"] = "quantity"
    grandchild.meta["source"] = "simulation"
    child.children.append(grandchild)
    ctx.children = [child]
    ctx.opt_context = copy.deepcopy(ctx)
    ctx.loss = 99.0
    ctx.input = {"x": 99.0}

    path = tmp_path / "best.json"
    CheckpointBestParameters(path)(0, [ctx])

    saved = json.loads(path.read_text())
    root = saved["root"]
    assert root["fit"] == {"loss": 1.0, "n_evals": 0}
    assert root["input"] == {"x": 1.0}
    assert root["children"][0]["meta"] == {"stage": "quantity"}
    assert root["children"][0]["children"][0]["meta"] == {"source": "simulation"}


def test_save_metadata_serializes_numpy_scalars_and_skipped_outputs(tmp_path: Path):
    ctx = FitContext()
    ctx.output = np.float32(1.25)
    child = ctx.spawn_ctx()
    child.output = (
        np.int64(4),
        Skipped(),
        Skipped(reason=ValueError("unavailable")),
    )
    ctx.children.append(child)

    SaveMetaData(tmp_path)(0, [ctx])

    saved = json.loads((tmp_path / "step_0_ctx_0.json").read_text())
    root = saved["root"]
    assert root["output"] == 1.25
    assert root["children"][0]["output"] == [
        4,
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


def test_checkpoint_dont_overwrite_uses_next_available_path(tmp_path: Path):
    path = tmp_path / "best.json"

    CheckpointBestParameters(path, dont_overwrite=True)(
        0, [make_context(loss=2.0, x=2.0)]
    )
    CheckpointBestParameters(path, dont_overwrite=True)(
        1, [make_context(loss=1.0, x=1.0)]
    )

    first = json.loads(path.read_text())
    second = json.loads((tmp_path / "best_0.json").read_text())

    assert first["root"]["fit"]["loss"] == 2.0
    assert first["root"]["input"] == {"x": 2.0}
    assert second["root"]["fit"]["loss"] == 1.0
    assert second["root"]["input"] == {"x": 1.0}


def test_callbacks_write_the_same_record_shape(tmp_path: Path):
    ctx = make_context(loss=1.0, x=2.0)

    SaveMetaData(tmp_path / "meta")(3, [ctx])
    CheckpointBestParameters(tmp_path / "best.json")(3, [ctx])

    metadata = json.loads((tmp_path / "meta" / "step_3_ctx_0.json").read_text())
    checkpoint = json.loads((tmp_path / "best.json").read_text())
    assert metadata == checkpoint


def test_checkpoint_can_write_legacy_format(tmp_path: Path):
    path = tmp_path / "best.json"

    CheckpointBestParameters(path, use_legacy_format=True)(
        4, [make_context(loss=1.0, x=2.0)]
    )

    saved = json.loads(path.read_text())
    assert saved["loss"] == 1.0
    assert saved["parameters"] == {"x": 2.0}
    assert "schema_version" not in saved


def test_save_metadata_failure_preserves_previous_file(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    callback = SaveMetaData(tmp_path)
    ctx = FitContext()
    ctx.meta["value"] = "complete"
    callback(0, [ctx])
    output_path = tmp_path / "step_0_ctx_0.json"
    original = output_path.read_text(encoding="utf-8")

    ctx.meta["value"] = {1: "non-string key"}
    with caplog.at_level(logging.ERROR):
        callback(0, [ctx])

    assert "Exception when trying to save metadata" in caplog.text
    assert output_path.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".*.tmp")) == []


def test_checkpoint_failure_preserves_previous_file_and_best_loss(tmp_path: Path):
    path = tmp_path / "best.json"
    callback = CheckpointBestParameters(path)
    callback(0, [make_context(loss=2.0, x=2.0)])

    unserializable = make_context(loss=1.0, x=1.0)
    assert unserializable.opt_context is not None
    unserializable.opt_context.input = {1: "non-string key"}
    with pytest.raises(TypeError, match="Non-string dictionary key"):
        callback(1, [unserializable])

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["root"]["fit"]["loss"] == 2.0
    assert saved["root"]["input"] == {"x": 2.0}
    assert callback.best_loss == 2.0
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize(
    ("only_improvements", "expected"),
    [
        (
            False,
            [
                (0, 0, 2.0, 2.0),
                (0, 1, 1.0, 1.0),
                (1, 0, 3.0, 3.0),
                (1, 1, 0.5, 0.5),
                (2, 0, 4.0, 4.0),
            ],
        ),
        (
            True,
            [
                (0, 0, 2.0, 2.0),
                (0, 1, 1.0, 1.0),
                (1, 0, 0.75, 7.5),
                (1, 1, 0.5, 0.5),
            ],
        ),
    ],
)
def test_save_trajectory(
    tmp_path: Path,
    only_improvements: bool,
    expected: list[tuple[int, int, float, float]],
):
    path = tmp_path / "trajectory.jsonl"
    callback = SaveTrajectory(path, only_improvements=only_improvements)

    callback(0, [make_context(2.0, 2.0), make_context(1.0, 1.0)])

    # Current evaluation is worse than the retained best evaluation.
    ctx = make_context(3.0, 3.0)
    assert ctx.opt_context is not None
    ctx.opt_context.loss = 0.75
    ctx.opt_context.input = {"x": 7.5}

    callback(1, [ctx, make_context(0.5, 0.5)])

    # The retained best is equal to the previous global minimum.
    # This should not produce another improvement record.
    ctx = make_context(4.0, 4.0)
    assert ctx.opt_context is not None
    ctx.opt_context.loss = 0.5
    ctx.opt_context.input = {"x": 0.5}

    callback(2, [ctx])
    callback(3, [])

    entries = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]

    assert [
        (
            entry["step"],
            entry["slot"],
            entry["record"]["root"]["fit"]["loss"],
            entry["record"]["root"]["input"]["x"],
        )
        for entry in entries
    ] == expected

    for entry in entries:
        assert entry["record"]["format"] == "chemfit.evaluate_context"
        assert entry["record"]["schema_version"] == 1
        assert "step" not in entry["record"]["root"]["fit"]

    assert callback.best_loss == 0.5
