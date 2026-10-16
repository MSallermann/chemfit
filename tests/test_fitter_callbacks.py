import json
import logging
from pathlib import Path

import pytest

from chemfit.fitter import FitContext
from chemfit.fitter_callbacks import CheckpointBestParameters, SaveMetaData


def make_context(loss: float, x: float) -> FitContext:
    ctx = FitContext()
    ctx.opt_loss = loss
    ctx.opt_params = {"x": x}
    return ctx


def test_save_metadata_includes_nested_summaries(tmp_path: Path):
    ctx = FitContext()
    child = ctx.spawn_ctx()
    ctx.children.append(child)
    child.meta["marker"] = "child"

    SaveMetaData(tmp_path)(0, [ctx])

    saved = json.loads((tmp_path / "step_0_ctx_0.json").read_text())
    assert saved["children"][0]["meta"] == {"marker": "child"}


def test_checkpoint_dont_overwrite_uses_next_available_path(tmp_path: Path):
    path = tmp_path / "best.json"

    CheckpointBestParameters(path, dont_overwrite=True)(
        0, [make_context(loss=2.0, x=2.0)]
    )
    CheckpointBestParameters(path, dont_overwrite=True)(
        1, [make_context(loss=1.0, x=1.0)]
    )

    first = json.loads((tmp_path / "best_0.json").read_text())
    second = json.loads((tmp_path / "best_1.json").read_text())

    assert first["loss"] == 2.0
    assert first["parameters"] == {"x": 2.0}
    assert second["loss"] == 1.0
    assert second["parameters"] == {"x": 1.0}


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

    ctx.meta["value"] = object()
    with caplog.at_level(logging.ERROR):
        callback(0, [ctx])

    assert "Exception when trying to save meta data" in caplog.text
    assert output_path.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".*.tmp")) == []


def test_checkpoint_failure_preserves_previous_file_and_best_loss(tmp_path: Path):
    path = tmp_path / "best.json"
    callback = CheckpointBestParameters(path)
    callback(0, [make_context(loss=2.0, x=2.0)])

    unserializable = make_context(loss=1.0, x=1.0)
    unserializable.opt_params = {"value": object()}
    with pytest.raises(TypeError, match="JSON serializable"):
        callback(1, [unserializable])

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["loss"] == 2.0
    assert saved["parameters"] == {"x": 2.0}
    assert callback.best_loss == 2.0
    assert list(tmp_path.glob(".*.tmp")) == []
