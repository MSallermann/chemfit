import json
from pathlib import Path

from chemfit.fitter import FitterEvaluateContext
from chemfit.fitter_callbacks import CheckpointBestParameters


def make_context(loss: float, x: float) -> FitterEvaluateContext:
    ctx = FitterEvaluateContext()
    ctx.opt_loss = loss
    ctx.opt_params = {"x": x}
    return ctx


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
