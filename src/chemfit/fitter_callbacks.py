"""
Predefined callback utilities for the ChemFit fitter.

These callbacks provide common functionality such as logging
optimization progress and persisting evaluation metadata during
optimization runs.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from chemfit.abstract_objective_function import Skipped
from chemfit.evaluate_context import NoOutput
from chemfit.fitter import FitContext
from chemfit.utils import next_free_path

logger = logging.getLogger(__name__)


# Logging progress
def log_progress(step: int, ctxs: list[FitContext]):
    """
    Log optimization progress.

    This callback prints a summary of the current optimization state
    for each evaluation context, including the current loss,
    parameters, and the best loss/parameters observed so far.

    It also reports the best loss and parameter set across all
    contexts.

    Args:
        step: Current optimizer step index.
        ctxs: List of ``FitContext`` instances used by the
            optimizer. Each context corresponds to one candidate slot.

    """

    logger.info("=" * 40)
    logger.info(f"Step = {step}")

    best_params: dict | None = {}
    best_loss: float | None = None
    for ictx, ctx in enumerate(ctxs):
        opt_context = ctx.opt_context
        logger.info(f"  Context {ictx}")
        logger.info(f"    Opt loss   = {opt_context.loss if opt_context else None}")
        logger.info(
            f"    Opt params = {opt_context.parameters if opt_context else None}"
        )
        logger.info(f"    Cur loss   = {ctx.loss}")
        logger.info(f"    Cur params = {ctx.parameters}")

        if opt_context is not None and (
            best_loss is None
            or (opt_context.loss is not None and best_loss > opt_context.loss)
        ):
            best_loss = opt_context.loss
            best_params = opt_context.parameters

    logger.info(f"  Opt loss (all contexts)   = {best_loss}")
    logger.info(f"  Opt params (all contexts) = {best_params}")
    logger.info("-" * 40)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, NoOutput):
            return None
        if isinstance(o, Skipped):
            return {"skipped": True, "reason": o.reason}
        if isinstance(o, BaseException):
            return f"{type(o).__name__}: {o}"
        return super().default(o)


def _write_json_atomic(
    path: Path,
    data: object,
    *,
    overwrite: bool = True,
) -> None:
    """Write JSON completely before exposing it at its destination path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output_file:
            json.dump(
                data,
                output_file,
                cls=NumpyEncoder,
                allow_nan=True,
            )
            output_file.flush()
            os.fsync(output_file.fileno())

        if overwrite:
            # Replacing only after serialization succeeds prevents an
            # interrupted write from destroying the previous checkpoint.
            temp_path.replace(path)
        else:
            # A hard link publishes the complete file atomically and fails if
            # another checkpoint already claimed this numbered destination.
            os.link(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_json_unique(path: Path, data: object) -> Path:
    """Write to the next free path without replacing an existing file."""

    while True:
        write_path = next_free_path(path)
        try:
            _write_json_atomic(write_path, data, overwrite=False)
        except FileExistsError:
            # Another writer may have claimed the path after
            # ``next_free_path`` checked it. Select a new path and retry.
            continue
        return write_path


def legacy_checkpoint_record(step: int, ctx: FitContext) -> dict[str, Any]:
    """Return the pre-``to_record`` best-parameter checkpoint shape."""

    return {
        "step": step,
        "loss": ctx.loss,
        "parameters": ctx.parameters,
        "meta": ctx.meta,
        "quantities": ctx.output if isinstance(ctx.output, dict) else None,
        "children": [child.to_summary(recursive=True) for child in ctx.children],
    }


class SaveMetaData:
    def __init__(self, output_folder: Path | str, use_legacy_format: bool = False):
        """
        Initialize a metadata-saving callback.

        This callback writes the metadata of each evaluation context
        to JSON files during optimization.

        Args:
            output_folder: Directory where metadata files will be
                written. The directory is created if it does not
                already exist.

        .. note::
            This callback is retained for backward compatibility.
            For recording optimization trajectories, prefer
            :class:`SaveTrajectory`

        """
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.use_legacy_format = use_legacy_format

    def __call__(self, step: int, ctxs: list[FitContext]):
        try:
            for ictx, ctx in enumerate(ctxs):
                record = (
                    legacy_checkpoint_record(step, ctx)
                    if self.use_legacy_format
                    else ctx.to_record()
                )
                _write_json_atomic(
                    self.output_folder / f"step_{step}_ctx_{ictx}.json",
                    record,
                )
        except Exception:
            logger.exception("Exception when trying to save metadata!")


class CheckpointBestParameters:
    """
    Callback that checkpoints the best parameters observed during fitting.

    Whenever a new best loss is detected across the provided
    ``FitContext`` instances, the corresponding parameters and
    metadata are written to disk. The file is overwritten whenever a
    better solution is found unless ``dont_overwrite`` is ``True``.

    If ``dont_overwrite`` is ``True``, each new best solution is written to a
    new file by inserting a numeric suffix into ``path``.

    This callback is useful for long-running optimizations, as it allows
    recovery of the best solution even if the optimization process
    crashes or is interrupted.

    """

    def __init__(
        self,
        path: Path | str,
        dont_overwrite: bool = False,
        use_legacy_format: bool = False,
    ):
        """
        Initialize the checkpoint callback.

        Args:
            path: File path where the best parameters will be written.
                By default, the file is overwritten whenever a better loss is
                found.
            dont_overwrite: Whether to preserve earlier checkpoints by adding a
                numeric suffix to each new checkpoint path.
            use_legacy_format: Write the deprecated flat checkpoint structure
                instead of the versioned ``FitContext.to_record()`` structure.

        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.best_loss = None
        self.dont_overwrite = dont_overwrite
        self.use_legacy_format = use_legacy_format

    def __call__(self, step: int, ctxs: list[FitContext]):
        for ctx in ctxs:
            opt_context = ctx.opt_context
            if opt_context is None or opt_context.loss is None:
                continue

            if self.best_loss is None or opt_context.loss < self.best_loss:
                record = (
                    legacy_checkpoint_record(step, opt_context)
                    if self.use_legacy_format
                    else opt_context.to_record()
                )
                if self.dont_overwrite:
                    write_out_path = write_json_unique(self.path, record)
                else:
                    write_out_path = self.path
                    _write_json_atomic(write_out_path, record)

                # Update in-memory state only after the durable checkpoint is
                # published, so a failed write can be retried on the next call.
                self.best_loss = opt_context.loss
                logger.info(
                    f"New best loss {opt_context.loss} written to {write_out_path}"
                )


class SaveTrajectory:
    def __init__(
        self,
        output_file: Path | str,
        only_improvements: bool = False,
        force_suffix: bool = False,
    ):
        """
        Initialize a trajectory saving callback.

        Args:
            output_file: File path where the trajectory will be saved in jsonl format

        """

        self.output_file = Path(output_file)

        if self.output_file.suffix != ".jsonl" and not force_suffix:
            msg = f"Use `.jsonl` as output file suffix. You are using `{self.output_file.suffix}`. Set `force_suffix=True` to override."
            raise ValueError(msg)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.only_improvements = only_improvements
        self.best_loss: float | None = None

    def __call__(self, step: int, ctxs: list[FitContext]):
        """Append the current evaluation records."""

        if len(ctxs) == 0:
            return

        lines = []
        best_loss = self.best_loss

        for slot, ctx in enumerate(ctxs):
            improved = ctx.loss is not None and (
                best_loss is None or ctx.loss < best_loss
            )

            if not self.only_improvements or improved:
                lines.append(
                    json.dumps(
                        {
                            "step": step,
                            "slot": slot,
                            "record": ctx.to_record(),
                        },
                        allow_nan=False,
                    )
                )

            if improved:
                best_loss = ctx.loss

        if not lines:
            return

        with self.output_file.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        self.best_loss = best_loss
