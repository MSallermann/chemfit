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
        logger.info(f"  Context {ictx}")
        logger.info(f"    Opt loss   = {ctx.opt_loss}")
        logger.info(f"    Opt params = {ctx.opt_params}")
        logger.info(f"    Cur loss   = {ctx.loss}")
        logger.info(f"    Cur params = {ctx.parameters}")

        if best_loss is None or (ctx.opt_loss is not None and best_loss > ctx.opt_loss):
            best_loss = ctx.opt_loss
            best_params = ctx.opt_params

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
    skipkeys: bool = False,
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
                indent=4,
                skipkeys=skipkeys,
                cls=NumpyEncoder,
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


class SaveMetaData:
    def __init__(self, output_folder: Path | str):
        """
        Initialize a metadata-saving callback.

        This callback writes the metadata of each evaluation context
        to JSON files during optimization.

        Args:
            output_folder: Directory where metadata files will be
                written. The directory is created if it does not
                already exist.

        """
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, step: int, ctxs: list[FitContext]):
        try:
            for ictx, ctx in enumerate(ctxs):
                _write_json_atomic(
                    self.output_folder / f"step_{step}_ctx_{ictx}.json",
                    ctx.to_summary(recursive=True),
                    skipkeys=True,
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

    def __init__(self, path: Path | str, dont_overwrite: bool = False):
        """
        Initialize the checkpoint callback.

        Args:
            path: File path where the best parameters will be written.
                By default, the file is overwritten whenever a better loss is
                found.
            dont_overwrite: Whether to preserve earlier checkpoints by adding a
                numeric suffix to each new checkpoint path.

        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.best_loss = None
        self.dont_overwrite = dont_overwrite
        self.counter = -1

    def __call__(self, step: int, ctxs: list[FitContext]):
        for ctx in ctxs:
            if ctx.opt_loss is None:
                continue

            if self.best_loss is None or ctx.opt_loss < self.best_loss:
                data = {
                    "step": step,
                    "loss": ctx.opt_loss,
                    "parameters": ctx.opt_params,
                    "meta": ctx.opt_meta,
                    "quantities": ctx.opt_quantities,
                }

                write_out_path = self.path
                if self.dont_overwrite:
                    next_counter = self.counter + 1
                    while True:
                        write_out_path = self.path.with_name(
                            f"{self.path.stem}_{next_counter}{self.path.suffix}"
                        )
                        try:
                            _write_json_atomic(
                                write_out_path,
                                data,
                                overwrite=False,
                            )
                            break
                        except FileExistsError:
                            next_counter += 1
                    self.counter = next_counter
                else:
                    _write_json_atomic(write_out_path, data)
                    self.counter += 1

                # Update in-memory state only after the durable checkpoint is
                # published, so a failed write can be retried on the next call.
                self.best_loss = ctx.opt_loss
                logger.info(f"New best loss {ctx.opt_loss} written to {write_out_path}")
