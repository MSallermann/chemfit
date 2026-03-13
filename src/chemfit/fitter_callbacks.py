"""
Predefined callback utilities for the ChemFit fitter.

These callbacks provide common functionality such as logging
optimization progress and persisting evaluation metadata during
optimization runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from chemfit.fitter import FitterEvaluateContext

logger = logging.getLogger(__name__)


# Logging progress
def log_progress(step: int, ctxs: list[FitterEvaluateContext]):
    """
    Log optimization progress.

    This callback prints a summary of the current optimization state
    for each evaluation context, including the current loss,
    parameters, and the best loss/parameters observed so far.

    It also reports the best loss and parameter set across all
    contexts.

    Args:
        step: Current optimizer step index.
        ctxs: List of ``FitterEvaluateContext`` instances used by the
            optimizer. Each context corresponds to one evaluation
            worker.

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
        return super().default(o)


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
        self.output_folder.mkdir(exist_ok=True)

    def __call__(self, step: int, ctxs: list[FitterEvaluateContext]):
        try:
            for ictx, ctx in enumerate(ctxs):
                with (self.output_folder / f"step_{step}_ctx_{ictx}.json").open(
                    "w"
                ) as f:
                    json.dump(
                        ctx.to_meta_data(), f, indent=4, skipkeys=True, cls=NumpyEncoder
                    )
        except Exception:
            logger.exception("Exception when trying to save meta data!")


class CheckpointBestParameters:
    """
    Callback that checkpoints the best parameters observed during fitting.

    Whenever a new best loss is detected across the provided
    ``FitterEvaluateContext`` instances, the corresponding parameters and
    metadata are written to disk. The file is overwritten whenever a
    better solution is found.

    This callback is useful for long-running optimizations, as it allows
    recovery of the best solution even if the optimization process
    crashes or is interrupted.

    """

    def __init__(self, path: Path | str):
        """
        Initialize the checkpoint callback.

        Args:
            path: File path where the best parameters will be written.
                The file is overwritten whenever a better loss is found.

        """
        self.path = Path(path)
        self.best_loss = None

    def __call__(self, step: int, ctxs: list[FitterEvaluateContext]):
        for ctx in ctxs:
            if ctx.opt_loss is None:
                continue

            if self.best_loss is None or ctx.opt_loss < self.best_loss:
                self.best_loss = ctx.opt_loss

                data = {
                    "step": step,
                    "loss": ctx.opt_loss,
                    "parameters": ctx.opt_params,
                    "meta": ctx.opt_meta,
                }

                with self.path.open("w") as f:
                    json.dump(data, f, indent=4)

                logger.info(f"New best loss {ctx.opt_loss} written to {self.path}")
