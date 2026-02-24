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
    logger.info("=" * 40)
    logger.info(f"Step = {step}")

    best_params: dict | None = {}
    best_loss: float | None = None
    for ictx, ctx in enumerate(ctxs):
        logger.info(f"  Context {ictx}")
        logger.info(f"    Opt loss = {ctx.opt_loss}")
        logger.info(f"    Opt params = {ctx.opt_params}")
        logger.info(f"    Cur loss = {ctx.loss}")
        logger.info(f"    Cur params = {ctx.parameters}")

        if best_loss is None or (ctx.opt_loss is not None and best_loss > ctx.opt_loss):
            best_loss = ctx.opt_loss
            best_params = ctx.opt_params

    logger.info(f"  Opt loss (all contexts)   = {best_loss}")
    logger.info(f"  Best params (all context) = {best_params}")
    logger.info("-" * 40)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


class SaveMetaData:
    def __init__(self, output_folder: Path | str):
        """Saves the meta data to a folder."""
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
