import json
from pathlib import Path
from typing import Any

import numpy as np
from pydictnest import flatten_dict


def next_free_folder(base: Path) -> Path:
    """If 'path/to/base' does not exist, return 'path/to/base'. Otherwise attempt 'path/to/base_0', 'path/to/base_1', etc. until finding a non-existent Path, then return that."""
    base = Path(base)

    if not base.exists():
        return base

    i = 0
    while True:
        candidate = base.with_name(f"{base.name}_{i}")
        if not candidate.exists():
            return candidate
        i += 1


class ExtendedJSONEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, Path):
            return str(o)
        super().default(o)
        return None


def dump_dict_to_file(file: Path, dictionary: dict) -> None:
    """Write `dictionary` as JSON to `file` (with indent=4)."""
    file.parent.mkdir(exist_ok=True, parents=True)
    with file.open("w") as f:
        json.dump(dictionary, f, indent=4, cls=ExtendedJSONEncoder)


def create_initial_params(
    adjustable_params: list[str], default_params: dict
) -> dict[str, float]:
    return {k: dict(default_params)[k] for k in adjustable_params}


def check_params_near_bounds(
    params: dict,
    bounds: dict,
    relative_tol: float,
) -> list[tuple[str, float, float, float]]:
    """Check if any of the parameters are near or beyond the bounds.

    The criterions checked are

    1. param_value < lower + relative_tol * (upper - lower)
    2. param_value > upper - relative_tol * (upper - lower)

    Args:
        params(dict): the dict of params to check
        bounds(dict): the dict of bounds to check
        relative_tol(float):
            The tolerance, relative to the span of the bounds.
            Positive numbers mean the values must fulfill a stricter bound
            Zero means the values must fulfill the exact bound
            Negative numbers mean the values must fulfill a looser bound

    Returns:
        A list of tuples with information about parameters, which violate the constraint.
        Each tuple contains
            - A string identifying the parameter in a flattened dict
            - The value of the parameter
            - The lower bound
            - The upper bound

    """
    flat_params = flatten_dict(params)
    flat_bounds = flatten_dict(bounds)

    problematic_params = []

    for kp, vp in flat_params.items():

        # Get the bounds (if they are not specified, we set them both to None)
        lower, upper = flat_bounds.get(kp, (None, None))

        if lower is not None and upper is not None:
            abs_tol = relative_tol * np.abs(upper - lower)

            if vp < lower + abs_tol or vp > upper - abs_tol:
                problematic_params.append((kp, vp, lower, upper))

    return problematic_params
