from pathlib import Path
import json
from pydictnest import flatten_dict
import numpy as np


def next_free_folder(base: Path) -> Path:
    """
    If 'path/to/base' does not exist, return 'path/to/base'. Otherwise attempt 'path/to/base_0', 'path/to/base_1', etc.
    until finding a non-existent Path, then return that.
    """
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
    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        else:
            super().default(o)


def dump_dict_to_file(file: Path, dictionary: dict) -> None:
    """
    Write `dictionary` as JSON to `file` (with indent=4).
    """
    file.parent.mkdir(exist_ok=True, parents=True)
    with open(file, "w") as f:
        json.dump(dictionary, f, indent=4, cls=ExtendedJSONEncoder)


def create_initial_params(
    adjustable_params: list[str], default_params: dict
) -> dict[str, float]:
    return {k: dict(default_params)[k] for k in adjustable_params}


def check_params_near_bounds(params: dict, bounds: dict, relative_tol: float) -> list:
    """Check if any of the parameters are near the bounds"""

    flat_params = flatten_dict(params)
    flat_bounds = flatten_dict(bounds)

    problematic_params = []

    for kp, vp in flat_params.items():

        # Get the bounds (if they are not specified, we set them both to None)
        lower, upper = flat_bounds.get(kp, (None, None))

        if lower is not None and upper is not None:
            abs_tol = relative_tol * np.abs(vp)

            if vp - lower < abs_tol or upper - vp < abs_tol:
                problematic_params.append([kp, vp, lower, upper])

    return problematic_params
