from pathlib import Path
import json


def next_free_folder(base: Path) -> Path:
    """
    If 'base' does not exist, return 'base'. Otherwise attempt 'base_0', 'base_1', etc.
    until finding a non‐existent Path, then return that.
    """
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
