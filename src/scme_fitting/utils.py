from pathlib import Path
import json
from collections.abc import MutableMapping
from typing import Any


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


# groked from here https://stackoverflow.com/a/6027615
def flatten_dict(dictionary: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dictionary into a flat dictionary by inserting a separator between sub keys.

    Args:
        dictionary (dict): The input dictionary
        parent_key (str, optional): Thee parent key. Defaults to "". Used for recursive implementation
        separator (str, optional): The separator to insert between keys. Defaults to ".".

    Returns:
        dict: flattened dictionary

    Example:
        >>> inp = {"a": {"b": 1.0, "c": 2.0, "d": {"e": "test"}}, "f": [1, 2]}
        >>> out = flatten_dict(inp, sep=".")
        >>> print(out)
        >>> {'a.b': 1.0, 'a.c': 2.0, 'a.d.e': 'test', 'f': [1, 2]}
    """
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def _insert_value(dictionary: dict, keys: list[str], value: Any):
    # Base case: just insert
    if len(keys) <= 1:
        dictionary[keys[0]] = value
    else:
        # recursion
        first_key = keys[0]
        if first_key not in dictionary:
            dictionary[first_key] = {}
        _insert_value(dictionary=dictionary[first_key], keys=keys[1:], value=value)


def unflatten_dict(dictionary: dict, sep: str = ".") -> dict:
    """Unflatten a dictionary by assuming subkeys are separated by a separator

    Args:
        dictionary (dict): The input dictionary
        sep (str, optional): The separator. Defaults to ".".

    Returns:
        dict: The unflattened output dictionary

    Example:
        >>> inp = {'a.b': 1.0, 'a.c': 2.0, 'a.d.e': 'test', 'f': [1, 2]}
        >>> out = unflatten_dict(inp, sep=".")
        >>> print(out)
        >>> {"a": {"b": 1.0, "c": 2.0, "d": {"e": "test"}}, "f": [1, 2]}
    """
    res = {}
    for key, value in dictionary.items():
        subkeys = key.split(sep)
        _insert_value(dictionary=res, keys=subkeys, value=value)
    return res
