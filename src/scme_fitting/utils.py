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


def iterate_nested_dict(d, subkeys: list[str] = []):
    """Iterates over a nested dict

    Args:
        dictionary (dict): The input dictionary
        subkeys (list[str], optional): The current list of subkeys. Only used for the recursive implementation

    Example:
        >>> inp = {"a": {"b": 1.0, "c": 2.0, "d": {"e": "test"}}, "f": [1, 2]}
        >>> for keys, value in iterate_nested_dict(inp):
        >>>     print(keys, value)
        >>> ['a', 'b'] 1.0
            ['a', 'c'] 2.0
            ['a', 'd', 'e'] test
            ['f'] [1, 2]
    """

    for key, value in d.items():
        if isinstance(value, MutableMapping):
            yield from iterate_nested_dict(
                value, subkeys=subkeys + [key]
            )  # Recursively yield from sub-dictionary
        else:
            yield subkeys + [key], value


def flatten_dict(dictionary: dict, sep: str = ".") -> dict:
    """Flatten a nested dictionary into a flat dictionary by inserting a separator between sub keys.

    Args:
        dictionary (dict): The input dictionary
        separator (str, optional): The separator to insert between keys. Defaults to ".".

    Returns:
        dict: flattened dictionary

    Example:
        >>> inp = {"a": {"b": 1.0, "c": 2.0, "d": {"e": "test"}}, "f": [1, 2]}
        >>> out = flatten_dict(inp, sep=".")
        >>> print(out)
        >>> {'a.b': 1.0, 'a.c': 2.0, 'a.d.e': 'test', 'f': [1, 2]}
    """
    res = {}
    for keys, value in iterate_nested_dict(dictionary):
        key_out = sep.join(keys)
        res[key_out] = value
    return res


def set_nested_value(
    dictionary: dict, keys: list[str], value: Any, subdict_factory=dict
):
    # Base case: just insert
    if len(keys) <= 1:
        dictionary[keys[0]] = value
    else:
        # recursion
        first_key = keys[0]
        if first_key not in dictionary:
            dictionary[first_key] = subdict_factory()
        set_nested_value(dictionary=dictionary[first_key], keys=keys[1:], value=value)


def has_nested_value(dictionary: dict, keys: list[str]):
    # Base case
    if len(keys) <= 1:
        return keys[0] in dictionary
    else:
        first_key = keys[0]
        return get_nested_value(dictionary=dictionary[first_key], keys=keys[1:])


def get_nested_value(dictionary: dict, keys: list[str], default=None):
    # Base case
    if len(keys) <= 1:
        return dictionary.get(keys[0], default)
    else:
        first_key = keys[0]
        return get_nested_value(
            dictionary=dictionary[first_key], keys=keys[1:], default=default
        )


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
        set_nested_value(dictionary=res, keys=subkeys, value=value)
    return res
