from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

DEFAULT_SLICE = slice(None, None, None)


def process_csv(
    paths_to_csv: Path | Sequence[Path],
    index: slice | list[slice] = DEFAULT_SLICE,
) -> tuple[list[Path], list[str], list[float]]:
    """
    Load a dataset CSV and extract file paths, tags, and reference energies.

    If a list of paths is passed it forwards them one by one to `process_single_csv` and collects
    the results.

    Args:
        paths_to_csv (Union[Path, Sequence[Path]]): Either a single path to a CSV for a list of paths
        index (Union[slice, Sequence[slice]]): Either a single slice or a list of slices which is applied to the data read from the CSVs

    Returns:
        tuple[list[Path], list[str], list[float]]:
        - **paths**: List of resolved `Path` objects to each data file.
        - **tags**: List of dataset tag strings.
        - **energies**: List of reference energies as floats.

    """
    # If it is a single path we just process it
    if isinstance(paths_to_csv, Path):
        return process_single_csv(paths_to_csv)

    if not isinstance(index, Sequence):
        index = [index] * len(paths_to_csv)

    paths = []
    tags = []
    energies = []

    for i, path_to_csv in zip(index, paths_to_csv):
        p, t, e = process_single_csv(path_to_csv, i)
        paths += p
        tags += t
        energies += e

    return paths, tags, energies


def process_single_csv(
    path_to_csv: Path, index: slice = DEFAULT_SLICE
) -> tuple[list[Path], list[str], list[float]]:
    """
    Load a dataset CSV and extract file paths, tags, and reference energies.

    The CSV must include the following columns:
      - Either `path` or `file`:
          * If `path` is present, each entry may be absolute or relative to the current working directory.
          * Otherwise, `file` entries are taken as relative to the CSV's parent directory.
          * If both are present, `path` takes precedence.
      - `tag`: A short string label for each dataset.
      - `reference_energy`: A numeric reference energy for each dataset.

    Additional columns are permitted and ignored.

    Args:
        path_to_csv (Path): Path to the CSV file describing the datasets.
        index (slice) slice(None, None, None): A slice which is applied to the data read from the CSV

    Returns:
        tuple[list[Path], list[str], list[float]]:
        - **paths**: List of resolved `Path` objects to each data file.
        - **tags**: List of dataset tag strings.
        - **energies**: List of reference energies as floats.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If neither `path` nor `file`, or if `tag` or `reference_energy` columns are missing.
        ValueError: If any `reference_energy` value cannot be converted to float.

    """
    df = pd.read_csv(path_to_csv)
    if "path" in df.columns:
        paths = [Path(p) for p in df["path"]]
    elif "file" in df.columns:
        base = path_to_csv.parent.resolve()
        paths = [base / Path(fname) for fname in df["file"]]
    else:
        msg = f"Error while processing {path_to_csv}. CSV must contain either a 'path' or 'file' column."
        raise KeyError(msg)

    if "tag" not in df.columns or "reference_energy" not in df.columns:
        msg = "Error while processing {path_to_csv}. CSV must contain 'tag' and 'reference_energy' columns."
        raise KeyError(msg)

    tags = list(df["tag"])
    try:
        energies = [float(e) for e in df["reference_energy"]]
    except Exception as err:
        msg = "Error while processing {path_to_csv}. All 'reference_energy' entries must be numeric."
        raise ValueError(msg) from err

    return paths[index], tags[index], energies[index]
