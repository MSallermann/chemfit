from pathlib import Path
import pandas as pd

def process_csv(path_to_csv: Path) -> tuple[list[Path], list[str], list[float]]:
    """Load a dataset CSV and extract file paths, tags, and reference energies.

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
        raise KeyError("CSV must contain either a 'path' or 'file' column.")

    if "tag" not in df.columns or "reference_energy" not in df.columns:
        raise KeyError("CSV must contain 'tag' and 'reference_energy' columns.")

    tags = list(df["tag"])
    try:
        energies = [float(e) for e in df["reference_energy"]]
    except Exception as err:
        raise ValueError("All 'reference_energy' entries must be numeric.") from err

    return paths, tags, energies
