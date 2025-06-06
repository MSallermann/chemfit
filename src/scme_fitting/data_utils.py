from pathlib import Path
import pandas as pd
from typing import List
from .scme_setup import SCMEParams
from .combined_objective_function import CombinedObjectiveFunction


def process_csv(path_to_csv: Path) -> (List[Path], List[str], List[float]):
    """
    Read a CSV that has columns:
      - either 'path' or 'file'
      - 'tag'
      - 'reference_energy'
    Return (list_of_Paths, list_of_tags, list_of_reference_energies).
    """
    df = pd.read_csv(path_to_csv)
    if "path" in df.columns:
        paths = [Path(p) for p in df["path"]]
    else:
        # assume 'file' refers to files relative to the CSV’s parent directory
        base = path_to_csv.parent.resolve()
        paths = [base / Path(fname) for fname in df["file"]]

    tags = list(df["tag"])
    energies = list(df["reference_energy"])
    return paths, tags, energies
