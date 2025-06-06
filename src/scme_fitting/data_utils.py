from pathlib import Path
import pandas as pd

def process_csv(path_to_csv: Path) -> tuple[list[Path], list[str], list[float]]:
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
