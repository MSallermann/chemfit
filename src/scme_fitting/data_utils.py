from pathlib import Path
import pandas as pd
from typing import List
from .scme_setup import SCMEParams
from .scme_energy_objective_function import SCMEEnergyObjectiveFunction
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


def load_objective_functions_from_csv(
    path_to_csv: Path,
    default_params: SCMEParams,
    parametrization_key: str,
    path_to_scme_expansions: Path,
    divide_by_n_atoms: bool = True,
) -> CombinedObjectiveFunction:
    """
    Convenience wrapper that:
      1. Calls process_csv(...), extracting (paths, tags, reference_energies).
      2. Builds one SCMEEnergyObjectiveFunction per row, with given default_params,
         parametrization_key, path_to_scme_expansions, reference_energy, divide_by_n_atoms, tag.
      3. Packages them all into a CombinedObjectiveFunction and returns it.
    """
    paths, tags, energies = process_csv(path_to_csv)
    obj_list = []
    for xyz_file, tag, energy in zip(paths, tags, energies):
        obj = SCMEEnergyObjectiveFunction(
            default_scme_params=default_params,
            parametrization_key=parametrization_key,
            path_to_scme_expansions=path_to_scme_expansions,
            path_to_reference_configuration=xyz_file,
            reference_energy=energy,
            divide_by_n_atoms=divide_by_n_atoms,
            tag=tag,
        )
        obj_list.append(obj)
    return CombinedObjectiveFunction(obj_list)
