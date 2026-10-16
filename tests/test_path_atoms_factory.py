from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write

import chemfit
from chemfit.ase_objective_function import PathAtomsFactory


def test_path_atoms_factory_requires_one_selected_structure(tmp_path: Path):
    path = tmp_path / "trajectory.xyz"
    write(path, [Atoms("H"), Atoms("He")])
    factory = PathAtomsFactory(str(path), index=0)

    first = factory()
    second = chemfit.ase_quantity(path, index=slice(1, 2)).atoms_factory()

    assert factory.path == path
    assert first.get_chemical_symbols() == ["H"]
    assert second.get_chemical_symbols() == ["He"]

    with pytest.raises(
        ValueError,
        match=r"selected 2 structures.*expected exactly one",
    ):
        PathAtomsFactory(path, index=":")()


def test_path_atoms_factory_reports_invalid_paths(tmp_path: Path):
    path = tmp_path / "missing.xyz"

    with pytest.raises(FileNotFoundError, match=str(path)):
        PathAtomsFactory(path)()

    with pytest.raises(ValueError, match=rf"not a file: {tmp_path}"):
        PathAtomsFactory(tmp_path)()


def test_path_atoms_factory_adds_context_to_ase_read_errors(tmp_path: Path):
    path = tmp_path / "broken.xyz"
    path.write_text("not an xyz file", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{path}.*index 0") as exception_info:
        PathAtomsFactory(path, index=0)()

    assert exception_info.value.__cause__ is not None
