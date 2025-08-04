import logging
from pathlib import Path
from typing import Optional

from ase import Atoms
from ase.geometry import find_mic
from pyscme.expansions import (
    get_energy_expansion_from_hdf5_file,
    get_moment_expansion_from_hdf5_file,
)
from pyscme.parameters import parameter_H2O
from pyscme.scme_calculator import SCMECalculator

logger = logging.getLogger(__name__)


def setup_expansions(
    calc: SCMECalculator, parametrization_key: str, path_to_scme_expansions: Path
):
    file = Path(path_to_scme_expansions)

    logger.debug("Setting up expansions")
    logger.debug(f"    {parametrization_key = }")
    logger.debug(f"    {file = }")

    if not file.exists():
        msg = f"Expansion file `{file}` does not exist"
        raise Exception(msg)

    energy_expansion = get_energy_expansion_from_hdf5_file(
        path_to_file=file, key_to_dataset="energy"
    )
    dipole_expansion = get_moment_expansion_from_hdf5_file(
        path_to_file=file, key_to_dataset=f"{parametrization_key}/dipole"
    )
    quadrupole_expansion = get_moment_expansion_from_hdf5_file(
        path_to_file=file, key_to_dataset=f"{parametrization_key}/quadrupole"
    )
    octupole_expansion = get_moment_expansion_from_hdf5_file(
        path_to_file=file, key_to_dataset=f"{parametrization_key}/octupole"
    )
    hexadecapole_expansion = get_moment_expansion_from_hdf5_file(
        path_to_file=file, key_to_dataset=f"{parametrization_key}/hexadecapole"
    )
    dip_dip_expansion = get_moment_expansion_from_hdf5_file(
        path_to_file=file, key_to_dataset=f"{parametrization_key}/dip_dip"
    )
    dip_quad_expansion = get_moment_expansion_from_hdf5_file(
        path_to_file=file, key_to_dataset=f"{parametrization_key}/dip_quad"
    )
    quad_quad_expansion = get_moment_expansion_from_hdf5_file(
        path_to_file=file, key_to_dataset=f"{parametrization_key}/quad_quad"
    )

    calc.scme.monomer_energy_expansion = energy_expansion
    calc.scme.static_dipole_moment_expansion = dipole_expansion
    calc.scme.static_quadrupole_moment_expansion = quadrupole_expansion
    calc.scme.static_octupole_moment_expansion = octupole_expansion
    calc.scme.static_hexadecapole_moment_expansion = hexadecapole_expansion
    calc.scme.dip_dip_polarizability_expansion = dip_dip_expansion
    calc.scme.dip_quad_polarizability_expansion = dip_quad_expansion
    calc.scme.quad_quad_polarizability_expansion = quad_quad_expansion


def setup_calculator(
    atoms: Atoms,
    params: dict,
    path_to_scme_expansions: Optional[Path],
    parametrization_key: str,
) -> SCMECalculator:
    atoms.calc = SCMECalculator(atoms, **params)
    parameter_H2O.Assign_parameters_H2O(atoms.calc.scme)

    if parametrization_key is not None and path_to_scme_expansions is not None:
        setup_expansions(
            atoms.calc,
            parametrization_key=parametrization_key,
            path_to_scme_expansions=path_to_scme_expansions,
        )

    return atoms.calc


def arrange_water_in_OHH_order(atoms: Atoms) -> Atoms:
    """Reorder atoms so each water molecule appears as O, H, H.

    Parameters
    ----------
    atoms : Atoms
        Original Atoms object containing water molecules.

    Returns
    -------
    Atoms
        New Atoms object with OHH ordering and no constraints.

    Raises
    ------
    ValueError
        If atom counts or ratios are inconsistent with water.

    """
    n_atoms = len(atoms)
    if n_atoms % 3 != 0:
        msg = f"Number of atoms {n_atoms} is not a multiple of 3"
        raise ValueError(msg)

    mask_O = atoms.numbers == 8
    mask_H = atoms.numbers == 1
    if 2 * mask_O.sum() != mask_H.sum():
        msg = "Mismatch between O and H counts for water molecules"
        raise ValueError(msg)

    new_order: list[Atoms] = []
    for atom_O in atoms[mask_O]:
        new_order.append(atom_O)
        H_sorted = sorted(
            atoms[mask_H],
            key=lambda a: find_mic(atom_O.position - a.position, cell=atoms.cell)[1],
        )
        new_order.extend(H_sorted[:2])

    result = atoms.copy()
    result.set_constraint()
    result.set_atomic_numbers([a.number for a in new_order])
    result.set_positions([a.position for a in new_order])
    return result


def check_water_is_in_OHH_order(atoms: Atoms, OH_distance_tol: float = 2.0) -> bool:
    """Validate that each water molecule is ordered O, H, H and within tolerance.

    Parameters
    ----------
    atoms : Atoms
        Atoms object to validate.
    OH_distance_tol : float, optional
        Maximum allowed O-H distance (default is 2.0 Å).

    Raises
    ------
    ValueError
        If ordering or distances violate water OHH assumptions.

    """
    n_atoms = len(atoms)
    if n_atoms % 3 != 0:
        msg = "Total atoms not divisible by 3"
        raise ValueError(msg)

    good = True
    for i in range(n_atoms // 3):
        idxO, idxH1, idxH2 = 3 * i, 3 * i + 1, 3 * i + 2
        if (
            atoms.numbers[idxO] != 8
            or atoms.numbers[idxH1] != 1
            or atoms.numbers[idxH2] != 1
        ):
            good = False
            break

        d1 = atoms.get_distance(idxO, idxH1, mic=True)
        d2 = atoms.get_distance(idxO, idxH2, mic=True)
        if d1 > OH_distance_tol or d2 > OH_distance_tol:
            good = False
            break

    return good
