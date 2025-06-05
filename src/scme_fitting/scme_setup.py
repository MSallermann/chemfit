from pyscme.parameters import parameter_H2O
from pyscme.scme_calculator import SCMECalculator
from pyscme.expansions import (
    get_energy_expansion_from_hdf5_file,
    get_moment_expansion_from_hdf5_file,
)
from pathlib import Path
from ase import Atoms
from ase.units import Bohr, Hartree
import numpy as np
from ase.constraints import FixBondLengths
import logging
from pydantic import BaseModel
from ase.geometry import find_mic

from typing import Optional, List


# SCME_COMMIT = "274aa6fa4881bcb662d12a8c80488fa103a55fd2"
# assert pyscme.version.commit() == SCME_COMMIT

__FOLDER__ = Path(__file__).parent.resolve()


def get_rotation_matrix():
    rand = np.random.random(size=(3, 3))
    Q, R = np.linalg.qr(rand)
    return Q


def get_random_h2o_molecule(theta=1.821207441224783, roh=0.9519607159623009):
    """Gives a water-molecule with random orientation and the oxygen at the origin"""

    pos = np.zeros(shape=(3, 3))

    # first row is the oxygen, so we leave it alone
    # second row is the first hydrogen, we place it at (roh, 0, 0)
    pos[1, :] = [roh, 0, 0]

    # third row is the second hydrogen, we place it at (sin(theta)*roh, cos(theta)*roh, 0)
    pos[2, :] = [np.cos(theta) * roh, np.sin(theta) * roh, 0]

    # lastly we multiply with a random rotation matrix
    pos = pos @ get_rotation_matrix()

    # return the transposed positions
    return pos


def setup_monomer() -> Atoms:
    h2o = get_random_h2o_molecule()
    atoms = Atoms(symbols="OHH", positions=h2o, pbc=[False, False, False])
    return atoms


def setup_dimer(oo_distance: float) -> Atoms:
    h2o_1 = get_random_h2o_molecule()
    h2o_2 = get_random_h2o_molecule() + oo_distance

    positions = np.vstack((h2o_1, h2o_2))
    atoms = Atoms(symbols="OHHOHH", positions=positions, pbc=[False, False, False])

    return atoms


def move_dimer_apart(atoms: Atoms, target_oo_distance: float):
    assert len(atoms) == 6

    oo_vector = atoms.get_distance(0, 3, vector=True, mic=False)
    current_oo_distance = np.linalg.norm(oo_vector)

    move_by = (
        (target_oo_distance - current_oo_distance) * oo_vector / current_oo_distance
    )

    atoms.positions[3:] += move_by


def constrain_dimer(atoms: Atoms):
    atoms.set_constraint(FixBondLengths(pairs=[[0, 3]]))


class SCMEParams(BaseModel):
    te: float = 1.2 / Bohr
    td: float = 7.5548 * Bohr

    Ar_OO: float = 8149.63 / Hartree
    Br_OO: float = -0.5515
    Cr_OO: float = -3.4695 * Bohr

    Ar_OH: float = 0.0  # 100.0
    Br_OH: float = 0.0  # -0.5
    Cr_OH: float = 0.0  # -3.5 * Bohr

    Ar_HH: float = 0.0  # 50.0
    Br_HH: float = 0.0  # -0.5
    Cr_HH: float = 0.0  # -3.5 * Bohr

    r_Br: float = 1.0 / Bohr

    rc_Disp: float = 8.0 / Bohr
    rc_Core: float = 7.5 / Bohr
    rc_Elec: float = 9.0 / Bohr

    C6: float = 46.4430e0
    C8: float = 1141.7000e0
    C10: float = 33441.0000e0

    w_rc_Elec: float = 2.0 / Bohr
    w_rc_Core: float = 2.0 / Bohr
    w_rc_Disp: float = 2.0 / Bohr

    max_iter_scf: int = 100
    scf_convcrit: float = 1e-8
    dms: bool = False
    qms: bool = False
    NC: list[int] = [0, 0, 0]


def setup_expansions(
    calc: SCMECalculator, parametrization_key: str, path_to_scme_expansions: Path
):
    file = Path(path_to_scme_expansions)
    assert file.exists()

    logging.debug("Setting up expansions")
    logging.debug(f"    {parametrization_key = }")
    logging.debug(f"    {file = }")

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


DEFAULT_PARAMS = SCMEParams()


def setup_calculator(
    atoms: Atoms,
    scme_params: SCMEParams,
    path_to_scme_expansions: Optional[Path],
    parametrization_key: str,
) -> SCMECalculator:
    atoms.calc = SCMECalculator(atoms, **dict(scme_params))
    parameter_H2O.Assign_parameters_H20(atoms.calc.scme)

    if parametrization_key is not None and path_to_scme_expansions is not None:
        setup_expansions(
            atoms.calc,
            parametrization_key=parametrization_key,
            path_to_scme_expansions=path_to_scme_expansions,
        )

    return atoms.calc


def arange_water_in_OHH_order(atoms: Atoms) -> Atoms:
    """
    Reorder atoms so each water molecule appears as O, H, H.

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
        raise ValueError(f"Number of atoms {n_atoms} is not a multiple of 3")

    mask_O = atoms.numbers == 8
    mask_H = atoms.numbers == 1
    if 2 * mask_O.sum() != mask_H.sum():
        raise ValueError("Mismatch between O and H counts for water molecules")

    new_order: List[Atoms] = []
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
    """
    Validate that each water molecule is ordered O, H, H and within tolerance.

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
        raise ValueError("Total atoms not divisible by 3")

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
