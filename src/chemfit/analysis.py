from scipy.spatial.transform.rotation import Rotation
from ase import Atoms


def kabsch(atoms1: Atoms, atoms2: Atoms):
    center1 = atoms1.center()
    center2 = atoms2.center()

    rot, rssd = Rotation.align_vectors(
        atoms1.get_positions(), atoms2.get_positions() + center1 - center2
    )

    return rssd
