<p align="center">
    <img src="https://github.com/msallermann/chemfit/blob/next/logo/chemfit_logo_portable.svg?raw=true" width="400"/>
</p>

# About

ChemFit is a Python package for concurrent force-field parameter optimization. It can be used with ASE calculators and external executables.

# Documentation

Please check the **documentation** for details [here](https://chemfit.readthedocs.io).


# Installation

From PyPi:

```bash
pip install chemfit
```

Or, locally:

```bash
git clone git@github.com:MSallermann/chemfit.git
pip install chemfit
```

# Citation

If you find ChemFit useful and happen to use it in any academic context, please use this reference to cite it:

```
@misc{sallermann2026chemfitframeworkautomatedhighdimensional,
      title={ChemFit: A framework for automated high-dimensional model parameter optimization},
      author={Moritz Sallermann and Amrita Goswami and Rosana Collepardo-Guevara and Alberto Ocana and Hannes Jónsson and Elvar Ö. Jónsson and Jorge R. Espinosa},
      year={2026},
      eprint={2603.11769},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2603.11769},
}
```

Thanks!

# Problems?

Please open an issue [here](https://github.com/MSallermann/chemfit/issues).

# Quick start: fit a Lennard-Jones potential

This complete example recovers the Lennard-Jones parameters
`epsilon = sigma = 1` from three reference dimer energies. It demonstrates ASE
integration, combined objectives, bounds, and parallel fitting.

```python
import chemfit
from ase import Atoms
from ase.calculators.lj import LennardJones


def squared_error(quantities, *, target):
    return (quantities["energy"] - target) ** 2


def reference_energy(distance: float) -> float:
    return 4.0 * (1.0/distance**12 - 1.0/distance**6)


def energy_term(distance: float):
    atoms = Atoms("Ar2", positions=[(0, 0, 0), (distance, 0, 0)])
    return (
        chemfit.ase_quantity(atoms)
        .with_calculator(
            lambda parameters, atoms, ctx: LennardJones(rc=100, **parameters)
        )
        .with_loss(squared_error, target=reference_energy(distance))
    )


distances = [0.95, 1.25, 1.60]
objective = chemfit.combine(
    *(energy_term(distance) for distance in distances),
    reduction=chemfit.mean_reducer,
)

result = chemfit.fit(
    objective,
    initial={"epsilon": 0.7, "sigma": 1.2},
    bounds={"epsilon": (0.1, 2.0), "sigma": (0.5, 1.5)},
    budget=400,
    workers=6,
    optimizer="TwoPointsDE",
)
print(result.best_parameters, result.best_loss)
```

Each distance becomes an independent objective term with its own evaluation
context. `chemfit.fit` evaluates up to six parameter candidates concurrently
and returns both the optimizer recommendation and the evaluation contexts.
