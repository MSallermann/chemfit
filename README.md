<p align="center">
    <img src="logo/chemfit_logo_portable.svg" width="400"/>
</p>

# About

ChemFit is a Python package for concurrent simulation-based parameter
optimization. It can be used with ASE calculators, external executables, and
custom Python objectives.

# Installation

From PyPI:

```bash
pip install chemfit
```

Or, locally:

```bash
git clone git@github.com:MSallermann/chemfit.git
pip install -e chemfit
```

# Quick start: fit a Lennard-Jones potential

This example fits the Lennard-Jones energy scale `epsilon` while keeping
`sigma` fixed:

```python
import chemfit
from ase import Atoms
from ase.calculators.lj import LennardJones


reference_energies = {
    1.1: -0.98,
    1.5: -0.32,
}

objective = chemfit.combine(
    chemfit.ase_quantity(
        Atoms("Ar2", positions=[(0, 0, 0), (distance, 0, 0)])
    )
    .with_calculator(
        lambda p, atoms, ctx: LennardJones(
            epsilon=p["epsilon"], sigma=1.0, rc=100.0
        )
    )
    .with_loss(
        lambda q, target: (q["energy"] - target) ** 2,
        target=target,
    )
    for distance, target in reference_energies.items()
)

result = chemfit.fit_nevergrad(
    objective,
    initial={"epsilon": 0.7},
    bounds={"epsilon": (0.1, 2.0)},
    budget=100,
)

print(result.best_parameters)
```

Each ASE calculation becomes an independent objective term, `with_loss()`
turns its energy into a fitting loss, and `chemfit.combine()` joins the terms.
`chemfit.fit_nevergrad()` optimizes the parameter mapping and returns a `FitResult`
with `best_parameters`, `best_loss`, the optimizer recommendation, and the
evaluation contexts.

Objective terms can also run across threads, processes, or MPI, and ChemFit supports external programs, custom schedulers, SciPy, hooks, and lower-level control; see the full [**documentation**](https://chemfit.readthedocs.io) for details.

## Concurrency

For example, we can run the same optimization with four Python threads:

```python
result = chemfit.fit_nevergrad(
    objective,
    initial={"epsilon": 0.7},
    bounds={"epsilon": (0.1, 2.0)},
    budget=100,
    batch_size=2,
    execution_workers=4
)
```

**Explanation**: With `batch_size=2`, we evaluate two parameter trials concurrently and `execution_workers=4` distributes their parallel leaf work (two leaves per trial) across up to four Python threads.
By passing an explicit `executor` we can use a different `concurrent.futures`-compatible executor, such as `loky.ProcessPoolExecutor`.

With a little more boilerplate, we can run the optimization with MPI instead. Simply make sure `mpi4py` is installed (e.g., with `pip install chemfit[mpi]`) and adjust the script like so:

```python
from chemfit.mpi_scheduler import MPITreeScheduler

# ...

scheduler = MPITreeScheduler()
with scheduler.prepare(objective) as schedule:
    if schedule.rank == 0:
        result = chemfit.fit_nevergrad(
            schedule,
            initial={"epsilon": 0.7},
            bounds={"epsilon": (0.1, 2.0)},
            budget=100,
            batch_size=2
        )
    else:
        schedule.worker_loop()
```

Then simply start the script with an MPI runner (e.g., `srun`, `mpirun` or `mpiexec` depending on your environment).

```bash
mpiexec -n 4 python fit.py
```

# Documentation

Please check the **documentation** for details [here](https://chemfit.readthedocs.io).


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
