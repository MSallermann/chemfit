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
integration, separated quantity and loss functions, a combined objective,
parallel Nevergrad workers, built-in and custom fitter callbacks, bounds, and
evaluation metadata.

```python
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ase import Atoms
from ase.calculators.lj import LennardJones

from chemfit.abstract_objective_function import EvaluateContext
from chemfit.ase_objective_function import SinglePointASEComputer
from chemfit.combined_objective_function import (
    CombinedObjectiveFunction,
    mean_reducer,
)
from chemfit.executor_wrapper_cob import ExecutorWrapperCOB
from chemfit.fitter import Fitter, FitterEvaluateContext
from chemfit.fitter_callbacks import log_progress


# Each evaluation receives a copied geometry and a fresh ASE calculator.
def attach_calculator(atoms: Atoms) -> None:
    atoms.calc = LennardJones(rc=100.0)


# Candidate parameters are applied immediately before ASE evaluates the atoms.
def apply_parameters(atoms: Atoms, parameters: dict[str, float]) -> None:
    assert atoms.calc is not None
    atoms.calc.set(**parameters)


# Quantity computers and loss functions remain separate and reusable.
def squared_error(quantities: dict[str, Any], target: float) -> float:
    return (quantities["energy"] - target) ** 2


# Generate synthetic reference data for epsilon = sigma = 1.
def reference_energy(distance: float) -> float:
    inverse_distance = 1.0 / distance
    return 4.0 * (inverse_distance**12 - inverse_distance**6)


# Turn one reference geometry into an energy quantity computer with a loss.
def energy_term(distance: float):
    computer = SinglePointASEComputer(
        calc_factory=attach_calculator,
        param_applier=apply_parameters,
        atoms_factory=lambda: Atoms(
            "Ar2", positions=[(0.0, 0.0, 0.0), (distance, 0.0, 0.0)]
        ),
        tag=f"distance={distance}",
    )
    return computer.with_loss(squared_error, target=reference_energy(distance))


# Custom callbacks receive the current step and one context per worker.
best_history: list[tuple[int, float]] = []
def remember_best(step: int, contexts: list[FitterEvaluateContext]) -> None:
    losses = [ctx.opt_loss for ctx in contexts if ctx.opt_loss is not None]
    if losses:
        best_history.append((step, min(losses)))


# Combine independent reference configurations into one mean loss.
distances = [0.95, 1.25, 1.60]
combined = CombinedObjectiveFunction(
    [energy_term(distance) for distance in distances],
    reduction=mean_reducer,
)
initial_params = {"epsilon": 0.7, "sigma": 1.2}
num_workers = 6


# Allow every energy term in a four-candidate batch to run concurrently.
with ThreadPoolExecutor(max_workers=num_workers * len(distances)) as term_executor:
    objective = ExecutorWrapperCOB(combined, executor=term_executor)
    fitter = Fitter(
        objective,
        initial_params=initial_params,
        bounds={"epsilon": (0.1, 2.0), "sigma": (0.5, 1.5)},
    )

    # Mix a built-in callback with any number of user-defined callbacks.
    logging.basicConfig(level=logging.INFO)
    fitter.register_callback(log_progress, n_steps=10)
    fitter.register_callback(remember_best, n_steps=1)

    # Nevergrad also evaluates batches of six candidates concurrently.
    result = fitter.fit_nevergrad(
        budget=400,
        optimizer_str="TwoPointsDE",
        num_workers=num_workers,
    )

    # An explicit context exposes quantities, losses, and nested term metadata.
    context = EvaluateContext()
    loss = objective(result, context)
    print(result, loss, best_history[-1], len(context.meta["children"]))
```

Each distance becomes an independent objective term with its own evaluation
context. `ExecutorWrapperCOB` evaluates those terms in a shared thread pool,
while Nevergrad evaluates four parameter candidates concurrently in a second
pool. The built-in callback logs detailed progress every ten optimizer steps,
while the custom callback records the best loss after every step. The final
parent context retains each term's quantities, loss, and metadata for
inspection.
