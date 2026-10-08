<p align="center">
    <img src="logo/chemfit_logo_portable.svg" width="400"/>
</p>

# About

**ChemFit** is a Python framework for building and optimizing simulation-based parameter-fitting problems in computational chemistry, molecular dynamics, and materials science.

A ChemFit objective can combine many systems, observables, reference data sets, and simulation workflows into a single fitting problem. Individual terms can use ASE calculators, external simulation programs, or arbitrary Python code.

The same scientific objective can be used with different execution and optimization strategies without being rewritten.

## What can ChemFit do?

ChemFit can be used to:

- fit model parameters simultaneously against energies, forces, structures, densities, or other observables;
- combine reference data from different systems and simulation conditions into one objective;
- use [ASE calculations and geometry optimizations](https://chemfit.readthedocs.io/en/latest/src/usage/ase_objective_function_api.html) as fitting terms;
- build [external simulation workflows](https://chemfit.readthedocs.io/en/latest/src/usage/external_computer.html) that generate input, run programs, post-process output, and extract quantities;
- evaluate independent calculations concurrently using [threads, processes, or custom schedulers](https://chemfit.readthedocs.io/en/latest/src/usage/parallel_execution.html), or distribute them using [MPI](https://chemfit.readthedocs.io/en/latest/src/usage/mpi.html);
- build [combined and nested objectives](https://chemfit.readthedocs.io/en/latest/src/usage/combined_objective_function.html) with weighting, custom reductions, and failure handling;
- use Nevergrad, SciPy, or a custom optimization loop.

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

A ChemFit fitting problem is built from **quantity computations**, **losses**, and **objective composition**.

The following example fits the Lennard-Jones energy scale `epsilon` against two reference energies while keeping `sigma` fixed:

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

Each ASE calculation produces quantities, `with_loss()` turns those quantities into a scalar fitting term, and `chemfit.combine()` joins the independent terms into one objective.

`chemfit.fit_nevergrad()` then optimizes the parameters and returns a `FitResult` containing the best parameters and loss, the optimizer recommendation, and the collected evaluation contexts.

For more examples of constructing objectives, see the [common workflows](https://chemfit.readthedocs.io/en/latest/src/usage/public_api.html).

## From small examples to larger fitting problems

The example above has only two energy terms, but the same structure can combine different systems, observables, and simulation methods:

```text
               ┌─ structure 1 / energy ── loss ─┐
               ├─ structure 2 / forces ── loss ─┤
parameters ────├─ liquid / density ────── loss ─┼─ objective
               ├─ external simulation ─── loss ─┤
               └─ custom observable ───── loss ─┘
```

The individual terms do not need to use the same simulation method. ASE calculations, external programs, and custom Python computations can participate in the same objective.

Objectives can also be [nested and combined](https://chemfit.readthedocs.io/en/latest/src/usage/combined_objective_function.html) with weights, custom reductions or aggregators, evaluation metadata, and exception handling.

# External simulation workflows

ChemFit can turn external simulation programs into reusable quantity computations.

For example:

```python
term = (
    chemfit.external_quantity("runs")
    .with_hook(write_input, template="input.template")
    .with_cmd(run_simulation, executable="my-simulator")
    .with_hook(postprocess)
    .with_parser(parse_results, "results.dat")
    .wait_for("simulation.done")
    .with_loss(loss, target=reference)
)
```

An external workflow can contain Python hooks, commands, output parsers, and completion files. Each evaluation runs in an isolated working directory.

This allows external simulation codes that can be launched and inspected from Python to participate in a ChemFit objective alongside native Python or ASE calculations.

See the [external simulation documentation](https://chemfit.readthedocs.io/en/latest/src/usage/external_computer.html) for details on commands, hooks, parsers, completion files, working directories, and failure handling.

# Concurrent execution

The scientific objective does not determine how it is executed.

For example, the Lennard-Jones fit above can evaluate multiple parameter candidates and objective terms concurrently:

```python
result = chemfit.fit_nevergrad(
    objective,
    initial={"epsilon": 0.7},
    bounds={"epsilon": (0.1, 2.0)},
    budget=100,
    batch_size=2,
    execution_workers=4,
)
```

`batch_size=2` allows Nevergrad to evaluate two parameter candidates at a time, while `execution_workers=4` allows up to four objective tasks to run concurrently.

An explicit `concurrent.futures`-compatible executor can also be supplied, for example to use processes for CPU-bound Python workloads.

The objective itself does not change.

See [parallel execution](https://chemfit.readthedocs.io/en/latest/src/usage/parallel_execution.html) for executors, schedulers, batching, and lower-level control.

## MPI

The same objective can also be evaluated across MPI ranks:

```python
from chemfit.mpi_scheduler import MPITreeScheduler

scheduler = MPITreeScheduler()

with scheduler.prepare(objective) as schedule:
    if schedule.rank == 0:
        result = chemfit.fit_nevergrad(
            schedule,
            initial={"epsilon": 0.7},
            bounds={"epsilon": (0.1, 2.0)},
            budget=100,
            batch_size=2,
        )
    else:
        schedule.worker_loop()
```

Run the script using the MPI launcher available on your system:

```bash
mpiexec -n 4 python fit.py
```

Install the MPI dependencies with:

```bash
pip install chemfit[mpi]
```

See the [MPI documentation](https://chemfit.readthedocs.io/en/latest/src/usage/mpi.html) for setup and execution details.

# Optimization

ChemFit does not tie an objective to a particular optimizer.

The high-level `chemfit.fit_nevergrad()` interface provides derivative-free optimization with Nevergrad. `chemfit.Fitter` provides lower-level control and can be used with SciPy, callbacks, explicit evaluation, batch evaluation, or custom optimization loops.

The objective and its execution strategy can therefore be reused when changing optimization methods.

See the [fitting documentation](https://chemfit.readthedocs.io/en/latest/src/usage/fitter.html) for the available fitting interfaces.

# Evaluation data and instrumentation

ChemFit retains information about individual evaluations through evaluation contexts. These can contain parameters, computed quantities, losses, metadata, and nested child evaluations.

[Evaluation hooks](https://chemfit.readthedocs.io/en/latest/src/usage/objective_hooks.html) can add instrumentation such as timing, unique evaluation IDs, logging, or application-specific diagnostics without changing the simulation or loss implementation.

# Limitations and alternatives

ChemFit is designed for simulation-based parameter fitting, particularly when an objective combines multiple calculations, observables, or simulation methods. It provides reusable building blocks for quantities, losses, and nested objectives, while keeping their execution and optimization separate. The same fitting problem can run locally or across an HPC cluster without requiring a workflow-management system.

ChemFit does not implement its own optimization algorithms or simulation engines. Out of the box, it integrates with SciPy and Nevergrad, and its objectives can also be optimized using Optuna or custom optimization loops. For simple objectives, using these tools directly may be easier.

Related frameworks offer different strengths:

- **[AiiDA](https://www.aiida.net/)** provides comprehensive workflow management, including persistent provenance, recovery, and HPC job management. Its [aiida-optimize](https://aiida-optimize.readthedocs.io/) plugin supports parameter optimization. If you already use AiiDA workflows, this is a natural choice. ChemFit is a lighter alternative when starting a fitting project without existing workflow infrastructure.
- **[Dakota](https://dakota.sandia.gov/)** offers extensive capabilities for optimization, parameter estimation, sensitivity analysis, and uncertainty quantification. ChemFit emphasizes defining and customizing fitting objectives directly in Python rather than providing a comprehensive numerical methods suite.
- **[pyiron](https://pyiron.org/)** provides an integrated environment for atomistic simulations, including job management, data storage, and potential fitting. ChemFit is more narrowly focused on fitting objectives that can combine arbitrary Python computations, ASE calculations, and external simulation programs.

ChemFit's built-in scheduler handles tree-structured objectives rather than arbitrary task-dependency graphs. Custom schedulers, executors, and hooks allow more sophisticated infrastructure to be integrated, but persistent workflow management and general DAG scheduling are not provided out of the box.

It is a fitting-specific library that can be used independently of a larger workflow ecosystem, while remaining extensible enough to integrate with one.

# Documentation

The [full ChemFit documentation](https://chemfit.readthedocs.io) covers the high-level API as well as ASE integration, external programs, combined objectives, fitting, parallel execution, MPI, hooks, and custom quantity computers.

# Citation

If you use ChemFit in academic work, please cite:

```bibtex
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

# Problems?

Please open an issue on the [ChemFit issue tracker](https://github.com/MSallermann/chemfit/issues).
