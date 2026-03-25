.. _parallel_execution:

#####################################
Parallel and Distributed Execution
#####################################

Once you have defined an objective in ChemFit, you have several ways to
run it in parallel.

This page shows common patterns you can copy and adapt.

--------------------------------------------------------------------
Recipe: parallelize a multi-term objective (executor)
--------------------------------------------------------------------

Use this when your objective consists of independent terms and you want
to evaluate them in parallel on a single machine.

.. code-block:: python

    from concurrent.futures import ThreadPoolExecutor

    from chemfit.abstract_objective_function import EvaluateContext
    from chemfit.combined_objective_function import CombinedObjectiveFunction
    from chemfit.executor_wrapper_cob import ExecutorWrapperCOB
    from chemfit.wrap_funcs import to_objective_functor

    @to_objective_functor()
    def term1(params):
        return (params["x"] - 1.0) ** 2

    @to_objective_functor()
    def term2(params):
        return (params["x"] - 2.0) ** 2

    @to_objective_functor()
    def term3(params):
        return (params["x"] - 3.0) ** 2

    objective = CombinedObjectiveFunction([term1, term2, term3])
    wrapped = ExecutorWrapperCOB(objective)

    ctx = EvaluateContext(executor=ThreadPoolExecutor(3))
    value = wrapped({"x": 0.0}, ctx)

----------------------------------
Recipe: parallelize a multi-term objective (MPI)
----------------------------------

Use this when your objective has many small terms and communication
overhead matters.

.. code-block:: python

    from chemfit.abstract_objective_function import EvaluateContext
    from chemfit.combined_objective_function import CombinedObjectiveFunction
    from chemfit.mpi_wrapper_cob import MPIWrapperCOB
    from chemfit.wrap_funcs import to_objective_functor

    @to_objective_functor()
    def term(params):
        return (params["x"] - 1.0) ** 2

    objective = CombinedObjectiveFunction([term] * 100)

    with MPIWrapperCOB(objective) as mpi:
        if mpi.rank == 0:
            ctx = EvaluateContext()
            value = mpi({"x": 0.0}, ctx)
        else:
            mpi.worker_loop()

This distributes terms across MPI ranks with very low communication
overhead.

--------------------------------------------------------------------
Recipe: evaluate objectives in parallel (no fitter)
--------------------------------------------------------------------

Use this when you want to evaluate a ChemFit-compatible callable yourself.

.. code-block:: python

    from concurrent.futures import ThreadPoolExecutor

    from chemfit.abstract_objective_function import EvaluateContext
    from chemfit.executor_utils import map_with_context
    from chemfit.wrap_funcs import to_objective_functor

    @to_objective_functor()
    def objective(params):
        return (params["x"] - 2.0) ** 2

    params_list = [{"x": i} for i in range(4)]
    ctxs = [EvaluateContext() for _ in params_list]

    with ThreadPoolExecutor(4) as executor:
        values = map_with_context(
            executor,
            objective,
            params_list,
            ctxs=ctxs,
        )

.. note::

    The :class:`~chemfit.fitter.Fitter` uses this same pattern internally
    when evaluating multiple parameter sets in parallel.

--------------------------------------------------------------------
Recipe: parallel optimization (Fitter)
--------------------------------------------------------------------

Use this when you want the optimizer to explore multiple parameter sets
concurrently.

.. code-block:: python

    from concurrent.futures import ThreadPoolExecutor

    from chemfit.fitter import Fitter
    from chemfit.wrap_funcs import to_objective_functor

    @to_quantity_computer()
    def objective(params):
        return (params["x"] - 2.0) ** 2

    fitter = Fitter(objective, {"x": 0.0})

    opt_params = fitter.fit_nevergrad(
        budget=100,
        num_workers=4,
        executor=ThreadPoolExecutor(4),
    )

--------------------------------------------------------------------
Recipe: run external simulations with Slurm (srun)
--------------------------------------------------------------------

Use this when ChemFit runs inside a Slurm allocation and each evaluation
launches an external program.

Minimal ``sbatch`` script:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=chemfit
    #SBATCH --nodes=1
    #SBATCH --ntasks=4
    #SBATCH --time=01:00:00

    module load python
    python run_fit.py

Example computer:

.. code-block:: python

    import subprocess
    from pathlib import Path

    from chemfit.abstract_objective_function import QuantityComputer


    class SlurmComputer(QuantityComputer):
        def __init__(self, workdir: str):
            super().__init__()
            self.workdir = Path(workdir)

        def _compute(self, params, ctx):
            input_file = self.workdir / "input.txt"
            output_file = self.workdir / "output.txt"

            with input_file.open("w") as f:
                f.write(f"x = {params['x']}\n")

            subprocess.run(
                ["srun", "my_simulation_code", str(input_file), str(output_file)],
                check=True,
            )

            with output_file.open() as f:
                value = float(f.read().strip())

            return {"value": value}

Run multiple evaluations in parallel:

.. code-block:: python

    from concurrent.futures import ThreadPoolExecutor

    from chemfit.abstract_objective_function import EvaluateContext
    from chemfit.executor_utils import map_with_context

    computer = SlurmComputer("run")
    objective = computer.with_loss(lambda q: q["value"] ** 2)

    params_list = [{"x": i} for i in range(4)]
    ctxs = [EvaluateContext() for _ in params_list]

    with ThreadPoolExecutor(4) as executor:
        values = map_with_context(
            executor,
            objective,
            params_list,
            ctxs=ctxs,
        )

The executor controls how many ``srun`` calls are launched concurrently.

--------------------------------------------------------------------
Which parallelization strategy should I use?
--------------------------------------------------------------------

**ExecutorWrapperCOB**
    Good default for parallelizing combined objectives on a single machine.

**MPIWrapperCOB**
    Preferred for many small objective terms where communication overhead
    matters.

**map_with_context**
    Use when you want to evaluate objectives in parallel yourself.

**Fitter (num_workers)**
    Use when running optimization and you want parallel parameter exploration.

**ThreadPoolExecutor**
    Good default for external codes, NumPy, and compiled workloads.

**ProcessPoolExecutor / loky**
    Better for CPU-heavy Python code.

**Dask**
    Use for distributed Python workloads (via ``client.get_executor()``).

**mpi4py.futures.MPIPoolExecutor**
    Use in MPI environments with an executor-style interface.

**Slurm + srun**
    Use when launching external simulations inside an HPC allocation.
