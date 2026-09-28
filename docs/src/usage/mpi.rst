.. _mpi:

==================
Running with MPI
==================

The MPI integration in ChemFit parallelizes the evaluation of a
:py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`
across MPI ranks. Each rank evaluates a slice of the combined objective's
terms, and rank 0 applies the combined objective's reduction to the gathered
term values.

Core idea
---------

- Build a multi-term objective with
  :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`.
- Assign an :py:class:`~chemfit.mpi_policy.MPIPolicy` as its execution policy.
- Rank 0 calls the optimizer on the combined objective; worker ranks run a loop and
  wait for broadcast work items.


Environment and dependencies
----------------------------

- A working MPI installation (mpich, Open MPI, etc.)
- ``mpi4py`` installed (optional extra: ``pip install chemfit[mpi]``)

Launch your script with:

::

   mpirun -n 4 python script.py


High-level workflow
-------------------

- **All ranks** construct the same :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction` and enter an :py:class:`~chemfit.mpi_policy.MPIPolicy` context.
- **All ranks** assign that policy to the combined objective.
- **Rank 0** runs fitting on the combined objective. Evaluation metadata is gathered into its evaluation context automatically.
- **Worker ranks (rank > 0)** enter ``worker_loop(cob)`` and wait for signals and parameter broadcasts.

Thanks to lazy loading patterns in quantity computers, building the combined
objective on every rank is typically cheap; heavy resources are only needed
on ranks that actually evaluate those terms.

Minimal example
---------------------------------------

This example shows the structure.

.. code-block:: python

   from chemfit.abstract_objective_function import EvaluateContext
   from chemfit.fitter import Fitter
   from chemfit.combined_objective_function import CombinedObjectiveFunction
   from chemfit.mpi_policy import MPIPolicy

   # all ranks construct the list of terms
   terms = magic_from_elsewhere()

   cob = CombinedObjectiveFunction(objective_functions=terms)  # weights default to 1.0

   # install the MPI policy and run
   with MPIPolicy() as mpi:
       cob.execution_policy = mpi
       if mpi.rank == 0:
           initial_params = {"epsilon": 2.0, "sigma": 1.5}
           fitter = Fitter(cob, initial_params=initial_params)
           opt_params = fitter.fit_scipy()

           # An explicit evaluation exposes gathered per-term metadata.
           ctx = EvaluateContext()
           cob(opt_params, ctx)
           print(opt_params)
           print(ctx.meta["children"])
       else:
           mpi.worker_loop(cob)

How it partitions work
----------------------

Within each evaluation:

- Rank 0 broadcasts the parameter dictionary to all ranks.
- Each rank evaluates its local slice of work.
- All ranks gather their term results and metadata to rank 0.
- Rank 0 applies the combined objective's configured reduction and returns the
  global loss to the optimizer.

Common pitfalls
---------------

- Forgetting to call ``worker_loop(cob)`` on ranks > 0 results in rank 0 blocking
  forever at the first broadcast.
- Different term lists or ordering on different ranks will mis-partition work.
  Construct the same final ``CombinedObjectiveFunction`` on every rank before
  starting the worker loop.

Troubleshooting
---------------

- Hang or deadlock at first evaluation:
- Ensure every non-zero rank entered ``worker_loop(cob)``.
- Ensure all ranks are using the same communicator and number of terms.
- Immediate exception on worker ranks:
- Check per-term code paths for assumptions about unavailable files, GPUs,
    or environment on worker nodes.
- Unexpectedly high wall-clock time:
- Imbalanced slices if terms differ vastly in cost. Consider ordering or grouping
  terms so that rank-local slices have similar total cost.

Summary
-------

- Parallelization is at the **objective-term** level via
  :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`.
- :py:class:`~chemfit.mpi_policy.MPIPolicy` broadcasts evaluation contexts,
  slices work, and gathers term results and metadata.
- Rank 0 runs the optimizer; all other ranks run a worker loop.
- Keep objectives sliceable, deterministic, and consistently constructed across ranks.
