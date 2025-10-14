.. _mpi:

==================
Running with MPI
==================

The MPI integration in ChemFit parallelizes the evaluation of a
:py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`
across MPI ranks. Each rank evaluates a slice of the combined objective's
terms, and rank 0 reduces their partial sums to a single scalar loss.

Core idea
---------

- Build a multi-term objective with
  :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`.
- Wrap it in :py:class:`~chemfit.mpi_wrapper_cob.MPIWrapperCOB`.
- Rank 0 calls the optimizer on the wrapper; worker ranks run a loop and
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

- **All ranks** construct the same :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction` and enter the :py:class:`~chemfit.mpi_wrapper_cob.MPIWrapperCOB` context.
- **Rank 0** runs fitting on the MPI wrapper object and may call ``gather_meta_data()`` when desired.
- **Worker ranks (rank > 0)** enter ``worker_loop()`` and wait for signals and parameter broadcasts.

Thanks to lazy loading patterns in quantity computers, building the combined
objective on every rank is typically cheap; heavy resources are only needed
on ranks that actually evaluate those terms.

Minimal example
---------------------------------------

This example shows the structure.

.. code-block:: python

   import numpy as np
   from chemfit.fitter import Fitter
   from chemfit.combined_objective_function import CombinedObjectiveFunction
   from chemfit.mpi_wrapper_cob import MPIWrapperCOB

   # all ranks construct the list of terms
   terms = magic_from_elsewhere()

   cob = CombinedObjectiveFunction(objective_functions=terms)  # weights default to 1.0

   # wrap with MPI and run
   with MPIWrapperCOB(cob) as mpi_cob:
       if mpi_cob.rank == 0:
           initial_params = {"epsilon": 2.0, "sigma": 1.5}
           fitter = Fitter(mpi_cob, initial_params=initial_params)
           opt_params = fitter.fit_scipy()

           # Optionally collect per-term metadata from all ranks
           meta = mpi_cob.gather_meta_data()
           print(opt_params)
           print(meta)
       else:
           mpi_cob.worker_loop()

How it partitions work
----------------------

Within each evaluation:

- Rank 0 broadcasts the parameter dictionary to all ranks.
- Each rank evaluates its local slice of work.
- All ranks participate in a reduction (sum) to rank 0.
- Rank 0 receives the global loss and returns it to the optimizer.

Common pitfalls
---------------

- Forgetting to call ``worker_loop()`` on ranks > 0 results in rank 0 blocking
  forever at the first broadcast.
- Creating different numbers of terms on different ranks will mis-partition work.
  Always construct the same ``CombinedObjectiveFunction`` on all ranks.
- Modifying the set of terms after constructing the MPI wrapper is not supported.
  Build the final combined objective first, then wrap.

Troubleshooting
---------------

- Hang or deadlock at first evaluation:
- Ensure every non-zero rank entered ``worker_loop()``.
- Ensure all ranks are using the same communicator and number of terms.
- Immediate exception on worker ranks:
- Check per-term code paths for assumptions about unavailable files, GPUs,
    or environment on worker nodes.
- Unexpectedly high wall-clock time:
- Imbalanced slices if terms differ vastly in cost. Consider grouping similar-cost
  terms, or split the combined objective into multiple parts and use
  ``add_flat`` to rebalance.

Summary
-------

- Parallelization is at the **objective-term** level via
  :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`.
- :py:class:`~chemfit.mpi_wrapper_cob.MPIWrapperCOB` broadcasts params, slices work,
  reduces losses, and provides metadata gathering.
- Rank 0 runs the optimizer; all other ranks run a worker loop.
- Keep objectives sliceable, deterministic, and consistently constructed across ranks.
