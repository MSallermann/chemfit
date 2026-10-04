.. _mpi:

Running with MPI
================

:class:`~chemfit.mpi_scheduler.MPITreeScheduler` executes objective leaf tasks
on persistent MPI worker ranks. Rank zero expands parameter batches into leaf
tasks, distributes them round-robin across nonzero ranks, restores returned
context state, and performs composite reductions. The same scheduler works for
ordinary and combined objectives.

Requirements
------------

- a working MPI installation such as MPICH or Open MPI;
- ``mpi4py`` (install ChemFit with ``pip install "chemfit[mpi]"``).

Launch an MPI-aware script with, for example:

.. code-block:: bash

   mpiexec -n 4 python fit.py

Direct evaluation
-----------------

Every rank must construct the same objective and prepare it with an MPI
scheduler. Rank zero evaluates; every nonzero rank enters the prepared
schedule's worker loop:

.. code-block:: python

   from chemfit.abstract_objective_function import EvaluateContext
   from chemfit.combined_objective_function import CombinedObjectiveFunction
   from chemfit.mpi_scheduler import MPITreeScheduler

   terms = magic_from_elsewhere()
   objective = CombinedObjectiveFunction(terms)

   scheduler = MPITreeScheduler()
   with scheduler.prepare(objective) as schedule:
       if schedule.rank == 0:
           ctx = EvaluateContext()
           value = schedule.evaluate({"epsilon": 2.0, "sigma": 1.5}, ctx)
           print(value)
           print(ctx.meta["children"])
       else:
           schedule.worker_loop()

``worker_loop()`` takes no objective argument: the objective tree was bound by
``prepare``. Leaving the rank-zero context closes the schedule and sends a
shutdown request to every worker. Worker contexts return parameters, loss,
quantities, and metadata to rank zero; ``ctx.temp`` and ``ctx.shared`` are not
part of returned result state.

Fitting on rank zero
--------------------

Pass the scheduler to :class:`~chemfit.fitter.Fitter` on rank zero. Nonzero
ranks prepare the same objective and wait in ``worker_loop()``:

.. code-block:: python

   from chemfit.fitter import Fitter
   from chemfit.mpi_scheduler import MPITreeScheduler

   scheduler = MPITreeScheduler()

   if scheduler.comm.Get_rank() == 0:
       fitter = Fitter(
           objective,
           initial_params={"epsilon": 2.0, "sigma": 1.5},
           scheduler=scheduler,
       )
       optimum = fitter.fit_nevergrad(budget=100, num_workers=4)
       print(optimum)
   else:
       with scheduler.prepare(objective) as schedule:
           schedule.worker_loop()

Here ``num_workers`` controls how many candidates Nevergrad places in a batch;
it does not set the MPI world size. Start the desired number of ranks with
``mpiexec``. The prepared MPI schedule sends all leaf tasks from a candidate
batch to the available nonzero ranks.

Communicators and debugging
---------------------------

By default, the scheduler uses ``MPI.COMM_WORLD``. Supply another communicator
with ``MPITreeScheduler(comm=...)`` when ChemFit should operate on a subgroup.
All ranks in that communicator must follow the same coordinator/worker
lifecycle.

Set ``mpi_debug_log=True`` to wrap communicator methods with verbose logging:

.. code-block:: python

   scheduler = MPITreeScheduler(mpi_debug_log=True)

This is intended for diagnosing protocol problems and can produce substantial
output.

Failure behavior
----------------

An ordinary exception raised by an objective leaf is returned as that
evaluation's outcome. Combined objectives apply their configured exception
handler; a fitter either re-raises or replaces the failed evaluation according
to its ``swallow_exceptions`` setting.

A failure in worker or MPI machinery is catastrophic. Rank zero raises
:class:`~chemfit.mpi_scheduler.MPIWorkerError`, closes the prepared schedule,
and requests worker shutdown. A catastrophically failed schedule cannot be
reused.

Common pitfalls
---------------

Worker hang at the first evaluation
   Ensure every nonzero rank prepared the objective and entered
   ``schedule.worker_loop()`` before rank zero started evaluating.

Mismatched results or deserialization errors
   Construct the same objective structure and term order on every rank. Leaf
   objectives, hooks, parameter mappings, and context worker-input state must
   be serializable.

Hooks missing on workers
   Register recursive hooks while constructing the objective on every rank.
   Registration on rank zero does not modify worker-side objective instances.

Poor scaling
   Leaf tasks are assigned round-robin, without cost-aware placement. Large
   differences in leaf cost can leave ranks idle. MPI overhead can also
   dominate when individual leaves are cheap.

Single-rank launch
   With only rank zero in the communicator, the MPI schedule evaluates leaf
   tasks locally. At least two ranks are required for distributed execution.
