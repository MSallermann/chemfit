.. _parallel_execution:

Parallel Execution
==================

There are two ways in which parallel execution enters the picture while dealing with
a ChemFit objective function:

1. Evaluating the same objective function for different parameters in parallel.

2. Evaluating the terms of a :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction` in parallel.


This page is meant to showcase example code, making use of these forms of parallelism.


1. Evaluate parameter sets in parallel
---------------------------------------

The main contribution ChemFit makes to enabling this form of parallelism is the context system (see :ref:`concepts_parallel_eval`).

In practice, if you use the :py:class:`~chemfit.fitter.Fitter` class you won't have to explicitly interact with these nitty gritty details
(simply supply ``num_workers`` to :py:meth:`~chemfit.fitter.Fitter.fit_nevergrad`).

If you, nonetheless, find yourself in the situation of wanting to evaluate on objective function for multiple parameters in parallel, this will work:

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor

   from chemfit.executor_utils import map_with_context
   from chemfit.abstract_objective_function import EvaluateContext

   executor = ThreadPoolExecutor(max_workers=4)

   params_list = [...]
   ctxs = [EvaluateContext() for _ in params_list]

   results = map_with_context(
       executor,
       objective,
       params_list,
       ctxs=ctxs,
   )

.. note::

    Why do we need :py:func:`~chemfit.executor_utils.map_with_context`?

    Yes, we would get the same results with the built-in ``map`` function of the ``executor``.
    The difference is that :py:func:`~chemfit.executor_utils.map_with_context` correctly propagates the side-effects
    of the function evaluation on the context.
    (With a ``ThreadPoolExecutor`` this distinction is meaningless, but for example a ``ProcessPoolExecutor``
    will only pickle the **result** of the function and send it back the main process).
    The little :py:func:`~chemfit.executor_utils.map_with_context` function helps us circumvent this little problem.
    It works by intermittently making the context a part of the result.

.. note::

    **Compute bound** pure python code (in non free-threading builds) will not be sped-up by using ``ThreadPoolExecutor``.
    The reason is the global interpreter lock (GIL).
    Generally it is recommended to avoids compute-heavy workloads in python...

    But if you really have to, you can speed up compute-bound python code by using a process pool.
    For example :py:class:`concurrent.futures.ProcessPoolExecutor` from the standard library.
    Be warned though that the required serialization can mean a significant overhead (always measure!).
    Furthermore, pickling certain functions can be tricky.

.. tip::

    The ``loky`` package provides a drop-in replacement for :py:class:`concurrent.futures.ProcessPoolExecutor`, which is able
    to pickle many more functions than the standard library version.


1. Parallelizing a combined objective
---------------------------------

If your objective is a combination of multiple terms, you can evaluate
those terms in parallel.

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor

   from chemfit.parallel_execution import ExecutorWrapperCOB

   executor = ThreadPoolExecutor(max_workers=4)
   objective = ExecutorWrapperCOB(objective, executor=executor)

   value = objective(params, ctx)

This only makes sense if:

- the objective is a CombinedObjectiveFunction
- each term actually does non-trivial work

If your terms are cheap, this will slow you down.

---

MPI: many small terms
---------------------

If you are already running under MPI and have many small terms, use the MPI wrapper.

.. code-block:: python

   from chemfit.parallel_execution import MPIWrapperCOB
   from chemfit.abstract_objective_function import EvaluateContext

   with MPIWrapperCOB(objective) as mpi:
       if mpi.rank == 0:
           value = mpi(params, EvaluateContext())
       else:
           mpi.worker_loop()

This is not just “another executor”.

MPI has a different execution model:

- one process drives the evaluation
- others wait for work in a loop
- communication cost is lower than Python executor overhead

Use this when:

- you already have MPI
- executor-based parallelism becomes the bottleneck

---

Running external programs (Slurm / srun)
----------------------------------------

This is where ChemFit is actually different from most libraries.

You do **not** parallelize inside Python.
You launch external jobs, and ChemFit keeps them organized.

The key abstraction is:

:class:`chemfit.file_based_computer.FileBasedQuantityComputer`

Each evaluation gets:

- its own working directory
- its own input/output files
- no shared state with other evaluations

That is what makes parallel execution safe.

---

Minimal srun integration
------------------------

You typically inject ``srun`` at command construction:

.. code-block:: python

   from chemfit.file_based_computer import FileBasedQuantityComputer

   class SrunComputer(FileBasedQuantityComputer):
       def build_cmd(self, parameters, ctx):
           srun = ctx.config.srun_spec.to_str()
           base_cmd = super().build_cmd(parameters, ctx)
           return [*srun, *base_cmd]

Then run as usual:

.. code-block:: python

   from chemfit.abstract_objective_function import EvaluateContext

   ctx = EvaluateContext()
   ctx.config.srun_spec = SrunSpec(...)

   value = objective(parameters, ctx)

Nothing special happens here. It is just a normal evaluation.

---

Parallel Slurm jobs
-------------------

This is where people usually get it wrong.

**Slurm does not give you parallelism here.**

This does *not* run things in parallel:

.. code-block:: bash

   srun simulation

It runs one job step.

If you want multiple jobs, you need multiple calls to ``srun``.

That is where the executor comes in:

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor

   from chemfit.executor_utils import map_with_context
   from chemfit.abstract_objective_function import EvaluateContext

   executor = ThreadPoolExecutor(max_workers=4)

   params_list = [...]
   ctxs = [EvaluateContext() for _ in params_list]

   for ctx in ctxs:
       ctx.config.srun_spec = my_srun_spec

   results = map_with_context(
       executor,
       objective,
       params_list,
       ctxs=ctxs,
   )

What happens:

- Python launches multiple ``srun`` commands
- Slurm schedules them
- each evaluation runs in its own working directory

No file collisions. No shared state. No hacks.

---

Choosing what to do
------------------

If you are unsure:

- many parameter sets → use ``map_with_context``
- expensive objective terms → use executor wrapper
- many small terms + MPI → use MPI wrapper
- external simulations → use file-based computers + executor

That’s it.

There is no hidden layer beyond this.

---

Summary
-------

ChemFit does not implement parallelism.

It gives you a clean way to **run the same computation under different execution models**:

- local threads or processes
- MPI
- Slurm / external programs

The objective stays the same. Only the execution changes.
