.. _parallel_execution:

Parallel Execution
====================

There are two ways in which parallel execution enters the picture while dealing with
a ChemFit objective function:

1. Evaluating the same objective function for different parameters in parallel.

2. Evaluating the terms of a :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction` in parallel.


This page is meant to showcase example code, making use of these forms of parallelism.


1. Evaluate parameter sets in parallel
-----------------------------------------

The main complication in this form of parallelism is that each evaluation carries state
(e.g. metadata, intermediate results). ChemFit's context system ensures that this state
is preserved and propagated correctly across parallel execution (see :ref:`concepts_parallel_eval`).

In practice, if you use the :py:class:`~chemfit.fitter.Fitter` class you won't have to explicitly interact with these nitty gritty details
(simply supply ``num_workers`` to :py:meth:`~chemfit.fitter.Fitter.fit_nevergrad`).

If you, nonetheless, find yourself in the situation of wanting to evaluate an objective function for multiple parameters in parallel, this will work:

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

    With a :py:class:`~concurrent.futures.ThreadPoolExecutor` this is usually not an issue,
    since execution happens in the same process, but a :py:class:`~concurrent.futures.ProcessPoolExecutor`
    on the other hand will only pickle the **result** of the function and send it back the main process.
    The :py:func:`~chemfit.executor_utils.map_with_context` function ensures that context updates
    are propagated correctly by including the context in the returned results.

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


2. Evaluate objective terms in parallel
------------------------------------------

A :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`
evaluates multiple terms for the same set of parameters.

If these terms are independent and expensive, it can make sense to evaluate them in parallel.

ChemFit provides two mechanisms for this:

- executor-based parallelism
- MPI-based parallelism


2.1 Executor-based parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use :py:class:`~chemfit.parallel_execution.ExecutorWrapperCOB` to evaluate terms
in parallel using an executor.

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor

   from chemfit.parallel_execution import ExecutorWrapperCOB
   from chemfit.abstract_objective_function import EvaluateContext

   executor = ThreadPoolExecutor(max_workers=4)

   wrapped = ExecutorWrapperCOB(objective, executor=executor)

   value = wrapped(parameters, EvaluateContext())

This is the simplest way to parallelize a combined objective.

Use this when:

- each term performs a non-trivial amount of work
- the overhead of the executor is small compared to the cost of each term

.. note::

   As in the previous section, the choice of executor matters.

   - :py:class:`~concurrent.futures.ThreadPoolExecutor` has low overhead, but is limited by the GIL
   - :py:class:`~concurrent.futures.ProcessPoolExecutor` allows true parallelism, but introduces serialization overhead


2.2 MPI-based parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use :py:class:`~chemfit.parallel_execution.MPIWrapperCOB` to distribute terms
across MPI processes.

.. code-block:: python

   from chemfit.parallel_execution import MPIWrapperCOB
   from chemfit.abstract_objective_function import EvaluateContext

   with MPIWrapperCOB(objective) as mpi:
       if mpi.rank == 0:
           value = mpi(parameters, EvaluateContext())
       else:
           mpi.worker_loop()

MPI does not behave like an executor.

One process drives the evaluation, while the others wait for work in a loop.

Use this when:

- you already run your code under MPI
- you have many small terms
- executor overhead becomes a bottleneck


Remarks
^^^^^^^^^

Parallelizing a combined objective only helps if the individual terms are sufficiently expensive.

If terms are cheap, the overhead of parallel execution will dominate and performance may degrade.
