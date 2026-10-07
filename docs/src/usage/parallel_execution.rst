.. _parallel_execution:

Parallel execution
==================

ChemFit can overlap work at two related levels:

1. several optimizer candidates can be evaluated in one batch;
2. leaf terms of a combined objective can run concurrently.

Tree schedulers handle both levels with the same pool of execution slots. A
batch of candidates is expanded into its reachable objective leaves, and the
backend executes those leaves serially, through an executor, or on MPI worker
ranks.

One-shot fitting
----------------

The top-level :py:func:`chemfit.api.fit_nevergrad` function separates optimizer
concurrency from execution concurrency:

.. code-block:: python

   result = chemfit.fit_nevergrad(
       objective,
       initial={"x": 1.0},
       budget=100,
       workers=4,
       execution_workers=8,
   )

``workers`` is the maximum number of candidates Nevergrad asks for in one
batch. ``execution_workers`` is the number of threads in the built-in
executor-backed scheduler; it defaults to ``workers``. Those execution slots
are shared by all leaf terms from all candidates in the batch.

To choose a different executor, pass a caller-owned executor and omit
``execution_workers``:

.. code-block:: python

   import loky

   with loky.ProcessPoolExecutor(max_workers=4) as executor:
       result = chemfit.fit_nevergrad(
           objective,
           initial={"x": 1.0},
           budget=100,
           workers=4,
           executor=executor,
       )

The executor controls leaf-task concurrency; ``workers`` still controls only
Nevergrad's candidate batch size. Pass ``scheduler=`` instead for complete
backend control. ``executor`` and ``scheduler`` are mutually exclusive.

Evaluating a parameter batch
----------------------------

:py:func:`chemfit.api.evaluate_many` evaluates parameter mappings through an
executor or scheduler and returns their populated contexts in input order:

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor

   from chemfit import evaluate_many

   parameters = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]

   with ThreadPoolExecutor(max_workers=4) as executor:
       contexts = evaluate_many(objective, parameters, executor=executor)

   losses = [ctx.loss for ctx in contexts]

The function creates one
:class:`~chemfit.abstract_objective_function.EvaluateContext` per parameter
mapping. It restores input order even when evaluations finish out of order and
re-raises an exception from a failed evaluation.

Preparing a scheduler directly
------------------------------

The lower-level scheduler API is useful when a schedule should be reused or
when individual results should be consumed as they complete. A scheduler is
backend configuration; :meth:`~chemfit.scheduling.Scheduler.prepare` binds it
to one objective and returns a prepared schedule.

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor

   from chemfit.abstract_objective_function import EvaluateContext
   from chemfit.executor_scheduler import ExecutorTreeScheduler
   from chemfit.scheduling import EvaluationRequest

   requests = [
       EvaluationRequest(params, EvaluateContext())
       for params in parameters
   ]

   with ThreadPoolExecutor(max_workers=4) as executor:
       scheduler = ExecutorTreeScheduler(executor=executor)
       with scheduler.prepare(objective) as schedule:
           completed = list(schedule.evaluate_many(requests))

   completed.sort(key=lambda result: result.index)

``evaluate_many`` yields
:class:`~chemfit.scheduling.EvaluationResult` objects in completion order.
Each result contains its original request index and either a numerical value or
an ordinary evaluation exception. A prepared schedule is also callable:
``schedule(parameters, ctx)`` handles one parameter mapping synchronously and
raises its evaluation exception directly. The context is optional when its
result state is not needed.

Executor ownership
~~~~~~~~~~~~~~~~~~

An executor supplied with ``executor=`` remains owned by the caller and is not
shut down when the prepared schedule closes. Alternatively, configure an
executor factory:

.. code-block:: python

   from functools import partial

   scheduler = ExecutorTreeScheduler(
       executor_factory=partial(ThreadPoolExecutor, max_workers=4),
   )

Every call to ``prepare`` then creates a new executor. Its prepared schedule
owns that executor and shuts it down on ``close`` or when its context manager
exits. Exactly one of ``executor`` and ``executor_factory`` is required.

Threads and processes
~~~~~~~~~~~~~~~~~~~~~

:class:`concurrent.futures.ThreadPoolExecutor` has low overhead and works well
for external programs and native libraries that release the GIL. CPU-bound
Python code normally needs a process executor for actual parallelism. Process
execution adds serialization overhead, and objectives, hooks, parameters, and
worker input context state must be serializable.

Executor and MPI schedules return result-bearing context state to the driver:
parameters, loss, quantities, and metadata. ``ctx.temp`` is scratch state and
``ctx.shared`` is not transported back. Store results that must survive worker
execution in ``ctx.meta`` or in the returned quantities.

Fitter batching and execution
-----------------------------

At the lower level,
:meth:`~chemfit.fitter.Fitter.fit_nevergrad` uses ``batch_size`` only as the
maximum candidate batch size. The prepared schedule passed to
:class:`~chemfit.fitter.Fitter` determines whether and how that batch executes
concurrently:

.. code-block:: python

   from functools import partial

   from concurrent.futures import ThreadPoolExecutor
   from chemfit.executor_scheduler import ExecutorTreeScheduler
   from chemfit.fitter import Fitter

   scheduler = ExecutorTreeScheduler(
       executor_factory=partial(ThreadPoolExecutor, max_workers=8),
   )
   with scheduler.prepare(objective) as schedule:
       fitter = Fitter(schedule, initial_params={"x": 1.0})
       optimum = fitter.fit_nevergrad(budget=100, batch_size=4)

A schedule prepared by
:class:`~chemfit.tree_schedule.SerialTreeScheduler` evaluates a Nevergrad
batch serially even when ``batch_size`` is greater than one.

MPI
---

:class:`~chemfit.mpi_scheduler.MPITreeScheduler` uses persistent nonzero MPI
ranks as leaf-task workers while rank zero coordinates objective-tree
semantics and fitting. MPI programs must enter the worker loop on every
nonzero rank. See :ref:`mpi` for the complete lifecycle and launch example.

Practical guidance
------------------

Parallel execution is useful only when leaf work is large enough to outweigh
scheduling and serialization overhead. Measure representative workloads.
Avoid mutable global or objective-instance state, allocate a distinct context
for every overlapping evaluation, and use a process backend only when all
worker inputs are serializable.
