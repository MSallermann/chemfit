.. _combined_objective_functions:

Combined Objective Functions
============================

Use :func:`chemfit.combine() <chemfit.api.combine>` to combine objective terms
in ordinary workflows.
It returns a
:py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`,
which evaluates several terms for the same parameter set and reduces their
values to a single scalar loss.

This is the standard way to combine several fitting contributions into one
objective. Typical examples are fitting against several datasets, several
configurations, or several physical properties at once.

.. important::

   A combined objective is immutable after construction. Its objective terms
   and configuration must not be changed in place.

   To add, remove, replace, or reorder terms, construct a fresh combined
   objective with :func:`chemfit.combine() <chemfit.api.combine>`. To change
   supported configuration such as weights, reduction, exception
   handling, or metadata, use the fluent ``with_*`` API. Every fluent method
   returns a new configured object and leaves the original unchanged.

Basic idea
----------

A combined objective is built from a sequence of objective functions and a
reduction step.

Each term is evaluated with the same ``parameters`` dictionary. Before the
reduction is applied, each term value is multiplied by its corresponding weight.

In other words, the weighted term values are what enter the reduction.

With the default reduction this gives a weighted sum,

.. math::

   L(\mathrm{params}) = \sum_i w_i L_i(\mathrm{params})

but the reduction does not have to be a sum. It can be any callable with the
appropriate signature, as shown below.

A minimal example
-----------------

.. testcode::

   import chemfit

   def term1(params):
       return (params["x"] - 1.0) ** 2

   def term2(params):
       return (params["x"] - 3.0) ** 2

   objective = chemfit.combine(term1, term2, weights=[0.5, 1.0])

   loss = objective({"x": 2.0})
   print(loss) # 1.5

.. testoutput::
    :hide:

    1.5

This evaluates both terms for ``{"x": 2.0}``, multiplies them by the supplied
weights, and then sums the weighted values.

The returned object is a
:py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`.
The remainder of this page documents that object's complete framework API.

The terms do not need to be plain functions. Generic callables are accepted and
wrapped internally. In practice many terms are instances of
:py:class:`~chemfit.abstract_objective_function.ObjectiveComponent`, for example
objectives constructed from quantity computers.

Fluent configuration
--------------------

``chemfit.combine(...)`` constructs the combined objective. Common semantic
configuration can then be applied as a fluent chain:

.. code-block:: python

   objective = (
       chemfit.combine(term1, term2)
       .with_weights([1.0, 0.2])
       .with_exception_handler(chemfit.nan_exception_handler)
       .with_reduction(chemfit.mean_reducer)
   )

Each method returns a configured copy and leaves its source object unchanged.
Child objective and hook objects remain shared. ``with_exception_handler()``
changes how failed terms contribute to the final result. The sections below
describe those semantics in detail.

What actually happens during evaluation
---------------------------------------

Calling a combined objective does three things.

First, it creates one nested
:py:class:`~chemfit.abstract_objective_function.EvaluateContext` per term.

Second, direct calls evaluate every term serially in its own child context.
Prepared graph schedules use their configured backend for atomic work while
preserving the same context and reduction semantics.

Third, it filters skipped terms and applies the configured reduction. The final
result is stored in ``ctx.output``.

If you pass an explicit context, the child evaluations are directly available
through ``ctx.incoming``. That makes combined objectives useful not just for
optimization but also for inspection and debugging.

.. testcode::

    from chemfit.abstract_objective_function import EvaluateContext
    from chemfit.combined_objective_function import CombinedObjectiveFunction
    from chemfit import objective

    @objective()
    def term(params, a):
        return (params["x"] - a) ** 2

    term1 = term.bind(a=1)
    term2 = term.bind(a=3)

    objective = CombinedObjectiveFunction([term1, term2], weights=[0.5, 1.0])

    ctx = EvaluateContext()
    loss = objective({"x": 2.0}, ctx)

    print(loss) # 1.5
    print(ctx.output) # 1.5
    print([child.output for child in ctx.incoming]) # child losses

.. testoutput::
    :hide:

    1.5
    1.5
    [1.0, 1.0]

Writing a custom reducer
------------------------

The simplest customization point is the reduction function.

A simple reducer receives the list of weighted term values and returns a scalar:

.. testcode::

    def my_reducer(terms):
        return max(terms)

    objective = CombinedObjectiveFunction(
        [term1, term2],
        weights=[1.0, 1.0],
        reduction=my_reducer,
    )

    print(objective({"x": 2.0})) # 1.0

.. testoutput::
    :hide:

    1.0

The library already provides a few simple reducers in
:py:mod:`chemfit.combined_objective_function`:

.. code-block:: python

    from chemfit.combined_objective_function import (
        CombinedObjectiveFunction,
        sum_reducer,
        mean_reducer,
        root_mean_reducer,
    )

    objective = CombinedObjectiveFunction(
        [term1, term2, term3],
        reduction=root_mean_reducer,
    )

    print(objective({"x" : 2.0})) # 1.0

.. testoutput::
    :hide:

    1.0

The important detail is that the reducer sees the weighted term values, not the
raw term outputs.

That means these two are equivalent:

.. testcode::

    from chemfit.combined_objective_function import sum_reducer

    objective = CombinedObjectiveFunction(
        [term1, term2],
        weights=[0.5, 2.0],
        reduction=sum_reducer,
    )

    print(objective({"x" : 2.0})) # 2.5

.. testoutput::
    :hide:

    2.5

.. testcode::

   def manual_sum(terms):
       return 0.5*terms[0] + 2.0*terms[1] + sum(terms[2:])

   objective = CombinedObjectiveFunction(
       [term1, term2],
       reduction=manual_sum,
   )

   print(objective({"x" : 2.0})) # 2.5

.. testoutput::
    :hide:

    2.5

Simple and context-aware reducers
---------------------------------

A simple reducer has the signature

.. code-block:: python

   def reducer(terms: list[float]) -> float:
       ...

A context-aware reducer has the richer signature

.. code-block:: python

   def reducer(terms: list[float], *, ctx: EvaluateContext) -> float:
       ...

Pass either form with ``reduction=`` or ``with_reduction()``. ChemFit detects
whether the callable accepts the keyword-only ``ctx`` argument. A
context-aware reducer can inspect the incoming term contexts and write
additional information into the combined context.

This is the right tool when the final loss should depend on more than the term
values alone.

Writing a context-aware reducer
-------------------------------

Context-aware reducers are useful when the terms are built from quantity
components.

In that case each incoming term context contains its pipeline stage contexts,
so the reducer can inspect both the scalar term values and intermediate
quantities.

The following example is based on the actual test setup.

.. testcode::

   from chemfit.abstract_objective_function import EvaluateContext
   from chemfit.combined_objective_function import CombinedObjectiveFunction
   from chemfit import quantity

   def custom_reducer(terms, *, ctx):
       ctx.meta["foo"] = "bar"
       quantities = [term_ctx.pipeline[0].output for term_ctx in ctx.incoming]
       return sum(q["test"] for q in quantities)

   @quantity()
   def q1(parameters, f):
       return {"test": f * parameters["x"] + parameters["y"]}

   objective = CombinedObjectiveFunction(
       [
           q1.bind(f=1).with_loss(lambda q: 0.0),
           q1.bind(f=2).with_loss(lambda q: 0.0),
       ],
       reduction=custom_reducer,
   )

   ctx = EvaluateContext()
   loss = objective({"x": 2.0, "y": 1.0}, ctx)

   print(loss)            # 3.0 + 5.0 = 8.0
   print(ctx.meta["foo"]) # "bar"

.. testoutput::
    :hide:

    8.0
    bar

This example is worth looking at carefully.

The two terms contribute zero loss individually, since both use
``with_loss(lambda q: 0.0)``. The final loss is therefore not coming from the
terms at all. Instead, the reducer computes the loss from the collected
quantities.

The keyword-only ``ctx`` argument is what distinguishes this form from a
simple reducer. It provides access to the weighted values, nested contexts,
and the combined context's metadata.

Exception handling
------------------

Combined objectives also let you control what happens when one of the terms
raises an exception.

The ``exception_handler`` is called with

.. code-block:: python

   exception_handler(exception, ctx, idx)

where ``idx`` is the term index and ``ctx`` is the child context for that term.

Three useful handlers are provided in
:py:mod:`chemfit.combined_objective_function`.

The default handler simply re-raises the exception. It can also be selected
explicitly in a fluent chain:

.. code-block:: python

   objective = (
       chemfit.combine(term1, term2)
       .with_exception_handler(chemfit.raising_exception_handler)
   )

Returning ``math.nan`` marks the whole reduction as invalid in the usual way:

.. testcode::

   import math

   def ok(params):
       return 1.0

   def broken(params):
       raise RuntimeError("Whoops")

   objective = chemfit.combine(ok, broken).with_exception_handler(
       chemfit.nan_exception_handler
   )

   ctx = chemfit.EvaluateContext()
   loss = objective({}, ctx)

   print(math.isnan(loss))     # True
   print(math.isnan(ctx.output)) # True

.. testoutput::
    :hide:

    True
    True

Returning ``None`` skips the failed term entirely:

.. testcode::

   def ok(params):
       return 1.0

   def broken(params):
       raise RuntimeError("Whoops")

   objective = chemfit.combine(ok, broken).with_exception_handler(
       chemfit.skip_exception_handler
   )

   ctx = chemfit.EvaluateContext()
   loss = objective({}, ctx)

   print(loss)                       # 1.0
   print(ctx.meta["skipped_indices"])  # [1]

.. testoutput::
    :hide:

    1.0
    [1]

Terms for which the exception handler returns ``None`` are removed before the
reduction is applied, and their indices are stored in
``ctx.meta["skipped_indices"]``.

Writing your own exception handler is straightforward:

.. testcode::

    def my_exception_handler(exception, ctx, idx):
        ctx.meta["last_failure"] = {
            "idx": idx,
            "message": str(exception),
        }
        return None

    objective = chemfit.combine(term1, broken).with_exception_handler(
        my_exception_handler
    )

    ctx = chemfit.EvaluateContext()
    print(objective({"x" : 2.0}, ctx))
    print(ctx.incoming[1].meta["last_failure"]) # {'idx': 1, 'message': 'Whoops'}

.. testoutput::
    :hide:

    1.0
    {'idx': 1, 'message': 'Whoops'}

.. note::

    The exception handler sees *only* the child context.

Using context configurators
---------------------------

Every component supports ``with_ctx_configurator()``. The configurator runs
when that component's context begins evaluation and has the signature

.. code-block:: python

   def configurator(ctx):
       ...

A combined objective can therefore configure its terms independently:

.. testcode::

    from chemfit.abstract_objective_function import EvaluateContext
    from chemfit.combined_objective_function import CombinedObjectiveFunction

    def with_term_number(term, number):
        def configure(ctx):
            ctx.meta["term_number"] = number

        return term.with_ctx_configurator(configure)

    objective = CombinedObjectiveFunction(
        [with_term_number(term1, 0), with_term_number(term2, 1)],
    )

    ctx = EvaluateContext()
    objective({"x": 2.0}, ctx)

    print([term_ctx.meta["term_number"] for term_ctx in ctx.incoming])

.. testoutput::

    [0, 1]

This is useful for term-specific metadata or configuration. A configurator
should modify only the context it receives; sibling and combined contexts are
separate evaluation scopes.

Parallel execution
------------------

Combined objectives are the main place where term-level parallelism makes sense.

Calling a combined objective directly evaluates its terms serially. To use a
different backend, prepare the objective with a scheduler and evaluate through
the returned schedule:

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor

   from chemfit.executor_graph_schedule import ExecutorGraphScheduler

   with ThreadPoolExecutor(max_workers=4) as executor:
       scheduler = ExecutorGraphScheduler(executor=executor)
       with scheduler.prepare(objective) as schedule:
           value = schedule(parameters)

In practice this means the same combined objective can be used

- serially
- with :py:class:`~chemfit.executor_graph_schedule.ExecutorGraphScheduler`
- with :py:class:`~chemfit.mpi_graph_schedule.MPIGraphScheduler`

without changing the terms themselves. See :ref:`parallel_execution` for
batch, executor, and MPI examples.
