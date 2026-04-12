.. _combined_objective_functions:

Combined Objective Functions
============================

A :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`
evaluates several objective terms for the same parameter set and then reduces
the resulting term values to a single scalar loss.

This is the standard way to combine several fitting contributions into one
objective. Typical examples are fitting against several datasets, several
configurations, or several physical properties at once.

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

   from chemfit.combined_objective_function import CombinedObjectiveFunction

   def term1(params):
       return (params["x"] - 1.0) ** 2

   def term2(params):
       return (params["x"] - 3.0) ** 2

   objective = CombinedObjectiveFunction(
       objective_functions=[term1, term2],
       weights=[0.5, 1.0],
   )

   loss = objective({"x": 2.0})
   print(loss) # 1.5

.. testoutput::
    :hide:

    1.5

This evaluates both terms for ``{"x": 2.0}``, multiplies them by the supplied
weights, and then sums the weighted values.

The terms do not need to be plain functions. Generic callables are accepted and
wrapped internally. In practice many terms are instances of
:py:class:`~chemfit.abstract_objective_function.ObjectiveFunctor`, for example
objectives constructed from quantity computers.

What actually happens during evaluation
---------------------------------------

Calling a combined objective does three things.

First, it spawns one child :py:class:`~chemfit.abstract_objective_function.EvaluateContext`
per term.

Second, it evaluates every term in its own child context. The per-term method
that does this is
:py:meth:`~chemfit.combined_objective_function.CombinedObjectiveFunction.evaluate_term`.

Third, it filters skipped terms and applies the configured reduction. The final
result is stored in ``ctx.loss``.

If you pass an explicit context, the parent context ends up containing a record
of the child evaluations in ``ctx.meta["children"]``. That makes combined
objectives useful not just for optimization but also for inspection and
debugging.

.. testcode::

    from chemfit.abstract_objective_function import EvaluateContext
    from chemfit.combined_objective_function import CombinedObjectiveFunction
    from chemfit.wrap_funcs import to_objective_functor

    @to_objective_functor()
    def term(params, a):
        return (params["x"] - a) ** 2

    term1 = term.bind(a=1)
    term2 = term.bind(a=3)

    objective = CombinedObjectiveFunction([term1, term2], weights=[0.5, 1.0])

    ctx = EvaluateContext()
    loss = objective({"x": 2.0}, ctx)

    print(loss) # 1.5
    print(ctx.loss) # 1.5
    print(ctx.meta["children"]) # child meta data

.. testoutput::
    :hide:

    1.5
    1.5
    [{'quantities': None, 'parameters': {'x': 2.0}, 'loss': 1.0, 'meta': {}}, {'quantities': None, 'parameters': {'x': 2.0}, 'loss': 1.0, 'meta': {}}]

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

Reducer vs aggregator
---------------------

There are two supported reduction interfaces.

A simple reducer has the signature

.. code-block:: python

   def reducer(terms: list[float]) -> float:
       ...

An aggregator has the richer signature

.. code-block:: python

   def aggregator(
       terms: list[float],
       quantities: list[dict[str, object]],
       ctx : EvaluateContext,
   ) -> float:
       ...

If ChemFit sees that the reduction callable takes only one argument, it treats it
as a simple reducer. If it takes three arguments, it treats it as an aggregator.

The difference is that an aggregator can inspect both the term values and the
child quantities, and it can write additional information into the parent context.

This is the right tool when the final loss should depend on more than the term
values alone.

Writing an aggregator
---------------------

Aggregators become useful when the terms are built from quantity computers.

In that case each child context may contain quantities produced during the term
evaluation, and the aggregator can use them.

The following example is based on the actual test setup.

.. testcode::

   from chemfit.abstract_objective_function import EvaluateContext
   from chemfit.combined_objective_function import CombinedObjectiveFunction
   from chemfit.wrap_funcs import to_quantity_computer

   def custom_aggregator(terms, quantities, ctx):
       ctx.meta["foo"] = "bar"
       return sum(q["test"] for q in quantities)

   @to_quantity_computer()
   def q1(parameters, f):
       return {"test": f * parameters["x"] + parameters["y"]}

   objective = CombinedObjectiveFunction(
       [
           q1.bind(f=1).with_loss(lambda q: 0.0),
           q1.bind(f=2).with_loss(lambda q: 0.0),
       ],
       reduction=custom_aggregator,
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
terms at all. Instead, the aggregator computes the loss from the collected
quantities.

That is exactly the difference between a reducer and an aggregator. A reducer
only sees the weighted term values. An aggregator sees the weighted term values,
the child quantities, and the parent context.

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

The default handler simply re-raises the exception:

.. code-block:: python

   from chemfit.combined_objective_function import (
       CombinedObjectiveFunction,
       raising_exception_handler,
   )

   objective = CombinedObjectiveFunction([term1, term2])
   objective.exception_handler = raising_exception_handler

Returning ``math.nan`` marks the whole reduction as invalid in the usual way:

.. testcode::

   import math

   from chemfit.abstract_objective_function import EvaluateContext
   from chemfit.combined_objective_function import (
       CombinedObjectiveFunction,
       nan_exception_handler,
   )

   def ok(params):
       return 1.0

   def broken(params):
       raise RuntimeError("Whoops")

   objective = CombinedObjectiveFunction([ok, broken])
   objective.exception_handler = nan_exception_handler

   ctx = EvaluateContext()
   loss = objective({}, ctx)

   print(math.isnan(loss))     # True
   print(math.isnan(ctx.loss)) # True

.. testoutput::
    :hide:

    True
    True

Returning ``None`` skips the failed term entirely:

.. testcode::

   from chemfit.abstract_objective_function import EvaluateContext
   from chemfit.combined_objective_function import (
       CombinedObjectiveFunction,
       skip_exception_handler,
   )

   def ok(params):
       return 1.0

   def broken(params):
       raise RuntimeError("Whoops")

   objective = CombinedObjectiveFunction([ok, broken])
   objective.exception_handler = skip_exception_handler

   ctx = EvaluateContext()
   loss = objective({}, ctx)

   print(loss)                       # 1.0
   print(ctx.meta["skipped_indices"])  # [1]

.. testoutput::
    :hide:

    1.0
    [1]

This skip behavior is implemented through
:py:meth:`~chemfit.combined_objective_function.CombinedObjectiveFunction.filter_terms`.
Terms for which the exception handler returns ``None`` are removed before the
reduction is applied.

Writing your own exception handler is straightforward:

.. testcode::

    def my_exception_handler(exception, ctx, idx):
        ctx.meta["last_failure"] = {
            "idx": idx,
            "message": str(exception),
        }
        return None

    objective = CombinedObjectiveFunction(
        [term1, broken],
        exception_handler=my_exception_handler,
    )

    ctx = EvaluateContext()
    print(objective({"x" : 2.0}, ctx))
    print(ctx.meta["children"][1]["meta"]["last_failure"]) # {'idx': 1, 'message': 'Whoops'}

.. testoutput::
    :hide:

    1.0
    {'idx': 1, 'message': 'Whoops'}

.. note::

    The exception handler sees *only* the child context.

Using a child context configurator
----------------------------------

Each term gets its own child context. Sometimes those child contexts need to be
configured before evaluation starts.

That is what ``child_context_configurator`` is for.

The configurator has the signature

.. code-block:: python

   def configurator(idx_child_ctx, child_ctx, num_children, parent_ctx):
       ...

A small example:

.. testcode::

    from chemfit.abstract_objective_function import EvaluateContext
    from chemfit.combined_objective_function import CombinedObjectiveFunction

    def configurator(idx_child_ctx, child_ctx, num_children, parent_ctx):
        child_ctx.meta["configurator_number"] = idx_child_ctx + num_children

    objective = CombinedObjectiveFunction(
        [term1, term2],
        child_context_configurator=configurator,
    )

    ctx = EvaluateContext()
    objective({"x": 2.0}, ctx)

    print([child["meta"]["configurator_number"] for child in ctx.meta["children"]])

.. testoutput::

    [2, 3]

This is mostly useful when the terms need slightly different evaluation setup.

For example, the configurator can assign different metadata, attach child-local
configuration in ``child_ctx.config``, or prepare term-specific execution state.
The parent context is also available, so the configurator can derive the
child-specific setup from information stored at the parent level.

Adding terms after construction
-------------------------------

Combined objectives can be extended in place with
:py:meth:`~chemfit.combined_objective_function.CombinedObjectiveFunction.add`.

.. code-block:: python

   from chemfit.combined_objective_function import CombinedObjectiveFunction

   objective = CombinedObjectiveFunction([term1], weights=[1.0])

   objective.add(term2, weights=0.5)
   objective.add([term3, term4], weights=[2.0, 3.0])

This mutates the existing combined objective and returns it again.

The same weight rules apply as in the constructor. A single weight is broadcast
to all added terms, while a sequence of weights must match the number of added
terms.

Flattening several combined objectives
--------------------------------------

If you already have several combined objectives and want to combine them into
one flat object, use
:py:meth:`~chemfit.combined_objective_function.CombinedObjectiveFunction.add_flat`.

.. code-block:: python

   from chemfit.combined_objective_function import CombinedObjectiveFunction

   cob1 = CombinedObjectiveFunction([term1, term2], weights=[1.0, 2.0])
   cob2 = CombinedObjectiveFunction([term3], weights=[0.5])

   flat = CombinedObjectiveFunction.add_flat(
       [cob1, cob2],
       weights=[1.0, 10.0],
   )

In this example, the terms of ``cob2`` are included with their internal weights
scaled by ``10.0``.

.. warning::

    One detail matters here: ``add_flat`` only flattens terms and weights. It does
    not preserve the execution policy or other custom behavior of the input objects.
    The returned object is a fresh combined objective using the class defaults unless
    you reconfigure it afterward.

Parallel execution
------------------

Combined objectives are the main place where term-level parallelism makes sense.

Serial evaluation calls
:py:meth:`~chemfit.combined_objective_function.CombinedObjectiveFunction.evaluate_term`
for each term in turn.

Parallel execution uses the same per-term interface but schedules the term
evaluations differently. See :ref:`parallel_execution`.

In practice this means the same combined objective can be used

- serially
- with :py:class:`~chemfit.executor_wrapper_cob.ExecutorWrapperCOB`
- with :py:class:`~chemfit.mpi_wrapper_cob.MPIWrapperCOB`

without changing the terms themselves.
