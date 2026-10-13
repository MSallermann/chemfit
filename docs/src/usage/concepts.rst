.. _concepts:

################
Concepts
################

This page introduces the core concepts behind ChemFit and explains how
objective functions are structured and evaluated.

The goal is to provide a mental model that helps you design your own
objective functions and understand how ChemFit executes them.

Most workflows begin with top-level helpers such as
:func:`chemfit.quantity() <chemfit.abstract_objective_function.quantity>`,
:func:`chemfit.combine() <chemfit.api.combine>`, and
:func:`chemfit.fit_nevergrad() <chemfit.api.fit_nevergrad>`.
Those helpers operate on the objects described here: ``quantity()`` creates a
:class:`~chemfit.abstract_objective_function.WrappedComponent` that satisfies
the ``QuantityComputer`` typing alias, ``combine()`` returns a
:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`, and
``fit_nevergrad()`` drives them through :class:`~chemfit.fitter.Fitter` and a prepared
:class:`~chemfit.scheduling.Scheduler`. This page explains that underlying
machinery rather than replacing it with convenience-function terminology.

-------------------------

Objective functions as compositions
=====================================

In ChemFit, an objective function is built from small, composable pieces.

The basic building blocks are:

- A quantity computer—an
  :class:`~chemfit.abstract_objective_function.ObjectiveComponent` matching the
  ``QuantityComputer`` typing alias—which computes intermediate quantities
  from a parameter dictionary.
- A loss function, which maps these quantities to a scalar value.

These two are combined into an :class:`~chemfit.abstract_objective_function.ObjectiveComponent`.

More complex objectives are constructed by combining multiple such terms using
:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`.

Conceptually, this looks like:

.. code-block:: text

    parameters
        ↓
    QuantityComputer
        ↓
    ObjectiveComponent (loss)
        ↓
    CombinedObjectiveFunction
        ↓
    total loss

This structure allows complex objectives to be built from many simple,
independent contributions.

-------------------------

.. _concepts_parallel_eval:

Independent terms and parallelism
==================================

A key design principle of ChemFit is that objective terms are evaluated
independently.

When using :class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`,
each term is evaluated in isolation and contributes a scalar value to the final loss.

This enables parallel evaluation:

- Different terms can be evaluated concurrently
- No shared mutable state is required
- The same objective definition works for serial, threaded, or MPI execution

As a result, parallelization is built into the structure of the objective function,
rather than being added on top.

Execution backends are selected with a
:class:`~chemfit.scheduling.Scheduler`. A scheduler prepares an
objective-specific schedule. The prepared schedule is callable for one
parameter mapping and provides ``evaluate_many`` for batches. Serial,
executor-backed, and MPI graph schedulers all operate on the same objective
definition; see :ref:`parallel_execution`.

-------------------------

Evaluation contexts
====================

ChemFit uses an :class:`~chemfit.evaluate_context.EvaluateContext`
to store all per-evaluation state.

A context is created for each evaluation and passed through the computation.

It stores the component input, finalized output, metadata, and the context
links created by structural compositions. A failed component stores its
exception in ``ctx.output``; there is no separate exception field. A quantity
component's successful output is its quantities dictionary, while a complete
objective's successful output is its scalar loss.

Each evaluation has its own context, which avoids shared mutable state and
makes parallel execution safe.

In simple use cases, you do not need to interact with the context explicitly.
However, it becomes useful when you want to inspect results or record additional
information.

Example
---------

The context can be used to record additional information during evaluation:

.. testcode::

    from chemfit import quantity
    from chemfit.abstract_objective_function import EvaluateContext

    @quantity(pass_ctx=True)
    def computer(params, *, ctx):
        value = params["x"] ** 2
        ctx.meta["x_squared"] = value
        return {"x2": value}

    ctx = EvaluateContext()
    computer({"x": 3.0}, ctx)

    print(ctx.output)      # {'x2': 9.0}
    print(ctx.meta)        # {'x_squared': 9.0}

.. testoutput::
    :hide:

    {'x2': 9.0}
    {'x_squared': 9.0}

---------------------------

Context graphs
==============

Structural components create a context graph mirroring their data flow.

- The top-level evaluation uses the context supplied by the caller.
- Each nested component receives its own context.
- ``incoming`` and ``outgoing`` expose dependency links without allowing the
  topology to be mutated through those properties.

Conceptually:

.. code-block:: text

    term 1 context ──┐
                     ├── combined-objective context
    term 2 context ──┘

Each incoming context stores its component's input, finalized output, and
metadata. A failed component's output is the exception object. Pipelines
additionally expose their ordered stage contexts through ``pipeline``. This
makes nested results inspectable without flattening or copying metadata into
the root context.

-------------------------

Metadata
==========

Each context contains a ``meta`` dictionary that can be used to store
additional information during evaluation.

Typical uses include:

- recording intermediate results
- storing diagnostic information
- tracking which terms contributed to the final loss

Metadata stays on the context whose component produced it. Inspect
``ctx.incoming`` or ``ctx.pipeline`` when nested component metadata is needed.

Static metadata belongs to a reusable computation or objective definition. Add
it fluently with
:py:meth:`~chemfit.abstract_objective_function.ObjectiveComponent.with_meta`:

.. code-block:: python

   term = (
       chemfit.ase_quantity(atoms)
       .with_meta(dataset="liquid", temperature=298)
       .with_calculator(make_calculator)
       .with_loss(loss, target=0.997)
       .with_meta(observable="density")
   )

``with_meta()`` returns a shallow configured copy, which makes it safe to
branch from one base definition. Metadata belongs to the component on which it
was configured and is merged into that component's context after ``_run()``
and before post-hooks. Thus quantity-computer metadata in this example is
stored on ``ctx.pipeline[0].meta`` and the metadata added after ``with_loss()``
is stored on the root ``ctx.meta``. The scopes are not flattened, so identical
keys at the two levels do not overwrite one another.

Values discovered during a particular evaluation are dynamic metadata and
should be written directly to ``ctx.meta`` by the evaluating callable or a
hook.

-------------------------

Configuration and shared state
===============================

Contexts provide two namespaces for passing information:

``config``
    A per-context namespace that is copied when nested contexts are spawned.
    This is useful for passing configuration or control information
    to nested evaluations.

``shared``
    A namespace shared by related contexts during local execution. Process
    executors serialize worker-input state and do not return worker mutations
    to the driver.

In most cases, you will not need to use these directly. They are primarily
intended for advanced use cases and custom objective implementations.


-------------------------


Parallel evaluations require separate contexts
===============================================

An :class:`~chemfit.evaluate_context.EvaluateContext` stores
the state of exactly one evaluation.

This means that if the same objective function is evaluated for multiple
parameter dictionaries in parallel, each evaluation must use its own
context.

Conceptually:

.. code-block:: text

    objective(params_1, ctx_1)
    objective(params_2, ctx_2)
    objective(params_3, ctx_3)

Using one context per evaluation avoids shared mutable state and makes
parallel evaluation safe.

This is distinct from the linked contexts created inside structural
components: those contexts represent different nodes of one evaluation,
whereas the contexts above represent separate root evaluations.


Design principles
==================

ChemFit is built around a few key ideas:

- **Composition over monoliths**
  Complex objectives are assembled from small, independent pieces.

- **Explicit data flow**
  All evaluation state is passed through contexts, avoiding hidden global state.

- **Separation of concerns**
  Quantity computation and loss evaluation are kept separate.

These principles make ChemFit flexible, extensible, and well-suited for
large-scale fitting problems.
