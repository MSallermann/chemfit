.. _concepts:

################
Concepts
################

This page introduces the core concepts behind ChemFit and explains how
objective functions are structured and evaluated.

The goal is to provide a mental model that helps you design your own
objective functions and understand how ChemFit executes them.

-------------------------

Objective functions as compositions
=====================================

In ChemFit, an objective function is built from small, composable pieces.

The basic building blocks are:

- A :class:`~chemfit.abstract_objective_function.QuantityComputer`,
  which computes intermediate quantities from a parameter dictionary.
- A loss function, which maps these quantities to a scalar value.

These two are combined into an :class:`~chemfit.abstract_objective_function.ObjectiveFunctor`.

More complex objectives are constructed by combining multiple such terms using
:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`.

Conceptually, this looks like:

.. code-block:: text

    parameters
        ↓
    QuantityComputer
        ↓
    ObjectiveFunctor (loss)
        ↓
    CombinedObjectiveFunction
        ↓
    total loss

This structure allows complex objectives to be built from many simple,
independent contributions.

.. note::

    For advanced use-cases, combined objective functions can compute their loss
    from the quantities of individual child terms, instead of simple scalar losses.
    This is achieved by supplying a custom :class:`~chemfit.combined_objective_function.Aggregator`.

-------------------------

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

The execution model is defined by the objective function itself.
Parallel execution is provided by wrapper implementations, which
evaluate the same objective structure using different execution backends
(e.g. threads or MPI).

-------------------------

Evaluation contexts
====================

ChemFit uses an :class:`~chemfit.abstract_objective_function.EvaluateContext`
to store all per-evaluation state.

A context is created for each evaluation and passed through the computation.

It serves several purposes:

- storing input parameters
- recording computed quantities
- storing the final loss
- collecting metadata during evaluation

Each evaluation has its own context, which avoids shared mutable state and
makes parallel execution safe.

In simple use cases, you do not need to interact with the context explicitly.
However, it becomes useful when you want to inspect results or record additional
information.

Example
---------

The context can be used to record additional information during evaluation:

.. testcode::

    from chemfit.wrap_funcs import to_quantity_computer
    from chemfit.abstract_objective_function import EvaluateContext

    @to_quantity_computer(pass_ctx=True)
    def computer(params, ctx):
        value = params["x"] ** 2
        ctx.meta["x_squared"] = value
        return {"x2": value}

    ctx = EvaluateContext()
    computer({"x": 3.0}, ctx)

    print(ctx.quantities)  # {'x2': 9}
    print(ctx.meta)        # {'x_squared': 9}

.. testoutput::
    :hide:

    {'x2': 9.0}
    {'x_squared': 9.0}

---------------------------

Parent and child contexts
===========================

When combining multiple objective terms, ChemFit creates a hierarchy of contexts.

- The top-level evaluation uses a *parent context*
- Each objective term is evaluated in its own *child context*

Conceptually:

.. code-block:: text

    parent context
        ├── child context (term 1)
        ├── child context (term 2)
        └── ...

Each child context stores the results of its corresponding term.

After evaluation, relevant information from child contexts can be collected
into the parent context as metadata.

This structure mirrors the composition of the objective function and is key
to enabling parallel execution.

-------------------------

Metadata
==========

Each context contains a ``meta`` dictionary that can be used to store
additional information during evaluation.

Typical uses include:

- recording intermediate results
- storing diagnostic information
- tracking which terms contributed to the final loss

Metadata is propagated upward from child contexts to the parent context,
allowing you to inspect the full evaluation after it completes.

In addition, :class:`~chemfit.abstract_objective_function.QuantityComputer`
instances can define static metadata via their ``static_meta_data`` attribute.
This information is automatically merged into the context's ``meta`` dictionary
during evaluation, allowing components to annotate results with descriptive or
structural information.

-------------------------

Configuration and shared state
===============================

Contexts provide two namespaces for passing information:

``config``
    A per-context namespace that is copied to child contexts.
    This is useful for passing configuration or control information
    to nested evaluations.

``shared``
    A namespace that is shared across related contexts.
    This can be used for read-only data that should be reused
    without duplication.

In most cases, you will not need to use these directly. They are primarily
intended for advanced use cases and custom objective implementations.


-------------------------


Parallel evaluations require separate contexts
===============================================

An :class:`~chemfit.abstract_objective_function.EvaluateContext` stores
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

This is distinct from the child-context mechanism used inside
:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`,
where one parent context owns multiple child contexts for the terms of a
single objective evaluation.


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
