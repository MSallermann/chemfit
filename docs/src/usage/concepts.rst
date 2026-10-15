.. _concepts:

################
Concepts
################

This page introduces the core concepts behind ChemFit and explains how
objective functions are structured and evaluated.

The goal is to provide a mental model that helps you design your own
objective functions and understand how ChemFit executes them.

Most workflows begin with :func:`chemfit.quantity()
<chemfit.abstract_objective_function.quantity>`, add a loss with
``with_loss()``, and assemble terms with :func:`chemfit.combine()
<chemfit.api.combine>`. Before looking at execution, it helps to separate the
runtime object model from the role each object plays.

-------------------------

.. _component_vocabulary:

Components, quantity computers, and objectives
===============================================

These terms describe two different things:

- **Component** is the runtime abstraction. Every ChemFit computation and
  composition object derives from
  :class:`~chemfit.abstract_objective_function.Component` and maps one
  input to one output.
- **Quantity computer** and **objective** describe particular input/output
  roles a component can have.

The name ``Component`` is deliberately general. Not every ``Component`` is a
complete objective that an optimizer can minimize. It may instead produce quantities, apply an
intermediate transform, join branches, or connect pipeline stages.

.. list-table:: Canonical ChemFit vocabulary
   :header-rows: 1
   :widths: 20 25 29 26

   * - Term
     - Contract
     - Typical construction
     - Meaning
   * - Component
     - ``InputT -> OutputT``
     - ``component()`` or a ``Component`` subclass
     - Any unit participating in ChemFit's evaluation lifecycle
   * - Quantity computer
     - ``parameters -> quantities dict``
     - ``quantity()``, ``ase_quantity()``, or ``external_quantity()``
     - Computes scientific values that a loss or later stage consumes
   * - Loss function
     - ``quantities -> float``
     - An ordinary callable passed to ``with_loss()``
     - Scores quantities against reference data; it is wrapped internally
   * - Objective term
     - ``parameters -> float``
     - ``objective()`` or ``quantity_computer.with_loss(...)``
     - Produces one scalar contribution suitable for optimization
   * - Combined objective
     - ``parameters -> float``
     - ``combine(term_a, term_b, ...)``
     - Weights and reduces several objective terms to the total loss

``QuantityComputer`` is a typing alias for a ``Component`` with the
quantity-computer contract; it is not another runtime base class. Likewise,
``component()``, ``quantity()``, and ``objective()`` all wrap a Python function
in :class:`~chemfit.abstract_objective_function.WrappedComponent`. The helpers
differ in their type contracts and communicate the intended role:

.. code-block:: text

    parameters -> quantity computer -> quantities -> loss function -> term loss
                  |_______________________________________________|
                                  objective term

    parameters -> objective term A --+
                 objective term B --+--> combined objective -> total loss

Calling ``with_loss()`` does not make the quantity computer itself become an
objective. It returns a new pipeline containing the quantity computer followed
by the loss. That complete pipeline has the ``parameters -> float`` contract,
so it is an objective term; the original quantity computer remains unchanged.
For an ordinary component the returned pipeline is opaque and forms one
scheduler task. For a structural component it is a structural pipeline, so the
existing child graph remains visible to schedulers.

Use *component* when discussing behavior shared by every node, such as hooks,
contexts, pipelines, and scheduling. Use *quantity computer* only for a
component that returns a quantity dictionary, and use *objective* for a
callable or component that maps fitting parameters to a scalar loss.

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
component-specific schedule. The prepared schedule is callable for one input
and provides ``evaluate_many`` for batches. Direct, serial-graph,
executor-backed, and MPI graph schedulers all operate on the same component
definition; see :ref:`parallel_execution`.

-------------------------

.. _evaluation_contexts:

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

Context hierarchy
=================

Components create a context tree mirroring their nested evaluation scopes.

- The top-level evaluation uses the context supplied by the caller.
- Each nested component receives its own context.
- Every context exposes its ordered immediate evaluation contexts through
  ``children``.

Conceptually:

.. code-block:: text

    combined-objective context
    ├── term 1 context
    └── term 2 context

Each child context stores its component's input, finalized output, and
metadata. A failed component's output is the exception object. Opaque and
structural pipelines both expose their stages as ``children``. Their scheduler
visibility differs, but their evaluation-context representation does not.

Structural metadata
-------------------

Every :class:`~chemfit.abstract_objective_function.StructuralComponent` also
stores its immediate declared structure in ``ctx.meta["chemfit.structure"]``.
The description is static and JSON-compatible. ``children`` identifies
immediate components by index and name; ``inputs`` records their ordered data
sources; ``dependencies`` records execution-order prerequisites separately;
and ``outputs`` records how the result is constructed. Sources identify the
parent input, another child by index, or a skipped value. Repeated sources are
not collapsed. Runtime values, exceptions, and timings remain in the context.

For example:

.. testcode::

    from chemfit import EvaluateContext, component, pipeline

    @component()
    def double(value):
        return 2 * value

    @component()
    def add_one(value):
        return value + 1

    workflow = pipeline(double, add_one)
    workflow_ctx = EvaluateContext()
    workflow(3, workflow_ctx)
    structure = workflow_ctx.meta["chemfit.structure"]

    print(structure["inputs"])
    print(structure["dependencies"])

.. testoutput::
    :hide:

    [[{'kind': 'parent_input'}], [{'kind': 'child', 'index': 0}]]
    [[], [0]]

Structural metadata is local rather than flattened. A nested structural
child's context has its own entry. Recursively pairing declared child indices
with context ``children`` reconstructs the evaluated structure, with paths
such as ``(0, 2)`` identifying components. The same information is included by
:meth:`~chemfit.evaluate_context.EvaluateContext.to_summary` when called with
``recursive=True``.

Declared components and runtime contexts are distinct. A structural child
skipped before its lifecycle begins retains its structural entry, but has
``NO_OUTPUT`` and no children. A structural pre-hook that short-circuits
likewise has structural metadata but no child contexts. ChemFit does not
fabricate deeper contexts for a scope that never began, so topology below that
scope is not available by context traversal alone.

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
``ctx.children`` recursively when nested component metadata is needed.

Static metadata belongs to a reusable computation or objective definition. Add
it fluently with
:py:meth:`~chemfit.abstract_objective_function.Component.with_meta`:

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
stored on ``ctx.children[0].meta`` and the metadata added after ``with_loss()``
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
components: those contexts record nested component invocations within one
evaluation, whereas the contexts above represent separate root evaluations.
The context hierarchy records evaluation information, not computational
dependencies; structural components declare those dependencies separately.


Design principles
==================

ChemFit is built around a few key ideas:

- **Composition:** Build complex objectives from reusable components.

- **Explicit dependencies:** Structural components describe computational
  dataflow and execution dependencies.

- **Evaluation contexts:** Contexts record inputs, outputs, metadata, and
  nested evaluation information.

- **Independent execution:** Scheduling is separate from component definitions
  and optimization.

These principles make ChemFit flexible, extensible, and well-suited for
large-scale fitting problems.

Continue with :ref:`advanced_concepts` for the distinction between atomic and
structural components, opaque and graph-visible pipelines, joins, data-flow
declarations, and skip semantics.
