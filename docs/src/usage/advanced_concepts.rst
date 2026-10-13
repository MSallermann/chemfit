.. _advanced_concepts:

#################
Advanced concepts
#################

The :ref:`concepts` page introduces quantity computers, objectives, and
evaluation contexts. This page goes one level deeper: it explains the common
component protocol, the difference between atomic and structural composition,
and the two different kinds of pipeline used by ChemFit.

These distinctions matter when an objective is executed by a graph scheduler.
Ordinary Python nesting and graph-visible composition can calculate the same
value, but they expose different scheduling boundaries and different context
topologies.

-------------------------

The general objective component
===============================

:class:`~chemfit.abstract_objective_function.ObjectiveComponent` is the
general abstraction behind quantities, losses, objective terms, pipelines,
and joins. Conceptually, an ``ObjectiveComponent[InputT, OutputT]`` maps one
input value to one output value::

    InputT -> ObjectiveComponent -> OutputT

The input and output do not have to be parameter mappings and scalar losses.
For example, a component may map parameters to quantities, quantities to a
loss, or one intermediate representation to another. ``QuantityComputer`` is
a typing alias for the first of those cases; it is not a separate runtime base
class.

The :func:`chemfit.component()
<chemfit.abstract_objective_function.component>` decorator turns an ordinary
callable into a
:class:`~chemfit.abstract_objective_function.WrappedComponent`:

.. testcode::

    import chemfit

    @chemfit.component()
    def normalize(values):
        scale = max(abs(value) for value in values)
        return [value / scale for value in values]

    print(normalize([1.0, -2.0]))

.. testoutput::
    :hide:

    [0.5, -1.0]

``quantity()`` and ``objective()`` use the same wrapping machinery but express
more specific intent in an objective definition. Custom classes can instead
subclass ``ObjectiveComponent`` and implement ``_run(input_, ctx)``; see
:ref:`writing_quantity_computers` for implementation guidance.

The evaluation lifecycle
------------------------

Calling a component runs a lifecycle around ``_run()``:

.. code-block:: text

    reset context state
            |
    apply context configurator
            |
    record ctx.input
            |
    run pre-evaluation hooks
            |
    _run(input, ctx), unless a pre-hook supplied the result
            |
    merge static metadata and record ctx.output
            |
    run post-evaluation hooks
            |
    return the value or raise the recorded exception

An ordinary exception from a pre-hook or ``_run()`` is finalized through the
same end lifecycle as a successful value. Consequently, post-hooks can inspect
the exception in ``ctx.output``. A post-hook may also replace ``ctx.output``;
the value stored there after the hooks is authoritative.

A pre-hook short-circuits ``_run()`` by returning a non-``None`` value. The
component still applies its static metadata and runs its post-hooks. This is
important for structural components: a structural component that is
short-circuited at its own pre-hook never creates or evaluates its children.

Fluent configuration methods such as ``with_meta()``, ``bind()``, and
``with_ctx_configurator()`` return configured copies. This makes it possible
to derive several component definitions from one base object without sharing
their registration lists or configuration dictionaries accidentally.

-------------------------

Atomic and structural components
================================

The graph scheduler divides components into two categories.

An **atomic component** is any ``ObjectiveComponent`` that is not a
:class:`~chemfit.abstract_objective_function.StructuralComponent`. The graph
scheduler treats its complete ``_run()`` as one work item. Atomic does not mean
that the implementation is small or that it makes only one function call. An
atomic component can run a simulation, launch an external program, or call
several other components internally.

A **structural component** publishes its immediate children and their data
flow. The graph scheduler expands that description recursively and schedules
the atomic leaves. The structural component's begin and finish phases preserve
its own hooks, metadata, context, and result-combination policy.

This means that merely calling a component from another component's ``_run()``
does not add a node to the scheduler's graph:

.. testcode::

    class HiddenComposition(chemfit.ObjectiveComponent):
        def __init__(self, child):
            super().__init__()
            self.child = child

        def _run(self, input_, ctx):
            # Valid, but the scheduler sees HiddenComposition as one atomic node.
            ctx.pipeline = [ctx.spawn_ctx()]
            return self.child(input_, ctx.pipeline[0])

    hidden = HiddenComposition(normalize)
    hidden_ctx = chemfit.EvaluateContext()
    print(hidden([1.0, -2.0], hidden_ctx))
    print(len(hidden_ctx.pipeline))

.. testoutput::
    :hide:

    [0.5, -1.0]
    1

Use ordinary nesting when the operation should remain indivisible. Use a
structural component when its children should be visible as independent graph
nodes.

-------------------------

Two kinds of pipeline
=====================

ChemFit provides both an opaque pipeline and a structural pipeline. Both pass
the output of one stage to the next, but the scheduler sees them differently.

.. list-table:: Opaque and structural pipelines
   :header-rows: 1
   :widths: 22 39 39

   * - Property
     - Opaque pipeline
     - Structural pipeline
   * - Type
     - :class:`~chemfit.abstract_objective_function.OpaquePipeline`
     - :class:`~chemfit.abstract_objective_function.PipelineComponent`
   * - Construction
     - Produced by ``component.with_loss(...)``; extend with ``then(...)``
     - Construct with ``chemfit.pipeline(first, second)``; extend with
       ``then(...)``
   * - Scheduler view
     - One atomic work item
     - One graph-visible node per stage, with structural start and finish nodes
   * - Intermediate transfer
     - Passed directly between stages in the same atomic call
     - Passed as the outcome of one scheduled node to the next
   * - Stage contexts
     - Stored in ``ctx.pipeline``
     - Stored as immediate children in ``ctx.incoming``
   * - Failure propagation
     - A raised stage exception immediately prevents later calls
     - Later stages are marked skipped by the default input policy
   * - Typical use
     - Keep a tightly coupled or cheap sequence in one scheduling unit
     - Expose a sequence, including nested structures, to the graph scheduler

Opaque pipelines
----------------

:meth:`~chemfit.abstract_objective_function.ObjectiveComponent.with_loss`
constructs an opaque pipeline containing the quantity component and a wrapped
loss component:

.. testcode::

    @chemfit.quantity()
    def simulate(params):
        return {"prediction": params["x"] ** 2}

    def loss(quantities, target):
        return (quantities["prediction"] - target) ** 2

    term = simulate.with_loss(loss, target=2.0)

The graph scheduler submits ``term`` as a single atomic node. The simulation's
result passes directly to the loss without becoming a scheduler-level result.
This avoids an extra scheduling boundary and keeps the usually cheap loss next
to the quantity computation that produced its input.

.. important::

    ``with_loss()`` makes the *complete returned pipeline* opaque. If it is
    called on a structural component, that component's descendants are inside
    the new atomic boundary and are no longer expanded by a graph scheduler.
    To keep an existing structure graph-visible, wrap the loss as a component
    and connect it with a structural pipeline instead:

    .. testcode::

        @chemfit.component()
        def loss_stage(quantities, target):
            return (quantities["prediction"] - target) ** 2

        @chemfit.component()
        def pass_quantities(quantities):
            return quantities

        simulation_graph = chemfit.pipeline(
            simulate,
            pass_quantities,
        )
        visible_term = chemfit.pipeline(
            simulation_graph,
            loss_stage.bind(target=2.0),
        )

        print(visible_term({"x": 3.0}))

    .. testoutput::
        :hide:

        49.0

Opaque does not mean unobservable. Each stage receives its own context, in
execution order, and the outer context records those contexts in
``ctx.pipeline``:

.. testcode::

    ctx = chemfit.EvaluateContext()
    value = term({"x": 3.0}, ctx)

    quantity_ctx = ctx.pipeline[0]
    loss_ctx = ctx.pipeline[1]

    assert quantity_ctx.output == {"prediction": 9.0}
    assert loss_ctx.output == value

    print(value)
    print(quantity_ctx.output)
    print(loss_ctx.output)

.. testoutput::
    :hide:

    49.0
    {'prediction': 9.0}
    49.0

The stage contexts are intentionally not structural children: they are not
listed in ``ctx.incoming``, and recursive hook registration does not descend
through an opaque pipeline. Register a hook or metadata on a stage before
constructing the pipeline when it belongs to that stage.

If a stage raises, its context records the exception and the normal Python call
chain stops before the next stage. The outer opaque-pipeline context then
records and re-raises the same failure through its own end lifecycle.

Structural pipelines
--------------------

:func:`chemfit.pipeline() <chemfit.api.pipeline>` constructs a
:class:`~chemfit.abstract_objective_function.PipelineComponent` that declares
the same left-to-right data flow structurally:

.. testcode::

    @chemfit.component()
    def double(value):
        return 2 * value

    @chemfit.component()
    def add_one(value):
        return value + 1

    pipeline = chemfit.pipeline(double, add_one).then(double)

    print(pipeline(3))

.. testoutput::
    :hide:

    14

The first stage receives the pipeline's parent input. Every later stage depends
on its immediate predecessor and receives that predecessor's outcome. The
final stage supplies the pipeline result.

Stages within this simple chain cannot run concurrently, but graph visibility
still matters. A stage can itself be a join with parallel children; schedulers
can place each atomic stage separately; and failures become explicit skipped
outcomes for downstream nodes. If the final stage was skipped because an
upstream stage failed, the pipeline re-raises the original failure. An
intentional final skip without a failure reason raises ``RuntimeError`` because
the pipeline has no value to return.

-------------------------

Joins and independent children
==============================

A join describes a fan-out/fan-in shape:

.. code-block:: text

                      +--> child 0 --+
    parent input -----+--> child 1 --+--> combine outcomes
                      +--> child 2 --+

:class:`~chemfit.abstract_objective_function.JoinedComponent` is the base
implementation for that shape. Every child receives the original parent input,
and the children have no dependencies on one another. Direct evaluation visits
them sequentially; a graph scheduler may run them concurrently.

:func:`chemfit.join() <chemfit.api.join>` constructs the general built-in join
and returns its child outcomes as a tuple:

.. testcode::

    @chemfit.component()
    def square(value):
        return value**2

    @chemfit.component()
    def cube(value):
        return value**3

    joined = chemfit.join(square, cube)
    joined_ctx = chemfit.EvaluateContext()

    print(joined(3, joined_ctx))
    print([child.output for child in joined_ctx.incoming])

.. testoutput::
    :hide:

    (9, 27)
    [9, 27]

``JoinedComponent`` leaves ``combine_outcomes()`` to subclasses. The built-in
:class:`~chemfit.abstract_objective_function.TupleJoinedComponent` returns the
child outcomes in declaration order, preserving exceptions and skip markers as
tuple entries.

:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction` is the
join used by :func:`chemfit.combine() <chemfit.api.combine>`. Its children are
independent scalar objective terms. Once all terms have completed, it:

1. passes failed term outcomes to the configured exception handler;
2. omits terms for which that handler returns ``None``;
3. multiplies the remaining values by their weights; and
4. passes the weighted values to the configured reducer.

The default exception handler re-raises a term failure during the combine
phase. Importantly, the failure is first captured as that child's outcome, so
independent siblings can still complete. Alternative handlers may replace a
failure with ``NaN`` or omit the failed term. The indices of omitted terms are
recorded in the combined context's ``meta["skipped_indices"]``.

-------------------------

The structural component contract
=================================

A custom :class:`~chemfit.abstract_objective_function.StructuralComponent`
describes one level of composition with six methods:

``components()``
    Returns the immediate children in topological order. The position of each
    child is its index in all other declarations.

``component_dependencies()``
    Returns one tuple of prerequisite child indices per child. Dependencies
    constrain readiness but do not supply the child's input value.

``component_inputs()``
    Returns the ordered input sources for every child. An integer source also
    creates the required graph dependency because the referenced result must
    exist before it can be consumed.

``resolve_child_input(index, inputs, ctx)``
    Converts the assembled sources into the actual child input or returns a
    :class:`~chemfit.abstract_objective_function.Skipped` marker. The default
    returns a sole source directly, groups multiple sources in a tuple, and
    skips when any source is an exception or is already skipped.

``combine_inputs()``
    Selects the sources passed to ``combine_outcomes()``. The default selects
    every immediate child outcome in component order.

``combine_outcomes(outcomes, ctx)``
    Interprets the selected values, exceptions, and skip markers and produces
    the structural component's result.

An input source can be:

``PARENT_INPUT``
    The input received by the structural component.

An integer
    The outcome of the child at that index. The child must precede its consumer
    in topological order.

A :class:`~chemfit.abstract_objective_function.Skipped` value
    An explicit instruction to skip. ``SKIPPED`` is the shared marker without
    a reason; ``Skipped(reason=exception)`` retains a causal failure.

Dependencies and inputs answer different questions. For example, a child may
need another child to finish before it starts but still receive the original
parent input. That relationship belongs in ``component_dependencies()``. If it
consumes the other child's value, that relationship belongs in
``component_inputs()`` and also implies execution ordering.

The reference ``StructuralComponent._run()`` evaluates this description
sequentially. Graph schedulers compile the same description and may overlap
independent children. Both paths must produce the same values, skips,
exceptions, hooks, and context state. The sequential implementation is
therefore also the semantic reference for a custom structure.

Selecting only some results in ``combine_inputs()`` does not let the parent
finish early. The structural scope waits for every immediate child to finish
or be skipped before it runs ``combine_outcomes()`` and its post-hooks. This
ensures that all child side effects and context results are complete when the
parent lifecycle ends.

The complete implementation pattern, including a custom two-way structure,
is covered in :ref:`writing_quantity_computers`.

-------------------------

Context topology
================

The context structure records evaluation scopes, but it uses different links
for the two composition models:

.. code-block:: text

    structural join                       opaque pipeline

    child_ctx_0 --+                       outer_ctx
                  +--> parent_ctx             |
    child_ctx_1 --+                       pipeline = [stage_ctx_0, stage_ctx_1]

For a structural component, ``parent_ctx.incoming`` contains one context per
immediate child in component order. Each child points back to the parent
through ``outgoing``. These links express containment by the structural scope;
they do not reproduce every dependency edge between siblings. Nested
structural components create nested context relationships recursively.

For an opaque pipeline, ``outer_ctx.pipeline`` contains the ordered stage
contexts. Those stages are deliberately absent from ``incoming`` because they
are not graph-visible children. ``ctx.to_summary(recursive=True)`` follows both
kinds of relationship.

Child contexts are created only after the structural parent's begin lifecycle
succeeds. If a parent pre-hook returns a result or raises, it has no child
contexts. Once child contexts have been created, a child skipped by input
resolution keeps its initial ``NO_OUTPUT`` value because none of its lifecycle
runs. A child that executes and fails instead stores the exception in its
``ctx.output``.

Each child gets a deep copy of its parent's ``config``, a fresh ``meta`` and
``temp``, and the same ``shared`` namespace for local execution. Worker-backed
schedulers transport result-bearing output, metadata, and opaque pipeline
state back to the coordinator; temporary state and worker mutations to
``shared`` are not result channels.

-------------------------

Choosing the right composition
==============================

Use the smallest abstraction that expresses the scheduling boundary you need:

- Wrap one callable with ``component()``, ``quantity()``, or ``objective()``
  when it is one logical work item.
- Call nested components directly, or use an
  :class:`~chemfit.abstract_objective_function.OpaquePipeline`, when the nested
  work must stay inside one atomic scheduler task.
- Use :func:`chemfit.pipeline() <chemfit.api.pipeline>` for an ordered sequence
  whose stages should remain graph-visible.
- Use :func:`chemfit.join() <chemfit.api.join>` for independent fan-out that
  returns every outcome, or
  :class:`~chemfit.combined_objective_function.CombinedObjectiveFunction` for
  independent scalar terms followed by configurable reduction.
- Implement :class:`~chemfit.abstract_objective_function.StructuralComponent`
  directly for a custom directed acyclic graph, conditional execution, or a
  specialized recovery policy.

Once the component structure is clear, :ref:`parallel_execution` explains how
the available graph nodes are mapped to serial, executor, and MPI backends.
