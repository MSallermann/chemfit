.. _writing_quantity_computers:

#############################
Writing Quantity Computers
#############################

This page contains practical guidance for writing a component with the
``parameters -> quantities dict`` contract: a quantity computer. See
:ref:`component_vocabulary` for how that role differs from a general component
and a scalar objective.

********************
Before we start...
********************

A quantity computer is a role, not a separate runtime class. It is an
:class:`~chemfit.abstract_objective_function.ObjectiveComponent` whose input is
a fitting-parameter mapping and whose output is a dictionary of quantities.
Use one when the scientific calculation should remain separate from the loss
that scores its results. If a function already computes the final scalar loss,
wrap it with ``objective()`` instead.

A recommended first step is to check if you can use one of the built-in ways:

1. If you already have a pure Python function implementing your computation,
   start with :func:`chemfit.quantity() <chemfit.abstract_objective_function.quantity>`, which
   also features in the :ref:`quickstart` examples.
2. If you use an external simulation tool such as LAMMPS, start with
   :func:`chemfit.external_quantity() <chemfit.api.external_quantity>` and see
   :ref:`external_computer`.
3. If you use ASE, start with
   :func:`chemfit.ase_quantity() <chemfit.api.ase_quantity>` and see
   :ref:`ase_objective_function_api`.

``chemfit.external_quantity(...)`` returns an
:class:`~chemfit.external_computer.ExternalQuantityComputer`, while
``chemfit.ase_quantity(...)`` returns an
:class:`~chemfit.ase_objective_function.ASEComputer`. The remainder of this
page uses the underlying class vocabulary because it covers advanced custom
implementations.

If none of the built-in computers are to your taste, think about sub-classing them.

********************
Let's cook
********************

For a custom quantity computer, derive from
:py:class:`~chemfit.abstract_objective_function.ObjectiveComponent` and
implement ``ObjectiveComponent._run``. ``QuantityComputer`` is the typing alias
for a component mapping parameter dictionaries to quantity dictionaries.

The ``_run`` method should accept exactly two arguments: a parameter
mapping (commonly ``dict[str, Any]``) and an
:py:class:`~chemfit.evaluate_context.EvaluateContext`.
It should return the dictionary of quantities.

This is probably a point at which we should familiarize ourselves with the...

**Golden Rule:**
    DO NOT MODIFY GLOBAL STATE FROM WITHIN THE COMPUTE METHOD. If you violate this rule, parallel evaluation of your quantity computer can be undefined.
    It does not *have* to be, but for everyone's sake let's assume it **will** be.

    Importantly, the golden rule applies to instance variables of the computer itself as well.

    Let's illustrate what **not** to do:

    .. code-block:: python

        class GoldenRuleViolator(ObjectiveComponent):
            # ...
            def _run(self, params, ctx):
                self.bad = params["x"] # <-- bad mojo
                # ...
                return {"mojo" : self.bad}

    Now what happens if you call the same instance of ``GoldenRuleViolator`` in parallel? That's right! Bad things. The reason is that the value of ``self.bad`` could be overwritten by another thread in the middle of the compute function, which would make your ``params`` and the returned quantities mismatched.

    You might say: "Why would I ever do something so stupid?". Well, you'd be surprised how easy it is to accidentally violate the **Golden Rule**. Even seemingly harmless patterns can violate this rule accidentally,
    especially when storing intermediate results on ``self``.

Therefore, if you have anything to communicate with the outside world, there are two options

1. Put it in the quantities dict and return it
2. Write to ``ctx.meta``

Let's fix the ``GoldenRuleViolator``:

.. testcode::

    from chemfit import EvaluateContext, ObjectiveComponent

    class GoodCitizen(ObjectiveComponent):
        def _run(self, params, ctx):
            bad = params["x"]
            ctx.meta["bad"] = bad # <-- no problemo
            return {"mojo" : bad}

    citizen = GoodCitizen()
    ctx = EvaluateContext()
    print(citizen({"x": 2.0}, ctx))
    print(ctx.meta)

.. testoutput::
    :hide:

    {'mojo': 2.0}
    {'bad': 2.0}

Now there is no problem. All we ever do is write to ``bad``, which is local to
the current function evaluation, or to ``ctx.meta["bad"]``, which belongs to
that evaluation's context.

******************************************
Configuring a computer
******************************************

Besides, the parameter dictionary passed on evaluation, a quantity computer may also want to be configured by different external parameters.

Symbolically we can imagine an external parameter :math:`f`, which influences the computation in some way:

.. math::

    \text{Quantities}(\text{params},f,\text{ctx}) = \{ ... \}

The external parameter :math:`f` may for example be

- A path to file with atomic coordinates
- A constant numeric prefactor
- The charge of certain atoms
- ...
- You name it! Anything that influences the quantities and is otherwise fixed.

If the quantity computer is a wrapped python function, it's easy to bind external parameters. Check this out:

.. code-block:: python

    from chemfit import quantity

    @quantity(pass_ctx=True)
    def computer(params, *, ctx, f):
        ...

    # Configure f=1
    f1_computer = computer.bind(f=1)
    # Configure f=2
    f2_computer = computer.bind(f=2)

.. note::

    If ``pass_ctx=True``, all required arguments except ``params`` and the
    injected keyword argument ``ctx`` must be bound.

    If ``pass_ctx=False``, all arguments except ``params`` have to be bound.

If we forego the :py:func:`~chemfit.abstract_objective_function.quantity` approach and we need external parameters, they should be accepted in the constructor.

.. code-block:: python

    from chemfit.abstract_objective_function import ObjectiveComponent

    class Computer(ObjectiveComponent):

        def __init__(self,f):
            super().__init__()
            self.f = f

        def _run(self, params, ctx):
            # We can make use of self.f in here
            ...

.. important::

    A quantity computer becomes fully specified once it depends only on ``(parameters, ctx)``. At that point, all external parameters have been
    fixed, either via
    :py:meth:`~chemfit.abstract_objective_function.WrappedComponent.bind` (for wrapped
    functions) or via the constructor (for class-based implementations).


******************************************
Using the evaluation context
******************************************

The :class:`~chemfit.evaluate_context.EvaluateContext`
provides a structured way to exchange information during evaluation
without relying on shared state.

Its main user-facing fields have distinct jobs:

- ``ctx.meta`` — for *results and diagnostics*
- ``ctx.temp`` — for *transient state within one component call*
- ``ctx.config`` — for *read-only configuration*
- ``ctx.shared`` — for *controlled shared state*

Scientific outputs needed by downstream components do not belong in any of
these fields: return them from ``_run()``. Use the context for operational
information around that result.

.. list-table:: Choosing where evaluation data belongs
   :header-rows: 1
   :widths: 38 20 42

   * - Example
     - Location
     - Reason
   * - Energy, forces, density, or another loss input
     - Return value
     - It is part of the scientific data flow
   * - SCF iterations, convergence, timing, or provenance
     - ``ctx.meta``
     - It should survive evaluation for inspection
   * - A timer start or cache key used only by hooks in this call
     - ``ctx.temp``
     - It is transient implementation state
   * - Whether this evaluation requests forces or extra diagnostics
     - ``ctx.config``
     - It changes execution policy for one context
   * - A large, read-only reference table reused by sibling terms
     - ``ctx.shared``
     - Local children can share it without deep-copying it

====================
ctx.meta
====================

``ctx.meta`` is a dictionary used to record auxiliary information about
the current evaluation. It is safe to write to and is typically used for:

- debugging information
- intermediate values that are not part of the returned quantities
- provenance (e.g. which configuration was used)
- performance metrics

.. code-block:: python

    ctx.meta["n_iterations"] = n_iter
    ctx.meta["converged"] = converged
    ctx.meta["structure_id"] = structure_id

Each evaluation has its own ``meta`` dictionary, so there are no race
conditions.

For descriptive metadata that is fixed when the computer is configured, use
``with_meta()`` instead of writing it inside ``_run()``:

.. code-block:: python

    base = Computer(...)
    liquid = base.with_meta(dataset="liquid", phase="condensed")
    gas = base.with_meta(dataset="gas", phase="vapor")

The method returns a shallow copy, so ``base`` is unchanged. Static metadata is
merged into that component's ``ctx.meta`` during finalization, after ``_run()``
and before its post-hooks. If ``with_loss()`` is used to return an objective
term containing the computer, metadata can be added to that new term as well:

.. code-block:: python

    term = (
        liquid
        .with_loss(density_loss, target=0.997)
        .with_meta(observable="density")
    )

The quantity and the resulting objective are distinct evaluation scopes.
Quantity metadata is therefore available on ``ctx.children[0].meta`` after the
objective runs, while metadata added after ``with_loss()`` is available on the
root ``ctx.meta``. Continue to write values calculated during an evaluation to
the context of the component that produced them.

Typical static annotations include:

- a label or tag identifying the term
- the origin of the data
- a fixed configuration identifier

**Important:**
    Quantities that are part of the computation should be returned from
    ``_run()``, not written to ``ctx.meta``.

**Rule of thumb:**
    If something is needed for the loss or further computation, return it.
    If it is only useful for inspection, debugging, or bookkeeping, store
    it in ``ctx.meta``.

====================
ctx.temp
====================

``ctx.temp`` carries private, per-call state between pre-hooks, ``_run()``, and
post-hooks. It is cleared when a context is reused and is not included in
summaries or transported back from process and MPI workers. This makes it the
right place for timers, cache keys, open-call bookkeeping, or prepared input
that has no value after the component finishes.

For example, a pre-hook can select reference data before the quantity
calculation begins without adding that preparation to ``_run()``:

.. testcode::

    from chemfit import EvaluateContext, ObjectiveComponent

    class ReferenceEnergyHook:
        def __init__(self, reference_energies):
            self.reference_energies = reference_energies

        def pre_eval(self, ctx):
            species = ctx.input["species"]
            ctx.temp.reference_energy = self.reference_energies[species]

    class ReferenceShift(ObjectiveComponent):
        def _run(self, params, ctx):
            return params["energy"] - ctx.temp.reference_energy

    shift = ReferenceShift().register_eval_hook(
        hook=ReferenceEnergyHook({"H2": -1.17})
    )

    result = shift(
        {"species": "H2", "energy": -1.0},
        EvaluateContext(),
    )
    print(round(result, 2))

.. testoutput::
    :hide:

    0.17

If the selected value should be preserved for later inspection, copy the
relevant diagnostic to ``ctx.meta`` in a post-hook. Do not rely on ``ctx.temp``
as a result channel.

====================
ctx.config
====================

``ctx.config`` provides configuration information to the computation.
It should be treated as **read-only**.

Typical use cases include:

- passing runtime options
- controlling execution modes
- toggling optional behavior

.. code-block:: python

    if getattr(ctx.config, "compute_forces", False):
        ...

The main purpose of ``ctx.config`` is to allow behavior to vary **per
evaluation**, without requiring reconstruction of the quantity computer
or objective function.

In particular, ``ctx.config`` is useful when the same computation needs
different per-evaluation modes. Different calls may:

- run on different cluster nodes
- access different scratch directories
- enable different diagnostics or approximations

.. code-block:: python

    scratch_dir = getattr(ctx.config, "scratch_dir", "/tmp")
    diagnostics = getattr(ctx.config, "diagnostics", False)

The built-in serial, executor, and MPI schedulers do not provide a standardized
CPU/GPU resource-requirement mapping or resource-aware placement. Provision
the selected backend appropriately and keep any application-specific resource
description in your own configuration or metadata.

**Rule of thumb:**
    Use ``ctx.config`` to *influence how the computation is carried out*,
    but never modify it inside ``_run()``.

====================
ctx.shared
====================

``ctx.shared`` allows controlled sharing of state across related parent and
child contexts. Separate root contexts share state only when they are
explicitly constructed with the same namespace.

This is useful for:

- reusing read-only data across related local or threaded evaluations
- supplying serializable worker-input data to process or MPI evaluations
- backend-specific shared state when the application controls its lifecycle
  and synchronization

It is not a backend-independent distributed cache or shared-memory facility.

**Important:**
    During local or threaded execution, related contexts use the same object,
    so mutable access may be concurrent and requires synchronization supplied
    by the application. Process and MPI schedulers serialize worker-input
    state instead; the namespace must be serializable and mutations made in a
    worker are not returned to the driver.

The portable use of ``ctx.shared`` is therefore limited to read-only,
serializable input. One practical case is a large reference dataset read by
several children: local child contexts can reuse the same object without a
deep copy, while process and MPI workers receive serializable copies.

.. testcode::

    from types import SimpleNamespace

    import chemfit

    @chemfit.component(pass_ctx=True)
    def energy_error(sample, *, ctx):
        reference = ctx.shared.reference["energy"]
        return (sample["energy"] - reference) ** 2

    @chemfit.component(pass_ctx=True)
    def force_error(sample, *, ctx):
        reference = ctx.shared.reference["forces"]
        return sum(
            (predicted - expected) ** 2
            for predicted, expected in zip(sample["forces"], reference)
        )

    residuals = chemfit.join(energy_error, force_error)
    shared = SimpleNamespace(
        reference={"energy": -10.0, "forces": [0.0, 0.0]}
    )
    ctx = chemfit.EvaluateContext(shared=shared)

    print(
        residuals(
            {"energy": -8.0, "forces": [2.0, 0.0]},
            ctx,
        )
    )
    print(ctx.children[0].shared is ctx.children[1].shared)

.. testoutput::
    :hide:

    (4.0, 4.0)
    True

A mutable cache in ``ctx.shared`` is safe only when its storage and locking
match the selected backend. For example, a ``threading.Lock`` works for a
thread-only application but is not serializable; an ordinary dictionary is
copied rather than shared by process and MPI workers. Use an external cache or
another backend-specific service when results must be shared across workers.
Cache lookup belongs in a pre-evaluation hook, which can return a cached value
to short-circuit ``_run()``; a post-evaluation hook can store a newly computed
result. See :ref:`objective_hooks` for the hook lifecycle.

**Rule of thumb:**
    Only use ``ctx.shared`` if you know exactly what you are doing.
    Prefer ``ctx.meta`` or returning values whenever possible.

===================================
External parameters vs ctx.config
===================================

Both external parameters (passed via ``bind`` or the constructor) and
``ctx.config`` can influence the behavior of a quantity computer, but
they serve different purposes.

External parameters define *what* is being computed. They are part of
the identity of the quantity computer or objective term.

Typical examples include:

- the system or structure being evaluated
- a file path or dataset
- physical constants or fixed model settings

These values should usually be fixed when constructing or specializing
the quantity computer.

.. code-block:: python

    computer.bind(atoms_factory=my_structure)
    Computer(f=2.0)

In contrast, ``ctx.config`` defines *how* a particular evaluation is
carried out.

Typical examples include:

- enabling or disabling optional work
- selecting approximate vs. exact evaluation modes
- turning diagnostics on or off
- passing evaluation-specific information such as scratch paths

The main reason to use ``ctx.config`` is that it can vary **per
evaluation** without requiring you to reconstruct the quantity computer
or objective term.

**Rule of thumb:**

- Use external parameters if changing the value creates a *different
  objective term*.
- Use ``ctx.config`` if changing the value only affects *how the same
  term is evaluated*.

For example, changing the atomic structure of a system should be an
external parameter, while enabling additional diagnostics or selecting a
cheap evaluation mode should be handled through ``ctx.config``.

====================
Summary
====================

- ``ctx.meta``: write freely, per-evaluation diagnostic data (plus static meta data from the computer)
- ``ctx.config``: read-only, per-evaluation control of execution
- ``ctx.shared``: shared state, use with care

******************************************
Composing custom components
******************************************

For the conceptual distinction between atomic nesting, opaque pipelines,
structural pipelines, and joins, read :ref:`advanced_concepts` first. This
section focuses on implementing a custom structure.

Calling another component directly from ``_run()`` is valid, but that nested
call is opaque to a graph scheduler. Use it only when the whole operation
should remain one atomic task. If the nested computations should be scheduled
independently, express them with a
:class:`~chemfit.abstract_objective_function.StructuralComponent`.

The built-in
:class:`~chemfit.abstract_objective_function.PipelineComponent` passes each
stage's output to the next stage.
:class:`~chemfit.abstract_objective_function.JoinedComponent` is the base for
components whose children can run independently, including
:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`.

==============================
Writing a structural component
==============================

A structural component describes its immediate children and their data flow:

- ``components()`` returns the children in topological order.
- ``component_dependencies()`` declares execution ordering.
- ``component_inputs()`` declares the sources assembled for each child.
- ``resolve_child_input()`` decides whether a ready child runs.
- ``combine_inputs()`` selects sources passed to ``combine_outcomes()``.
- ``combine_outcomes()`` produces the structural component's result.

For example, this component evaluates two quantity computations independently
and returns both dictionaries:

.. testcode::

    from chemfit.abstract_objective_function import (
        PARENT_INPUT,
        StructuralComponent,
    )
    from chemfit import component

    class QuantityPair(StructuralComponent):
        def __init__(self, left, right):
            super().__init__()
            self._components = (left, right)

        def components(self):
            return self._components

        def component_dependencies(self):
            return ((), ())

        def component_inputs(self):
            return ((PARENT_INPUT,), (PARENT_INPUT,))

        def combine_outcomes(self, outcomes, ctx):
            return outcomes

    @component()
    def original(params):
        return {"value": params["x"]}

    @component()
    def doubled(params):
        return {"value": 2 * params["x"]}

    pair = QuantityPair(original, doubled)
    pair_ctx = EvaluateContext()
    print(pair({"x": 3.0}, pair_ctx))
    print([child.output for child in pair_ctx.children])

.. testoutput::
    :hide:

    ({'value': 3.0}, {'value': 6.0})
    [{'value': 3.0}, {'value': 6.0}]

Both children receive the structural component's original input. Their empty
dependency tuples allow a graph scheduler to run them concurrently. Direct
evaluation uses the same declarations but visits the children sequentially.

==========================
Input sources and skipping
==========================

An input source is one of:

``PARENT_INPUT``
    The input received by the enclosing structural component.

An integer
    The outcome of the child with that index.

A :class:`~chemfit.abstract_objective_function.Skipped` value
    A literal skip. ``SKIPPED`` is the shared no-reason instance.

Once a child's dependencies are complete, ChemFit assembles its declared
sources and calls the parent's ``resolve_child_input(index, inputs, ctx)``.
The default implementation skips the child when any source is an exception or
already skipped. Exceptions become ``Skipped(reason=exception)`` so the
original failure remains available to downstream policy.

Override ``resolve_child_input()`` for conditional execution or recovery:

.. code-block:: python

    from chemfit.abstract_objective_function import Skipped

    def resolve_child_input(self, index, inputs, ctx):
        if index == 1 and not ctx.config.run_expensive:
            return Skipped()
        return super().resolve_child_input(index, inputs, ctx)

A skipped child does not run pre-hooks, ``_run()``, or post-hooks. If it is
itself structural, its entire subtree and finish lifecycle are skipped.

================
Reduction inputs
================

By default, ``combine_inputs()`` selects every immediate child outcome in
component order. It may instead select a subset, ``PARENT_INPUT``, or literal
``Skipped`` values. Selection affects only the values passed to
``combine_outcomes()``: the structural scope still waits until every immediate
child has completed or been skipped.

``combine_outcomes()`` may therefore receive values, exceptions, and
``Skipped`` instances. The structural component owns their interpretation. A
pipeline re-raises the reason attached to its skipped final stage, while a
join may recover from a failed or intentionally skipped term.

=================
Context hierarchy
=================

A structural evaluation creates one context for each immediate child after its
begin lifecycle succeeds and before ordinary child execution starts. The
structural context exposes them through ``ctx.children`` in declaration order.
A structural pre-hook that short-circuits or fails creates no child contexts.
Nested structural components form a context hierarchy matching the component's
structural nesting. Sibling dependency edges are declared by the component and
are represented by the call graph rather than the context tree.

Each spawned context receives:

- a fresh ``meta`` dictionary and ``temp`` namespace;
- a deep copy of the parent's ``config`` namespace;
- the same ``shared`` namespace during local execution.

A skipped child's context remains at ``NO_OUTPUT`` because its lifecycle never
runs. Completed and failed children store their finalized value or exception
in ``ctx.output``. There is no separate exception field.

Use ``ctx.to_summary(recursive=True)`` for a serializable nested summary.
Executor and MPI backends return input, output, children, and metadata state
from atomic workers; ``temp`` and mutations to process-local ``shared`` state
are not transported back.

==============================
Configuring component contexts
==============================

Use ``with_ctx_configurator()`` when a particular component needs context
configuration at evaluation start:

.. testcode::

    @component(pass_ctx=True)
    def configurable_computer(params, *, ctx):
        return {"mode": ctx.config.mode, "x": params["x"]}

    def configure(ctx, *, label):
        ctx.meta["label"] = label
        ctx.config.mode = "diagnostic"

    configured = configurable_computer.with_ctx_configurator(
        configure,
        label="reference",
    )

    configured_ctx = EvaluateContext()
    print(configured({"x": 1.0}, configured_ctx))
    print(configured_ctx.meta)

.. testoutput::
    :hide:

    {'mode': 'diagnostic', 'x': 1.0}
    {'label': 'reference'}

The configurator runs after per-call state is reset and before pre-hooks. It
should modify only the context it receives. Apply different configurators to
different child components when their setup differs. Backend placement remains
the responsibility of the prepared scheduler.
