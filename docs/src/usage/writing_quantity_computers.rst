.. _writing_quantity_computers:

#############################
Writing Quantity Computers
#############################

This page contains practical guidance for writing your own quantity computers from scratch.

********************
Before we start...
********************

Before we start, let us clearly state that "quantity computers" are merely a useful convention.
Nothing prevents you from using ChemFit without them. That being said, you should probably use them.

A recommended first step is to check if you can use one of the built-in ways:

1. If you already have a pure python function implementing your computation, have a look at the :py:func:`~chemfit.wrap_funcs.quantity` decorator, which also features in the :ref:`quickstart` examples.
2. If you are using an external simulation tool, like LAMMPS for example, have a look at the :py:class:`~chemfit.external_computer.ExternalQuantityComputer` and its corresponding doc page: :ref:`external_computer`.
3. If you are using ASE, try the :py:class:`~chemfit.ase_objective_function.ASEComputer` described in :ref:`ase_objective_function_api`.

If none of the built-in computers are to your taste, think about sub-classing them.

********************
Let's cook
********************

For a completely fresh QuantityComputer, derive from the
:py:class:`~chemfit.abstract_objective_function.QuantityComputer` base class
and implement ``QuantityComputer._compute``. That's it.

The ``_compute`` method should accept exactly two arguments: a parameter
mapping (commonly ``dict[str, Any]``) and an
:py:class:`~chemfit.abstract_objective_function.EvaluateContext`.
It should return the dictionary of quantities.

This is probably a point at which we should familiarize ourselves with the...

**Golden Rule:**
    DO NOT MODIFY GLOBAL STATE FROM WITHIN THE COMPUTE METHOD. If you violate this rule, parallel evaluation of your quantity computer can be undefined.
    It does not *have* to be, but for everyone's sake let's assume it **will** be.

    Importantly, the golden rule applies to instance variables of the computer itself as well.

    Let's illustrate what **not** to do:

    .. code-block:: python

        class GoldenRuleViolator(QuantityComputer):
            # ...
            def _compute(self, params, ctx):
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

.. code-block:: python

    class GoodCitizen(QuantityComputer):
        # ...
        def _compute(self, params, ctx):
            bad = params["x"]
            ctx.meta["bad"] = bad # <-- no problemo
            # ...
            return {"mojo" : bad}

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

    from chemfit.wrap_funcs import quantity

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

    If ``pass_ctx==False``, all arguments except ``params`` have to be bound.

If we forego the :py:func:`~chemfit.wrap_funcs.quantity` approach and we need external parameters, they should be accepted in the constructor.

.. code-block:: python

    from chemfit.abstract_objective_function import QuantityComputer

    class Computer(QuantityComputer):

        def __init__(self,f):
            self.f = f

        def _compute(self, params, ctx):
            # We can make use of self.f in here
            ...

.. important::

    A quantity computer becomes fully specified once it depends only on ``(parameters, ctx)``. At that point, all external parameters have been
    fixed, either via
    :py:meth:`~chemfit.wrap_funcs.WrappedQuantityComputer.bind` (for wrapped
    functions) or via the constructor (for class-based implementations).


******************************************
Using the evaluation context
******************************************

The :class:`~chemfit.abstract_objective_function.EvaluateContext`
provides a structured way to exchange information during evaluation
without relying on shared state.

It exposes three main fields:

- ``ctx.meta`` — for *results and diagnostics*
- ``ctx.config`` — for *read-only configuration*
- ``ctx.shared`` — for *controlled shared state*

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
``with_meta()`` instead of writing it inside ``_compute()``:

.. code-block:: python

    base = Computer(...)
    liquid = base.with_meta(dataset="liquid", phase="condensed")
    gas = base.with_meta(dataset="gas", phase="vapor")

The method returns a shallow copy, so ``base`` is unchanged. This static
metadata is merged into ``ctx.meta`` before ``_compute()`` runs. If the
computer is turned into an objective with ``with_loss()``, objective metadata
can be added afterward as well:

.. code-block:: python

    term = (
        liquid
        .with_loss(density_loss, target=0.997)
        .with_meta(observable="density")
    )

Quantity metadata is applied first and objective metadata afterward; objective
metadata wins if both define the same key. Continue to write values calculated
during an evaluation directly to ``ctx.meta``.

Typical static annotations include:

- a label or tag identifying the term
- the origin of the data
- a fixed configuration identifier

**Important:**
    Quantities that are part of the computation should be returned from
    ``_compute``, not written to ``ctx.meta``.

**Rule of thumb:**
    If something is needed for the loss or further computation, return it.
    If it is only useful for inspection, debugging, or bookkeeping, store
    it in ``ctx.meta``.

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

Static requirements such as CPU or GPU counts belong in the objective or
computer's ``resources`` mapping. Resource annotations describe requirements;
the built-in serial, executor, and MPI schedulers currently do not perform
resource-aware placement.

**Rule of thumb:**
    Use ``ctx.config`` to *influence how the computation is carried out*,
    but never modify it inside ``_compute``.

====================
ctx.shared
====================

``ctx.shared`` allows controlled sharing of state across related parent and
child contexts. Separate root contexts share state only when they are
explicitly constructed with the same namespace.

This is useful for:

- caching expensive results
- sharing resources between evaluations
- coordinating work across parallel tasks

However, this is also the most dangerous field.

**Important:**
    During local or threaded execution, related contexts use the same object,
    so access may be concurrent and must be thread-safe. Process executors
    serialize worker-input state instead; mutations made in a worker are not
    returned to the driver.

Example (initialize a cache before concurrent evaluation):

.. code-block:: python

    cache = ctx.shared.cache

    key = tuple(sorted(params.items()))
    if key in cache:
        return cache[key]

    result = expensive_computation(...)
    cache[key] = result

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
Calling computers from within computers
******************************************

.. note::

    This section is for fairly advanced use and is particularly relevant when
    implementing a component that performs its own nested computations.

If we want to make calls to other computers from our custom computer, the recommended approach is to make use of the child context system to supply fresh contexts to the inner computers.

Here is a simple demonstration of the idea: We have an outer computer, which accepts a parent :py:class:`~chemfit.abstract_objective_function.EvaluateContext` and then later on splits of two child contexts using the :py:meth:`~chemfit.abstract_objective_function.EvaluateContext.child_contexts` context manager.

.. code-block:: python

    class OuterComputer(QuantityComputer):

        def _compute(self, params, ctx):
            # ...
            with ctx.child_contexts(2) as child_contexts:
                q1 = inner_computer1(params, child_contexts[0])
                q2 = inner_computer2(params, child_contexts[1])
            # ...

The benefit of this approach is two-fold

1. We get full meta-data provenance. All of the child meta data can be found in ``ctx.meta["children"]``.
2. Since the inner computers have their own context they can also be evaluated in parallel ... although the example above does not make use of this.

.. note::

    For parallel evaluation with an executor, use the :py:func:`~chemfit.executor_utils.map_with_context` function.
    Differently from the regular executor ``map`` function, it correctly handles the ``ctx`` fields even if execution happens in different processes.
    See also :ref:`parallel_execution`.

=============================================================
Child-parent relationships for the different context fields
=============================================================

When creating child contexts via
:py:meth:`~chemfit.abstract_objective_function.EvaluateContext.child_contexts`,
the different fields of the context behave differently.

Understanding this behavior is important when composing quantity
computers.

--------------------
ctx.meta
--------------------

Each child context receives its own independent ``meta`` dictionary.

During evaluation, child computers write to their own ``ctx.meta``.
After the ``child_contexts`` block exits, the parent context collects
all child meta data under:

.. code-block:: python

    ctx.meta["children"]

This is a list containing the meta data of each child evaluation, in
order.

This ensures full provenance: all information produced by child
computations is preserved and accessible from the parent.

--------------------
ctx.config
--------------------

The ``config`` namespace is deep-copied from the parent into each child.

All child contexts see the same configuration, allowing them to adapt
their behavior consistently.

.. code-block:: python

    value = getattr(ctx.config, "mode", None)

Child contexts should treat their local ``config`` as read-only. A child
context configurator may customize the copy before evaluation.

--------------------
ctx.shared
--------------------

The same ``shared`` namespace object is used by parent and child contexts
during local execution.

This allows child computations to communicate and reuse data, for
example through caching.

.. code-block:: python

    cache = ctx.shared.cache

Because ``ctx.shared`` may be accessed concurrently, all access must be
thread-safe. Process executors serialize worker-input shared state but do not
transport mutations to it back to the driver.

===================================
Configuring child contexts
===================================

Besides the number of children,
:py:meth:`~chemfit.abstract_objective_function.EvaluateContext.child_contexts`
accepts an optional argument of type
:py:class:`~chemfit.abstract_objective_function.ChildContextConfigurator`.

A child context configurator allows you to customize how child contexts
are created.

This can be useful when:

- assigning identifiers or indices to child evaluations
- modifying configuration for individual children

The configurator is called once per child context and can modify the
child context before it is used.

Conceptually, it allows you to control:

.. code-block:: python

    def configurator(idx_child_ctx, child_ctx, num_children, parent_ctx):
        ...

For example, you may want to assign each child a unique identifier:

.. code-block:: python

    def configurator(idx_child_ctx, child_ctx, num_children, parent_ctx):
        child_ctx.meta["child_index"] = idx_child_ctx

Or adjust configuration per child:

.. code-block:: python

    def configurator(idx_child_ctx, child_ctx, num_children, parent_ctx):
        child_ctx.config.worker_id = idx_child_ctx

This mechanism configures evaluation semantics and metadata. Backend placement
is controlled separately by the prepared scheduler.

**Rule of thumb:**
    Use a child context configurator when child evaluations need
    systematic differences in their context. Otherwise, the default
    behavior is sufficient.
