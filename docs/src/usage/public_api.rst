.. _public_api:
.. _common_workflows:

Common workflows
================

This is the recommended starting point for ordinary ChemFit work. The
:mod:`chemfit` package exposes the common workflow directly: wrap a quantity
computation, attach a loss, combine terms, and fit parameters. Each helper
returns or uses the same framework objects documented in the advanced pages,
so you can move down a layer without rewriting the scientific code.

Compute, attach a loss, combine, fit
------------------------------------

.. code-block:: python

   import chemfit

   @chemfit.quantity()
   def simulate(params, *, distance):
       ratio = params["sigma"] / distance
       energy = 4.0 * params["epsilon"] * (ratio**12 - ratio**6)
       return {"energy": energy}

   def energy_loss(quantities, *, reference):
       return (quantities["energy"] - reference) ** 2

   near_term = simulate.bind(distance=1.1).with_loss(
       energy_loss,
       reference=-0.98,
   )
   far_term = simulate.bind(distance=1.5).with_loss(
       energy_loss,
       reference=-0.32,
   )
   objective = chemfit.combine(near_term, far_term)

   result = chemfit.fit_nevergrad(
       objective,
       initial={"epsilon": 0.7, "sigma": 1.0},
       bounds={"epsilon": (0.1, 2.0), "sigma": (0.5, 2.0)},
       optimizer="NgIohTuned",
       budget=100,
       batch_size=4,
   )

``chemfit.quantity()`` produces a
:class:`~chemfit.wrap_funcs.WrappedQuantityComputer`, and ``with_loss()``
turns each configured simulation into an
:class:`~chemfit.abstract_objective_function.ObjectiveFunctor`. Here the two
terms compare the same model with reference energies at different distances.
``chemfit.combine()`` joins them into a
:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`.

:py:func:`chemfit.api.fit_nevergrad` returns a
:class:`~chemfit.api.FitResult`. ``result.recommendation`` is Nevergrad's
recommended parameter mapping. ``best_parameters`` and ``best_loss`` identify
the best optimizer-visible evaluation recorded by ChemFit, and ``contexts``
contains one :class:`~chemfit.fitter.FitterEvaluateContext` per candidate slot.
Internally, the helper configures a :class:`~chemfit.fitter.Fitter`; see
:ref:`fitter` when you need direct lifecycle control, SciPy, or custom loops.

Concurrency in ``fit_nevergrad``
--------------------------------

:py:func:`chemfit.api.fit_nevergrad` separates candidate batching from
execution concurrency:

- ``batch_size`` is the number of candidate slots exposed to Nevergrad and the
  maximum number of candidates in one ask/evaluate/tell batch.
- ``execution_workers`` is the maximum number of objective leaf tasks run at
  once by the built-in thread scheduler. It defaults to ``batch_size``.

Leaf tasks include independent terms within a combined objective, so the
execution limit applies across both candidates and terms. For example,
``batch_size=4, execution_workers=8`` asks Nevergrad for batches of up to four
candidates while allowing up to eight leaf tasks from the batch to execute at
once.

Supplying ``executor=`` or ``scheduler=`` replaces the built-in scheduler.
That object determines execution concurrency, and ``execution_workers`` must
be omitted. ``executor`` and ``scheduler`` are mutually exclusive. Use a
process executor for CPU-bound Python work that does not release the GIL. See
:ref:`parallel_execution` for executor ownership, prepared schedules, and MPI.

An already prepared schedule may instead be passed as the first argument to
``fit_nevergrad``. It remains caller-owned and cannot be combined with
``executor``, ``scheduler``, or ``execution_workers``.

Direct objectives and contexts
------------------------------

Use :func:`chemfit.wrap_funcs.objective` for a function that already returns a
scalar loss. Set ``pass_ctx=True`` when it needs the evaluation context:

.. code-block:: python

   @chemfit.objective(pass_ctx=True)
   def square(params, *, ctx):
       ctx.meta["kind"] = "square"
       return params["x"] ** 2

   ctx = chemfit.EvaluateContext()
   loss = square({"x": 2.0}, ctx)

The decorators do not infer context use from the callable signature.
``pass_ctx=False`` is the default, in which case the wrapped callable receives
only its parameter mapping plus arguments configured with ``bind``. The
returned :class:`~chemfit.wrap_funcs.WrappedObjectiveFunctor` participates in
the full objective lifecycle described in :ref:`concepts`.

Fluent configuration and chaining
---------------------------------

ChemFit configuration methods are designed to chain. ``bind()``, the
``with_*()`` methods, ``wait_for()``, and ASE's ``minimize()`` return a
configured object and leave the object they were called on unchanged. Read a
chain from top to bottom: create a quantity computer, add or replace its
configuration, and finally attach a loss to produce an objective term.

Use ``with_meta()`` to attach reusable descriptive metadata anywhere in that
chain:

.. code-block:: python

   term = (
       chemfit.ase_quantity(atoms)
       .with_meta(dataset="liquid", temperature=298)
       .with_calculator(make_calculator)
       .with_loss(loss, target=0.997)
       .with_meta(observable="density")
   )

Metadata attached before ``with_loss()`` belongs to the quantity computer and
usually describes computation or provenance. Metadata attached afterward
belongs to the objective term. Both are merged into the evaluation context's
``ctx.meta`` dictionary. Quantity metadata is applied first, followed by
objective metadata, so the objective value wins when both define the same key.
Use direct ``ctx.meta`` writes instead for values computed during one
evaluation.

Use ``bind`` to specialize a wrapped quantity or objective function:

.. code-block:: python

   @chemfit.quantity()
   def scaled(params, *, scale):
       return {"value": scale * params["x"]}

   doubled = scaled.bind(scale=2.0)

Use keyword arguments to ``with_loss`` to bind loss configuration:

.. code-block:: python

   def squared_error(quantities, *, target):
       return (quantities["value"] - target) ** 2

   term = doubled.with_loss(squared_error, target=4.0)

A loss function must accept either ``loss(quantities)`` or
``loss(quantities, parameters)`` after its configuration arguments are bound.
If both forms are valid, ChemFit calls the loss with ``quantities`` only.

The same style configures integrations without a large constructor call. For
example, an external-program term can be assembled as one pipeline:

.. code-block:: python

   term = (
       chemfit.external_quantity("runs")
       .with_hook(write_input, template="input.template")
       .with_cmd(run_model, executable="my-model")
       .with_parser(parse_output, "results.json")
       .wait_for("task.done")
       .with_loss(squared_error, target=reference)
   )

Here ``with_hook()`` and ``with_cmd()`` append execution steps in call order,
``with_parser()`` appends an output parser, and ``wait_for()`` adds a completion
file. Keyword arguments supplied to ``bind()``, ``with_loss()``,
``with_hook()``, ``with_cmd()``, ``with_calculator()``, ``with_evaluator()``,
``with_atoms_modifier()``, and ``with_processor()`` are bound to the
corresponding callable.

The main fluent families are:

- Every :class:`~chemfit.abstract_objective_function.QuantityComputer` and
  :class:`~chemfit.abstract_objective_function.ObjectiveFunctor` provides
  ``with_meta()`` for copy-on-write static metadata.
- Every :class:`~chemfit.abstract_objective_function.QuantityComputer` has
  :meth:`~chemfit.abstract_objective_function.QuantityComputer.with_loss`,
  which ends the quantity-building chain and returns an objective term.
- :class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`
  provides ``with_weights()``, ``with_reduction()``, ``with_aggregator()``, and
  ``with_exception_handler()``. These return configured copies; reducer and
  aggregator calls replace one another. See :ref:`combined_objective_functions`.
- :class:`~chemfit.ase_objective_function.ASEComputer` provides
  :meth:`~chemfit.ase_objective_function.ASEComputer.with_atoms_setup`,
  :meth:`~chemfit.ase_objective_function.ASEComputer.with_atoms_modifier`,
  :meth:`~chemfit.ase_objective_function.ASEComputer.with_calculator`,
  :meth:`~chemfit.ase_objective_function.ASEComputer.with_evaluator`, and
  :meth:`~chemfit.ase_objective_function.ASEComputer.with_processor`. Setup
  callbacks, atoms modifiers, and processors append; calculator and evaluator
  configuration replace the previous value.
  :meth:`~chemfit.ase_objective_function.ASEComputer.minimize` replaces the
  evaluator with the built-in BFGS workflow. See
  :ref:`ase_objective_function_api`.
- :class:`~chemfit.external_computer.ExternalQuantityComputer` provides
  :meth:`~chemfit.external_computer.ExternalQuantityComputer.with_hook`,
  :meth:`~chemfit.external_computer.ExternalQuantityComputer.with_cmd`, and
  :meth:`~chemfit.external_computer.ExternalQuantityComputer.with_parser`;
  all append to the pipeline.
  :meth:`~chemfit.external_computer.ExternalQuantityComputer.wait_for` adds
  completion files. See :ref:`external_computer`.
- Wrapped Python quantities and objectives provide
  :meth:`~chemfit.wrap_funcs.WrappedQuantityComputer.bind` and
  :meth:`~chemfit.wrap_funcs.WrappedObjectiveFunctor.bind` for specializing
  additional callable arguments.

Because the source object is unchanged, it can be used as a reusable base for
multiple variants:

.. code-block:: python

   base = chemfit.ase_quantity(atoms).with_calculator(make_calculator)
   energy_term = base.with_processor(read_energy).with_loss(energy_loss)
   force_term = base.with_processor(read_forces).with_loss(force_loss)

Composition
-----------

:py:func:`chemfit.api.combine` accepts objective terms as separate positional
arguments or as one iterable:

.. code-block:: python

   objective = chemfit.combine(term_a, term_b, weights=[1.0, 0.5])

   terms = (make_term(index) for index in range(10))
   objective = chemfit.combine(terms)

Each term may be an
:class:`~chemfit.abstract_objective_function.ObjectiveFunctor` or a plain
``objective(parameters) -> float`` callable. Plain callables are wrapped
without context injection.

The returned
:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction` also
supports reducers, aggregators, exception handlers, child-context
configuration, and fluent reconfiguration. Those semantics live in
:ref:`combined_objective_functions` rather than being repeated here.

Batch evaluation and term execution
-----------------------------------

The convenience function :py:func:`chemfit.api.evaluate_many` evaluates a
parameter batch through either an executor or a scheduler:

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor

   with ThreadPoolExecutor(max_workers=4) as executor:
       contexts = chemfit.evaluate_many(
           objective,
           [{"x": 1.0}, {"x": 2.0}],
           executor=executor,
       )

It returns populated contexts in parameter-input order. The executor is owned
by the caller. Underneath, it prepares the objective with a
:class:`~chemfit.scheduling.Scheduler`. For direct access to completion order,
schedule reuse, or MPI, use the interfaces described in
:ref:`parallel_execution`.

ASE and external programs
-------------------------

``chemfit.ase_quantity(atoms)`` constructs an
:class:`~chemfit.ase_objective_function.ASEComputer` from an ``ase.Atoms``
object, a path, or a zero-argument atoms factory. The optional ``index`` is
valid only for path inputs. Continue with :ref:`ase_objective_function_api`
for calculators, evaluators, processors, setup callbacks, and caching.

``chemfit.external_quantity(workdir)`` constructs an
:class:`~chemfit.external_computer.ExternalQuantityComputer` rooted at that
directory. Configure it with ``with_hook``, ``with_cmd``, ``with_parser``, and
``wait_for``. See :ref:`external_computer` for the complete isolated-working-
directory and parsing lifecycle.

Going deeper
------------

The convenience helpers are entry points, not a separate object model:

All of :func:`chemfit.quantity() <chemfit.wrap_funcs.quantity>`,
:func:`chemfit.ase_quantity() <chemfit.api.ase_quantity>`, and
:func:`chemfit.external_quantity() <chemfit.api.external_quantity>` return a
subclass of :class:`~chemfit.abstract_objective_function.QuantityComputer`,
which provides the general evaluation semantics and
:meth:`~chemfit.abstract_objective_function.QuantityComputer.with_loss`.

More specifically:

- :func:`chemfit.quantity() <chemfit.wrap_funcs.quantity>` returns a
  :class:`~chemfit.wrap_funcs.WrappedQuantityComputer` for wrapping ordinary
  Python functions; see :ref:`writing_quantity_computers` for custom
  implementations.
- :func:`chemfit.ase_quantity() <chemfit.api.ase_quantity>` returns an
  :class:`~chemfit.ase_objective_function.ASEComputer` used with ASE
  calculators; see :ref:`ase_objective_function_api`.
- :func:`chemfit.external_quantity() <chemfit.api.external_quantity>` returns
  an :class:`~chemfit.external_computer.ExternalQuantityComputer` for working
  with external executables; see :ref:`external_computer`.

The remaining helpers connect these objects to composition and execution:

- :func:`chemfit.combine() <chemfit.api.combine>` returns a
  :class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`;
  see :ref:`combined_objective_functions` for its full behavior.
- :func:`chemfit.fit_nevergrad() <chemfit.api.fit_nevergrad>` drives a
  :class:`~chemfit.fitter.Fitter`; use that class
  directly for SciPy, callbacks, custom optimizers, supplied contexts, or a
  prepared schedule.
- :func:`chemfit.evaluate_many() <chemfit.api.evaluate_many>` prepares a
  :class:`~chemfit.scheduling.Scheduler`; see :ref:`parallel_execution` for
  the lower-level request/result protocol.

The :ref:`concepts` page explains how all of these objects fit together.
