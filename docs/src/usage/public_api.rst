.. _public_api:

Public API
==========

The :mod:`chemfit` package exports the common workflow directly: wrap a
quantity computation, attach a loss, combine terms, and fit parameters. The
lower-level classes remain available for custom execution and optimization
loops.

Compute, attach a loss, combine, fit
------------------------------------

.. code-block:: python

   import chemfit

   @chemfit.quantity()
   def simulate(params):
       return {"density": params["sigma"] ** 2}

   def density_loss(quantities, *, reference):
       return (quantities["density"] - reference) ** 2

   density_term = simulate.with_loss(density_loss, reference=1.4)
   objective = chemfit.combine(density_term)

   result = chemfit.fit(
       objective,
       initial={"sigma": 1.0},
       bounds={"sigma": (0.1, 3.0)},
       optimizer="NgIohTuned",
       budget=100,
       workers=4,
   )

:py:func:`chemfit.api.fit` uses Nevergrad and returns a
:class:`~chemfit.api.FitResult`. ``result.recommendation`` is Nevergrad's
recommended parameter mapping. ``best_parameters`` and ``best_loss`` identify
the best optimizer-visible evaluation recorded by ChemFit, and ``contexts``
contains one :class:`~chemfit.fitter.FitterEvaluateContext` per candidate slot.

Concurrency in ``fit``
----------------------

The two worker settings have separate meanings:

- ``workers`` is the number of candidate slots exposed to Nevergrad and the
  maximum number of candidates in one ask/evaluate/tell batch.
- ``execution_workers`` is the maximum number of objective leaf tasks run at
  once by the built-in thread scheduler. It defaults to ``workers``.

Leaf tasks include independent terms within a combined objective, so the
execution limit applies across both candidates and terms. For example,
``workers=4, execution_workers=8`` asks Nevergrad for batches of up to four
candidates while allowing up to eight leaf tasks from the batch to execute at
once.

Supplying ``executor=`` or ``scheduler=`` replaces the built-in scheduler.
That object determines execution concurrency, and ``execution_workers`` must
be omitted. ``executor`` and ``scheduler`` are mutually exclusive. Use a
process executor for CPU-bound Python work that does not release the GIL.

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
only its parameter mapping plus arguments configured with ``bind``.

Configuring wrapped functions and losses
----------------------------------------

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

A loss function must accept either ``loss(quantities)`` or the legacy
two-positional-argument form ``loss(quantities, parameters)`` after its
configuration arguments are bound. Its signature is inspected when the
objective is constructed. A ``TypeError`` raised inside the loss is propagated
without retrying another calling convention.

Composition
-----------

:py:func:`chemfit.api.combine` accepts objective terms as separate positional
arguments:

.. code-block:: python

   objective = chemfit.combine(term_a, term_b, weights=[1.0, 0.5])

Each term may be an
:class:`~chemfit.abstract_objective_function.ObjectiveFunctor` or a plain
``objective(parameters) -> float`` callable. Plain callables are wrapped
without context injection.

Use ``reduction=`` for a callable that receives the successful weighted term
values. Use ``aggregator=`` when reduction also needs child quantities and the
parent context:

.. code-block:: python

   def aggregate(terms, quantities, ctx):
       ctx.meta["terms_with_quantities"] = sum(q is not None for q in quantities)
       return sum(terms)

   objective = chemfit.combine(term_a, term_b, aggregator=aggregate)

``reduction`` and ``aggregator`` are mutually exclusive. Separate terms that
share a quantity computer still compute it separately; composition does not
introduce dependency caching.

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
by the caller. For direct access to completion order, schedule reuse, or MPI,
use the scheduler interfaces described in :ref:`parallel_execution`.

ASE and external programs
-------------------------

``chemfit.ase_quantity(atoms)`` constructs an
:class:`~chemfit.ase_objective_function.ASEComputer` from an ``ase.Atoms``
object, a path, or a zero-argument atoms factory. The optional ``index`` is
valid only for path inputs.

``chemfit.external_quantity(workdir)`` constructs an
:class:`~chemfit.external_computer.ExternalQuantityComputer` rooted at that
directory. Configure it with ``with_hook``, ``with_cmd``, ``with_parser``, and
``wait_for``. See :ref:`ase_objective_function_api` and
:ref:`external_computer` for their complete lifecycles.

Manual fitting
--------------

For SciPy or a user-owned optimization loop, use
:class:`~chemfit.fitter.Fitter`. Its constructor uses the
``initial_params=`` spelling and accepts a scheduler:

.. code-block:: python

   fitter = chemfit.Fitter(square, initial_params={"x": 1.0})
   fitter.init(num_workers=1)
   loss = fitter.evaluate({"x": 2.0})
   # Feed loss to the external optimizer here.
   fitter.step()
   optimum = fitter.finish()

``num_workers`` limits candidate batch size; the fitter's configured scheduler
controls actual execution. ``evaluate`` accepts one mapping or a list of at
most ``num_workers`` mappings, ``step`` dispatches callbacks for a completed
optimizer step, and ``finish`` runs final callbacks and closes the prepared
schedule. See :ref:`fitter` for the complete API.
