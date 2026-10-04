.. _public_api:

Small public API (draft)
========================

This thin facade implements the common workflow proposed in ``api_2.md``.
The existing classes implement signature inference, reductions, and execution;
the facade only supplies shorter names and convenience functions. Existing
modules, constructors, decorators, and extension points remain available.

Compute, attach a loss, combine, fit
------------------------------------------

.. code-block:: python

    import chemfit

    @chemfit.quantity
    def simulate(params):
        return {"density": params["sigma"] ** 2}

    def density_loss(q, reference):
        return (q["density"] - reference) ** 2

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

The one-shot ``fit`` function currently uses Nevergrad and returns a
``FitResult`` containing the optimizer recommendation and evaluation contexts.
Its two concurrency settings have separate meanings:

* ``workers`` is the number of candidate slots exposed to Nevergrad and the
  maximum number of candidates in one ask/evaluate/tell batch.
* ``execution_workers`` is the maximum number of objective leaf tasks run at
  once by the built-in thread scheduler. It defaults to ``workers``.

Leaf tasks include independent terms within a combined objective, so the
execution limit applies across both candidates and terms. For example,
``workers=4, execution_workers=8`` asks Nevergrad for batches of up to four
candidates while allowing up to eight leaf tasks from that batch to execute at
once.

Supplying ``executor=`` or ``scheduler=`` disables the built-in scheduler;
that object then determines execution concurrency, and ``execution_workers``
must be omitted. Use an explicit process executor for CPU-bound Python work.

For advanced configuration or SciPy, use ``chemfit.Fitter``. Its constructor
accepts ``initial=`` or the existing ``initial_params=`` spelling, but not both.
It is the same class as ``chemfit.fitter.Fitter``, not a separate facade class.

Direct objectives and contexts
------------------------------

.. code-block:: python

    @chemfit.objective
    def square(params, *, ctx):
        ctx.meta["kind"] = "square"
        return params["x"] ** 2

    ctx = chemfit.Context()
    loss = square({"x": 2.0}, ctx=ctx)

``Context`` is an alias for ``EvaluateContext`` and ``Objective`` is an alias
for ``ObjectiveFunctor``. These are not replacements with different runtime
semantics. A context is created automatically if omitted.

Optional keyword-only injection
------------------------------------------

The facade inspects callable signatures when adapting them:

* Quantity functions and direct objectives may request ``ctx``.
* Losses may request ``params`` and ``ctx``.
* Aggregators receive child ``quantities`` and the parent ``ctx``.

Injected arguments must be explicitly named and keyword-only. A bare
``**kwargs`` does not request injection. The primary input is positional; its
name is not prescribed. For example:

.. code-block:: python

    def loss(q, reference, *, params, ctx):
        ctx.meta["reference"] = reference
        return (q["density"] - reference) ** 2 + params["sigma"] ** 2

    def aggregator(terms, quantities, ctx):
        ctx.meta["reduced_terms"] = len(terms)
        return sum(terms)

    term = simulate.with_loss(loss, reference=1.4)
    objective = chemfit.combine(term, aggregator=aggregator)

Use ``.bind(...)`` for quantity-function configuration and ``.with_loss(...,
reference=...)`` for loss configuration. Loss ``params`` and ``ctx`` are
reserved for injection, not configuration keywords.

Loss errors are propagated unchanged, without catching ``TypeError`` and
retrying the user function with another calling convention. This now applies
to all quantity computers, not just the short decorator. Legacy positional
``loss(q, params)`` signatures are resolved at construction; the parameter
argument must be named ``params``, ``parameters``, or ``p``. Bind other required
configuration arguments explicitly through ``with_loss``.

Existing function-wrapper constructors and decorators also infer keyword-only
``ctx`` by default. Explicit ``pass_ctx=True`` or ``False`` remains available
for compatibility, but is unnecessary for the common path.

Composition and term execution
------------------------------

``combine(a, b)`` and ``combine([a, b])`` both accept existing objectives or
plain functions. Plain functions are adapted using the same context-injection
rules. Supply ``weights=[...]`` and either a simple ``reduction=`` callable or
a context-aware ``aggregator=`` callable. The two options are mutually exclusive.

To parallelize terms, provide a caller-owned executor:

.. code-block:: python

    from concurrent.futures import ThreadPoolExecutor
    from chemfit.executor_policy import ExecutorPolicy
    from chemfit.objective_hooks import TimingHook

    with ThreadPoolExecutor(4) as executor:
        objective = chemfit.combine(square, density_term)
        objective.execution_policy = ExecutorPolicy(executor)
        objective.register_eval_hook(TimingHook(), recursive=True)
        loss = objective({"x": 2.0, "sigma": 1.0})

Hooks stay on the returned combined objective and run once around its
evaluation. Its execution policy controls term scheduling. ``ExecutorPolicy``
uses ``ctx.executor`` when present, otherwise its configured executor, and
otherwise creates a thread pool lazily.

Executors are driver-local resources. Do not assume candidate process
parallelism automatically gives nested term parallelism.

As before, separate terms sharing a quantity computer still compute it
separately: no dependency-graph caching is introduced.

Manual fitting
--------------

.. code-block:: python

    fitter = chemfit.Fitter(square, initial={"x": 1.0})
    fitter.start(workers=1)
    try:
        loss = fitter.evaluate({"x": 2.0})
        # Feed loss to your external optimizer here.
        fitter.step()
    finally:
        result = fitter.finish()

``start``, ``evaluate``, and ``step`` delegate to the existing ``init``, ``ask``,
and ``tell`` methods. They preserve existing batching and context-index
behavior; ``evaluate`` does not yet accept an arbitrary ``ctx`` argument.
Call ``finish`` to close resources owned by a manually started fitter session.
The old method names remain available.

Deferred design work
--------------------

The following proposals remain outside this draft: backend classes and MPI
facades, a session context manager, declarative error policies, shared quantity
computation, a rich fit result object, and hiding or renaming existing internal
classes. They should be designed independently rather than silently changing
the existing lower-level API.
