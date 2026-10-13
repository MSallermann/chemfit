.. _objective_hooks:

Evaluation hooks
================

Evaluation hooks attach instrumentation to an
:class:`~chemfit.abstract_objective_function.ObjectiveComponent` without changing
its evaluation implementation. Register either a hook object with
``pre_eval(ctx)`` and/or ``post_eval(ctx)`` methods, or pass ordinary callables
through the ``pre=`` and ``post=`` keyword arguments. No hook base class is
required. Callbacks receive the evaluation context and should store their
output there. Hooks may be registered on a quantity computer, the objective
produced by ``computer.with_loss(...)``, or any other component. The chosen
component determines which evaluation scope and context the hook observes.

Hooks are most useful for behavior that surrounds an evaluation but is not
part of its scientific calculation:

.. list-table:: Common hook use cases
   :header-rows: 1
   :widths: 28 36 36

   * - Use case
     - Hook behavior
     - Context field
   * - Reject unphysical parameters
     - A pre-hook returns a penalty without running the component
     - Record the violated constraint in ``ctx.meta``
   * - Reuse a cached result
     - A pre-hook returns a cache hit; a post-hook stores a cache miss
     - Keep only the per-call cache key or hit status in ``ctx.temp`` or
       ``ctx.meta``
   * - Record timing, convergence, or provenance
     - A pre-hook starts collection and a post-hook finalizes it
     - Use ``ctx.temp`` for transient state and ``ctx.meta`` for the result
   * - Normalize failures or non-finite results
     - A post-hook inspects and, when appropriate, replaces ``ctx.output``
     - Preserve diagnostic details in ``ctx.meta``

The cache itself is not portable shared context state. For process or MPI
execution, use a backend-appropriate external cache whose client can be
created or serialized safely on each worker.

Built-in hooks
--------------

``UUIDHook`` assigns a fresh UUID string to ``ctx.meta["evaluation_id"]`` before
each evaluation. ``TimingHook`` records elapsed wall-clock time in
``ctx.meta["timing"]["elapsed_seconds"]`` after evaluation, including when the
objective raises. Both accept a ``meta_key`` argument to customize their output
key.

.. testcode::

    import chemfit
    from chemfit.objective_hooks import TimingHook, UUIDHook

    @chemfit.objective()
    def objective(params):
        return params["x"] ** 2

    objective.register_eval_hook(hook=UUIDHook()).register_eval_hook(
        hook=TimingHook()
    )

    ctx = chemfit.EvaluateContext()
    loss = objective({"x": 2.0}, ctx)
    print(loss)
    print(isinstance(ctx.meta["evaluation_id"], str))
    print(ctx.meta["timing"]["elapsed_seconds"] >= 0.0)

.. testoutput::
    :hide:

    4.0
    True
    True

``chemfit.objective()`` returns a wrapped
:class:`~chemfit.abstract_objective_function.ObjectiveComponent`; the same hook
API applies to every component, including quantity computers and composed
objective terms.

``register_eval_hook`` returns the component, allowing chained registrations.
Register metadata-producing post-hooks before consumers such as a logger, so
the consumer sees their output. Registering the same callback again on the same
component is a no-op; distinct callbacks are appended in registration order.
In particular, do not install multiple distinct ``TimingHook`` instances on
the same evaluation scope, even with different output keys: they use the same
scratch attribute.

Writing a custom hook
---------------------

Use ``ctx.temp`` for temporary per-call state and ``ctx.meta`` for results that
should be included in collected metadata. A practical pre-hook can reject an
unphysical parameter proposal before an expensive simulation starts:

.. testcode::

    @chemfit.objective()
    def lj_fit_loss(params):
        return (params["epsilon"] - 1.0) ** 2

    class PositiveParametersHook:
        def __init__(self, *names):
            self.names = names

        def pre_eval(self, ctx):
            invalid = [name for name in self.names if ctx.input[name] <= 0]
            if invalid:
                ctx.meta["rejected_parameters"] = invalid
                return float("inf")

    lj_fit_loss.register_eval_hook(
        hook=PositiveParametersHook("epsilon", "sigma")
    )

    valid_ctx = chemfit.EvaluateContext()
    rejected_ctx = chemfit.EvaluateContext()

    print(lj_fit_loss({"epsilon": 0.5, "sigma": 1.0}, valid_ctx))
    print(lj_fit_loss({"epsilon": 0.5, "sigma": -1.0}, rejected_ctx))
    print(rejected_ctx.meta["rejected_parameters"])

.. testoutput::
    :hide:

    0.25
    inf
    ['sigma']

For a one-sided hook, simply omit the other method. Keep hook instance
attributes for configuration, not mutable per-evaluation state.

A post-hook may transform the objective result by assigning ``ctx.output``.
The transformed value is authoritative for downstream components and callers.

Registering functions directly
------------------------------

For callbacks that do not need a configuration object, pass functions
directly with ``pre=`` and ``post=``. Here the callbacks collect optimizer
diagnostics without changing the objective's return value:

.. testcode::

    import math

    def record_parameter_scale(ctx):
        ctx.meta["max_abs_parameter"] = max(map(abs, ctx.input.values()))

    def record_loss_diagnostics(ctx):
        if not isinstance(ctx.output, Exception):
            ctx.meta["finite_loss"] = math.isfinite(ctx.output)

    objective.register_eval_hook(
        pre=record_parameter_scale,
        post=record_loss_diagnostics,
    )

    ctx = chemfit.EvaluateContext()
    print(objective({"x": 4.0}, ctx))
    print(ctx.meta["max_abs_parameter"])
    print(ctx.meta["finite_loss"])

.. testoutput::
    :hide:

    16.0
    4.0
    True

Either keyword may be supplied on its own, or both may be registered in one
call. At least one callback is required. The ``hook=`` form is mutually
exclusive with ``pre=`` and ``post=``; use either a hook object or direct
callbacks in a given registration. ``recursive=True`` applies the supplied
pre- and post-callbacks to the current descendant objectives in exactly the
same way as it does for a hook object. Recursion follows graph-visible
:class:`~chemfit.abstract_objective_function.StructuralComponent` children; it
does not enter the stages of an
:class:`~chemfit.abstract_objective_function.OpaquePipeline`.

Lifecycle and errors
--------------------

Each objective call follows this order:

1. Reset per-call context state and set ``ctx.input`` to the supplied value.
2. Run pre-hooks in registration order.
3. Unless a pre-hook returned a result, run ``_run``.
4. Merge the component's static metadata into ``ctx.meta``.
5. Assign the result or exception to ``ctx.output``.
6. Run post-hooks in registration order, not reverse order.
7. Return the final value of ``ctx.output`` or raise it if it is an exception.


For a combined objective, each leaf's final value is
what enters its parent reduction. A post-hook on the root combined objective
may then replace the reduced loss. These semantics are the same for direct
calls and prepared schedulers, including graph schedulers.

If ``_run`` or a pre-hook raises, post-hooks still run and see that exception
in ``ctx.output``. A pre-hook may instead return a value to short-circuit
``_run`` successfully. More than one non-``None`` pre-hook result is an error.

If a post-hook raises, its exception replaces ``ctx.output`` and becomes the
final failure; later post-hooks do not run. This is also how a post-hook can
replace an earlier evaluation failure.

Contexts can be reused: input, output, metadata, scratch data, pipeline state,
and incoming links are reset at call entry, while ``config`` and ``shared``
are preserved. Use a separate context for every concurrent evaluation.

Nested objectives
-----------------

By default, registration affects only the given objective. To instrument a
combined objective and its nested terms, register after building the call graph:

.. testcode::

    import chemfit
    from chemfit.objective_hooks import TimingHook

    inner = chemfit.combine(lambda params: params["x"] ** 2)
    combined = chemfit.combine(inner, lambda params: params["x"] + 1)
    combined.register_eval_hook(hook=TimingHook(), recursive=True)

    ctx = chemfit.EvaluateContext()
    print(combined({"x": 2.0}, ctx))
    child_timing = ctx.incoming[0].meta["timing"]
    print("elapsed_seconds" in child_timing)

.. testoutput::
    :hide:

    7.0
    True

Each term runs with its own child context. Child contexts are available through
``ctx.incoming``. Further descendants are available through each child's
``incoming`` property.
Parent timing includes nested work; child timing measures the individual term.

Recursive registration follows distinct child evaluation scopes through
combined objectives and fitter wrappers. Schedulers change where leaf
objectives run; they do not introduce another objective or hook scope. Register
hooks on the combined objective and, for MPI, construct the same hooked
objective on every rank.

Recursive registration modifies the descendant component instances directly;
it does not create composition-local registrations. If the same descendant
instance is also used in another pipeline or join, the recursively registered
hook is therefore active there as well. This is intentional: compositions
share their components by reference. Construct the compositions with distinct
component copies when their hook configuration must be isolated.

The same hook instance is registered throughout the traversed graph. Custom
structural components expose their child scopes through ``components()``.

Parallel execution
------------------

Register hooks before submitting work to a process executor. For MPI, register
term hooks on every rank before nonzero ranks enter
``MPIGraphSchedule.worker_loop()``; registering only on rank zero does not
update worker objectives.

Hooks can execute concurrently or in another process. Do not append to a
captured shared list, mutate global state, or keep start times on the hook
instance. Such writes can race between threads and do not propagate back from
worker processes. Store observations in ``ctx.meta`` instead, which executor
and MPI schedules transport with evaluation results. ``ctx.temp`` is local
scratch state and is not transported.

Process-based hooks must be serializable. Prefer module-level hook classes
with simple configuration attributes; avoid storing live connections,
executors, or locks on hook instances. See :ref:`mpi` and
:doc:`parallel_execution` for the execution setup.
