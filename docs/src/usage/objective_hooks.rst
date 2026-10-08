.. _objective_hooks:

Evaluation hooks
================

Evaluation hooks attach instrumentation to an
:class:`~chemfit.abstract_objective_function.ObjectiveFunctor` without changing
its evaluation implementation. Register either a hook object with
``pre_eval(ctx)`` and/or ``post_eval(ctx)`` methods, or pass ordinary callables
through the ``pre=`` and ``post=`` keyword arguments. No hook base class is
required. Callbacks receive the evaluation context and should store their
output there. Register them on the objective produced by
``computer.with_loss(...)``, not on the quantity computer itself.

Built-in hooks
--------------

``UUIDHook`` assigns a fresh UUID string to ``ctx.meta["evaluation_id"]`` before
each evaluation. ``TimingHook`` records elapsed wall-clock time in
``ctx.meta["timing"]["elapsed_seconds"]`` after evaluation, including when the
objective raises. Both accept a ``meta_key`` argument to customize their output
key.

.. code-block:: python

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
    print(ctx.meta["evaluation_id"])
    print(ctx.meta["timing"]["elapsed_seconds"])

``chemfit.objective()`` returns a wrapped
:class:`~chemfit.abstract_objective_function.ObjectiveFunctor`; the same hook
API applies to custom and composed objective functors.

``register_eval_hook`` returns the objective, allowing chained registrations.
Register metadata-producing post-hooks before consumers such as a logger, so
the consumer sees their output. Registering a hook again appends its callbacks;
it does not replace an earlier registration. In particular, do not install
multiple ``TimingHook`` instances on the same evaluation scope, even with
different output keys: they use the same scratch attribute.

Writing a custom hook
---------------------

Use ``ctx.temp`` for temporary per-call state and ``ctx.meta`` for results that
should be included in collected metadata. The following symmetric hook records
whether evaluation succeeded:

.. code-block:: python

    class StatusHook:
        def pre_eval(self, ctx):
            ctx.meta["status"] = "running"

        def post_eval(self, ctx):
            ctx.meta["status"] = (
                "failed" if ctx.temp.exception is not None else "completed"
            )

    objective.register_eval_hook(StatusHook())

For a one-sided hook, simply omit the other method. Keep hook instance
attributes for configuration, not mutable per-evaluation state.

A successful post-hook may also transform the objective result by replacing
``ctx.loss``. The replacement becomes both the stored context loss and the
value returned to the caller:

.. code-block:: python

    def clamp_loss(ctx):
        if ctx.loss is not None:
            ctx.loss = max(ctx.loss, 0.0)

    objective.register_eval_hook(post=clamp_loss)

Registering functions directly
------------------------------

For callbacks that do not need a configuration object, pass functions
directly with ``pre=`` and ``post=``:

.. code-block:: python

    def mark_running(ctx):
        ctx.meta["status"] = "running"

    def record_outcome(ctx):
        ctx.meta["status"] = (
            "failed" if ctx.temp.exception is not None else "completed"
        )

    objective.register_eval_hook(
        pre=mark_running,
        post=record_outcome,
    )

Either keyword may be supplied on its own, or both may be registered in one
call. At least one callback is required. The ``hook=`` form is mutually
exclusive with ``pre=`` and ``post=``; use either a hook object or direct
callbacks in a given registration. ``recursive=True`` applies the supplied
pre- and post-callbacks to the current descendant objectives in exactly the
same way as it does for a hook object.

Lifecycle and errors
--------------------

Each objective call follows this order:

1. Set ``ctx.parameters`` to the supplied mapping and reset ``ctx.loss`` to
   ``None``.
2. Run pre-hooks in registration order.
3. Run ``_evaluate`` and assign its return value to ``ctx.loss``.
4. Run post-hooks in registration order, not reverse order.
5. Return the final value of ``ctx.loss``.

On a successful evaluation, a post-hook may replace ``ctx.loss``. Later
post-hooks observe the replacement, and the final value is the objective's
caller-visible result. For a combined objective, each leaf's final value is
what enters its parent reduction. A post-hook on the root combined objective
may then replace the reduced loss. These semantics are the same for direct
calls and prepared schedulers, including tree schedulers.

If ``_evaluate`` raises, post-hooks still run. They see ``ctx.loss is None`` and
the exception in ``ctx.temp.exception``. On successful evaluation,
``ctx.temp.exception`` is ``None`` before post-hooks run.

Ordinary exceptions from post-hooks are collected while the remaining
post-hooks are attempted. If evaluation succeeded, ChemFit raises
``ObjectiveFunctor.PostEvalHookError``; its ``exceptions`` tuple contains the
individual failures in registration order. If evaluation failed, its original
exception is re-raised instead, and post-hook failures are not reported
separately. Process-control exceptions such as ``KeyboardInterrupt`` are not
collected from post-hooks.

A failing pre-hook aborts the call immediately: later pre-hooks, evaluation,
and post-hooks do not run. Hooks therefore cannot rely on post-hooks for
cleanup if a pre-hook fails.

Contexts can be reused, but only parameters and loss are reset at call entry.
Metadata, quantities, and scratch values may still contain previous state.
Initialize any values your hook owns in its pre-hook. Use a separate context
for every concurrent evaluation.

Nested objectives
-----------------

By default, registration affects only the given objective. To instrument a
combined objective and its nested terms, register after building the call tree:

.. code-block:: python

    from chemfit.combined_objective_function import CombinedObjectiveFunction

    inner = CombinedObjectiveFunction([
        WrappedObjectiveFunctor(lambda params: params["x"] ** 2),
    ])
    combined = CombinedObjectiveFunction([
        inner,
        WrappedObjectiveFunctor(lambda params: params["x"] + 1),
    ])
    combined.register_eval_hook(TimingHook(), recursive=True)

    ctx = EvaluateContext()
    combined({"x": 2.0}, ctx)
    child_timing = ctx.children[0].meta["timing"]

Each term runs with its own child context. Child contexts are available through
``ctx.children`` and expose ``parameters``, ``quantities``, ``loss``, and
``meta`` directly. Further descendants are available through each child's
``children`` property.
Parent timing includes nested work; child timing measures the individual term.

Recursive registration follows distinct child evaluation scopes through
combined objectives and fitter wrappers. Schedulers change where leaf
objectives run; they do not introduce another objective or hook scope. Register
hooks on the combined objective and, for MPI, construct the same hooked
objective on every rank.

The same hook instance is registered throughout the current tree. Use distinct
objective instances for distinct positions in the tree; recursive registration
does not deduplicate references or detect cycles. Custom composite objectives
can expose their child scopes by implementing ``_child_objectives()``.

Parallel execution
------------------

Register hooks before submitting work to a process executor. For MPI, register
term hooks on every rank before nonzero ranks enter
``MPITreeSchedule.worker_loop()``; registering only on rank zero does not
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
