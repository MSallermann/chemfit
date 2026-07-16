.. _fitter:

##################################
Fitter
##################################

The :py:class:`~chemfit.fitter.Fitter` class drives optimization of
objective functions defined on parameter dictionaries.

The Fitter is responsible for driving the optimization process, but
does not impose any structure on the objective beyond accepting
parameter dictionaries.

It provides a uniform interface to different optimization backends and
adds fitter-specific functionality such as:

- tracking the number of evaluations
- tracking the best loss, parameters, and metadata
- handling invalid return values
- optional exception handling
- callback hooks for monitoring and persistence

----------------------------------
Supported backends
----------------------------------

ChemFit currently supports two optimization backends:

1. :py:meth:`~chemfit.fitter.Fitter.fit_scipy`
2. :py:meth:`~chemfit.fitter.Fitter.fit_nevergrad`

Custom optimizers can be connected through the user-driven
``init``/``ask``/``tell`` interface.

Both operate on the same parameter-dictionary interface.

----------------------------------
Basic usage
----------------------------------

The minimal setup requires:

1. an objective function
2. a dictionary of initial parameters

.. code-block:: python

    from chemfit.fitter import Fitter

    def objective(params, ctx=None):
        return 2.0 * (params["x"] - 2)**2 + 3.0 * (params["y"] + 1)**2

    fitter = Fitter(
        objective_function=objective,
        initial_params={"x": 0.0, "y": 0.0},
    )

    opt_params = fitter.fit_scipy()
    print(opt_params)

The same objective can also be optimized with Nevergrad:

.. code-block:: python

    opt_params = fitter.fit_nevergrad(budget=100)

----------------------------------
Objective interface
----------------------------------

The fitter accepts either:

- a plain callable ``f(params) -> float``
- an :py:class:`~chemfit.abstract_objective_function.ObjectiveFunctor`

Internally, the objective is wrapped in a
:py:class:`~chemfit.fitter.FitterObjectiveFunctor`, which adds
robustness checks and bookkeeping.

Advanced objectives may also accept an evaluation context:

.. code-block:: python

    f(params, ctx) -> float

If no context is provided, ChemFit creates one automatically.

----------------------------------
Parameter dictionaries
----------------------------------

The parameter dictionary may be nested to arbitrary depth, as long as
all leaf values are numeric.

.. code-block:: python

    params = {
        "pair": {
            "epsilon": 1.0,
            "sigma": 2.0,
        },
        "threebody": {
            "lambda": 3.0,
        },
    }

Internally, the parameter dictionary is flattened before being passed
to the optimizer and reconstructed on each evaluation.

----------------------------------
Bounds
----------------------------------

Bounds are specified as a dictionary mirroring the structure of the
parameter dictionary. Each bounded leaf is given as a ``(lower, upper)``
tuple.

.. code-block:: python

    bounds = {
        "pair": {
            "epsilon": (0.0, 5.0),
            "sigma": (1.0, 4.0),
        }
    }

Bounds may be omitted for individual parameters.

----------------------------------
FitterEvaluateContext
----------------------------------

During optimization, ChemFit uses
:py:class:`~chemfit.fitter.FitterEvaluateContext`, which extends
:py:class:`~chemfit.abstract_objective_function.EvaluateContext`
with optimization-specific fields:

- ``n_evals``: number of evaluations
- ``opt_loss``: best loss seen so far
- ``opt_params``: best parameters seen so far
- ``opt_meta``: metadata associated with the best evaluation

For SciPy, a single context is used for the entire optimization.
For Nevergrad, one context is used per worker.

These contexts are available via ``fitter.contexts``.

----------------------------------
Callbacks
----------------------------------

Callbacks can be registered with
:py:meth:`~chemfit.fitter.Fitter.register_callback`.

Each callback has the form:

.. code-block:: python

    def callback(step: int, contexts: list[FitterEvaluateContext]) -> None:
        ...

Callbacks are invoked every ``n_steps`` optimizer steps.

.. code-block:: python

    def print_progress(step, contexts):
        best = contexts[0].opt_loss
        print(step, best)

    fitter.register_callback(print_progress, n_steps=5)

You may register multiple callbacks. They are executed in order of
registration.

Note that ``step`` refers to optimizer iterations, not necessarily the
number of objective evaluations.

----------------------------------
Predefined callbacks
----------------------------------

ChemFit provides ready-to-use callback utilities in
:py:mod:`chemfit.fitter_callbacks`.

These implement common patterns such as:

- logging optimization progress
- saving evaluation metadata to disk
- checkpointing the best parameters seen so far

.. code-block:: python

    from chemfit.fitter_callbacks import (
        CheckpointBestParameters,
        SaveMetaData,
        log_progress,
    )

    fitter.register_callback(log_progress, n_steps=10)
    fitter.register_callback(SaveMetaData("meta"), n_steps=20)
    fitter.register_callback(CheckpointBestParameters("best.json"), n_steps=5)

Saving and replaying evaluations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~chemfit.fitter_callbacks.SaveMetaData` callback pairs naturally
with the ``initial_observations`` argument of
:py:meth:`~chemfit.fitter.Fitter.fit_nevergrad`.

It allows evaluation results to be persisted and later reused to seed a
new optimization:

.. code-block:: python

    opt_params = fitter.fit_nevergrad(
        budget=100,
        initial_observations=loaded_observations,
    )

This enables:

- approximate continuation of interrupted runs
- warm-starting optimization
- reuse of expensive evaluations

In contrast, :class:`~chemfit.fitter_callbacks.CheckpointBestParameters`
stores only the best solution and is primarily intended for fault
tolerance.

----------------------------------
SciPy backend
----------------------------------

:py:meth:`~chemfit.fitter.Fitter.fit_scipy` uses
``scipy.optimize.minimize``.

.. code-block:: python

    opt_params = fitter.fit_scipy(method="L-BFGS-B")

This backend is synchronous and uses a single context.

----------------------------------
Nevergrad backend
----------------------------------

:py:meth:`~chemfit.fitter.Fitter.fit_nevergrad` uses Nevergrad's
ask/tell interface.

.. code-block:: python

    opt_params = fitter.fit_nevergrad(
        budget=200,
        optimizer_str="NgIohTuned",
    )

----------------------------------
User-supplied ask/tell interface
----------------------------------

An optimizer can be integrated without a dedicated backend. The user owns the
loop while ChemFit evaluates parameter dictionaries, maintains contexts and
dispatches its callbacks.

.. warning::

    The ``initial_params`` and ``bounds`` passed to :class:`~chemfit.fitter.Fitter`
    are not used to initialize or constrain a user-supplied optimizer. Configure
    them directly on the optimizer; ``fitter.ask`` evaluates any parameter
    dictionary it receives, including candidates outside those bounds.

.. code-block:: python

    optimizer = MyOptimizer(initial_params, bounds)
    fitter.init()

    for _ in range(100):
        params = optimizer.ask()
        loss = fitter.ask(params)
        optimizer.tell(params, loss)
        fitter.tell()  # dispatch registered progress callbacks

    opt_params = fitter.finish(optimizer.recommendation())

``fitter.ask`` also accepts a list of candidates and evaluates it in parallel
when ``fitter.init`` is configured with ``num_workers`` and an executor. If no
recommendation is passed to ``finish``, ChemFit returns the best parameters it
actually evaluated.

A complete custom-loop example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optimizer only needs to propose parameter dictionaries and accept their
losses. For example, this small optimizer searches a predefined set of points:

.. code-block:: python

    class CandidateSearch:
        def __init__(self, candidates):
            self.candidates = iter(candidates)
            self.observations = []

        def ask(self):
            return next(self.candidates)

        def tell(self, params, loss):
            self.observations.append((loss, params))

        def recommendation(self):
            return min(self.observations, key=lambda item: item[0])[1]


    def objective(params):
        return (params["x"] - 2.0) ** 2


    fitter = Fitter(objective, initial_params={"x": 0.0})
    optimizer = CandidateSearch([{"x": 0.0}, {"x": 2.0}, {"x": 4.0}])

    fitter.register_callback(print_progress, n_steps=1)
    fitter.init()

    for _ in range(3):
        params = optimizer.ask()
        loss = fitter.ask(params)
        optimizer.tell(params, loss)
        fitter.tell()

    opt_params = fitter.finish(optimizer.recommendation())

Calling ``fitter.ask`` applies the same objective wrapping, invalid-value
handling and context bookkeeping as the built-in SciPy and Nevergrad fitting
methods. Calling ``fitter.tell`` marks the end of an optimizer step and invokes
callbacks registered for that step. ``fitter.finish`` runs the usual post-fit
checks.

User-driven parallel batches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Candidates can also be evaluated in batches while the user retains control of
the optimizer loop:

.. code-block:: python

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(4) as executor:
        fitter.init(num_workers=4, executor=executor)

        for _ in range(25):
            candidates = [optimizer.ask() for _ in range(4)]
            losses = fitter.ask(candidates)

            for params, loss in zip(candidates, losses):
                optimizer.tell(params, loss)

            fitter.tell()

        opt_params = fitter.finish(optimizer.recommendation())

One :class:`~chemfit.fitter.FitterEvaluateContext` is maintained per worker.
The losses returned by ``fitter.ask`` have the same order as the candidate
list. If no executor is supplied, ``fitter.init(num_workers=...)`` creates and
later shuts down a thread pool automatically.

----------------------------------
Parallel Nevergrad execution
----------------------------------

Parallel evaluation is supported via ``num_workers``:

.. code-block:: python

    opt_params = fitter.fit_nevergrad(
        budget=100,
        num_workers=4,
    )

The evaluation budget is exact. When it is not divisible by ``num_workers``,
the final batch contains only the remaining candidates. For example,
``budget=10`` with four workers evaluates batches of four, four and two.

Each worker uses its own
:py:class:`~chemfit.fitter.FitterEvaluateContext`.

An executor may be provided:

.. code-block:: python

    from concurrent.futures import ThreadPoolExecutor

    opt_params = fitter.fit_nevergrad(
        budget=100,
        num_workers=4,
        executor=ThreadPoolExecutor(4),
    )

When using parallel execution, objective functions must avoid modifying
shared state outside of the evaluation context.

----------------------------------
Configuring the contexts
----------------------------------

In some cases you may wish to configure or persist the contexts.
This can, for example, be necessary if terms require special fields in the
``ctx.config``.

For this reason both, :py:meth:`~chemfit.fitter.Fitter.fit_scipy` and
:py:meth:`~chemfit.fitter.Fitter.fit_nevergrad` support passing in
external contexts as an argument.

For example:

.. code-block:: python

    from chemfit.fitter import Fitter, FitterEvaluateContext

    fitter = Fitter(...)

    ctxs = [FitterEvaluateContext() for _ in range(NUM_WORKERS)]

    for ctx in ctxs:
        ctx.config.gandalf = "the white" # <-- make sure it's not the grey

    fitter.fit_nevergrad(..., ctxs)

.. important::

    Make sure to pass instances of :py:class:`~chemfit.fitter.FitterEvaluateContext` and **not** the base class
    :py:class:`~chemfit.abstract_objective_function.EvaluateContext`!

.. warning::

    Beware of the **anti**- pattern:

    .. code-block:: python

        ctxs = [FitterEvaluateContext()] * NUM_WORKERS # <-- bad

    This will create a list with ``NUM_WORKERS`` references to the **same** ``ctx``. Very bad!


----------------------------------
Initial observations
----------------------------------

The Nevergrad backend can be seeded with previously observed
``(parameters, loss)`` pairs:

.. code-block:: python

    observations = [
        ({"x": 0.0, "y": 0.0}, 10.0),
        ({"x": 2.0, "y": -1.0}, 0.0),
    ]

    opt_params = fitter.fit_nevergrad(
        budget=100,
        initial_observations=observations,
    )

These observations are replayed into the optimizer before the main loop.

- observations violating bounds are skipped
- they do not count towards the evaluation budget
- they do not trigger callbacks

This does not restore the internal state of the optimizer.

----------------------------------
Robustness features
----------------------------------

The fitter adds several safety mechanisms:

Invalid returns
    Non-numeric or NaN values are replaced by ``value_bad_params``.

Exceptions
    Exceptions can be logged and optionally swallowed.

Near-bound warnings
    If ``near_bound_tol`` is set, parameters near bounds trigger warnings.

----------------------------------
Lifecycle hooks
----------------------------------

Each fit runs through:

- :py:meth:`~chemfit.fitter.Fitter._hook_pre_fit`
- :py:meth:`~chemfit.fitter.Fitter._hook_post_fit`

These are mainly intended for subclassing.

----------------------------------
Summary
----------------------------------

- Works with parameter dictionaries (possibly nested)
- Supports SciPy and Nevergrad backends
- Adds robustness and tracking via a wrapper objective
- Uses ``FitterEvaluateContext`` for evaluation bookkeeping
- Supports callbacks and predefined callback utilities
- Allows replaying observations for warm-starting
- Supports parallel evaluation with Nevergrad
