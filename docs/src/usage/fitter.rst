Fitter
==================================

The :py:class:`~chemfit.fitter.Fitter` class is a lightweight wrapper around optimization
backends that minimizes objective functions defined in terms of parameter dictionaries.

It provides a consistent interface for both local and global optimizers, automatic logging,
and flexible callback hooks for monitoring progress.

Supported backends
----------------------------------

1. `SciPy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
   via :py:meth:`~chemfit.fitter.Fitter.fit_scipy`
2. `Nevergrad <https://github.com/facebookresearch/nevergrad>`_
   via :py:meth:`~chemfit.fitter.Fitter.fit_nevergrad`

Both operate on the same callable objective interface.

Basic usage
----------------------------------

The minimal setup requires two things:

1. An objective function (any callable ``f(params: dict) -> float``)
2. A dictionary of initial parameter values

Example:

.. code-block:: python

    from chemfit import Fitter

    def objective(params):
        return 2.0 * (params["x"] - 2)**2 + 3.0 * (params["y"] + 1)**2

    initial_params = {"x": 0.0, "y": 0.0}

    fitter = Fitter(objective_function=objective, initial_params=initial_params)
    opt_params = fitter.fit_scipy()

    print(opt_params)  # Expected: {"x": 2.0, "y": -1.0}

The same objective can also be optimized globally using Nevergrad:

.. code-block:: python

    opt_params = fitter.fit_nevergrad(budget=100)

Parameters dictionary format
----------------------------------

The parameter dictionary may be nested to any depth as long as all *leaf values*
are floating-point numbers.

.. code-block:: python

    # Allowed
    params = {
        "foo": {
            "bar": {"a": 1.0},
            "b": 2.0,
        }
    }

    # Not allowed
    params = {
        "foo": 2.0,
        "bar": [1.0, 2.0],  # <-- lists are not allowed
    }

The Fitter automatically flattens and unflattens nested dictionaries internally using
``pydictnest`` during optimization.

Bounds
----------------------------------

Bounds can be specified for each parameter as a dictionary that mirrors the structure
of the ``params`` dictionary. Each leaf node is a tuple ``(lower, upper)`` of floats.

Example:

.. code-block:: python

    bounds = {
        "foo": {
            "bar": {"a": (0.0, 2.0)},
            "b": (-1.0, 3.0),
        }
    }

The bounds are passed to the constructor:

.. code-block:: python

    fitter = Fitter(
        objective_function=objective,
        initial_params=initial_params,
        bounds=bounds,
    )

Notes:

- You can omit bounds for any parameter.
- If bounds are given, both lower and upper must be specified.

Bad and near-bound regions
----------------------------------

To make optimization more robust, the Fitter monitors numerical issues:

- **Invalid or non-float returns** from the objective are replaced with
  ``value_bad_params`` (default: ``1e5``) and logged.
- If the initial loss equals or exceeds this threshold, a warning is issued.
- After fitting, if ``near_bound_tol`` is provided, parameters that are
  close to their bounds trigger a log message listing the affected parameters.

These checks help detect misconfigured or unstable objective functions early.

Callback system
----------------------------------

The Fitter allows you to register one or more callbacks that are executed every
``n_steps`` of the optimization process.

Each callback receives a :py:class:`~chemfit.fitter.CallbackInfo` dataclass with the fields:

- ``opt_params``: Best parameters found so far
- ``opt_loss``: Best loss value found so far
- ``cur_params``: Parameters of the most recent step
- ``cur_loss``: Loss value of the most recent step
- ``step``: Step counter (not necessarily equal to number of function evaluations)
- ``info``: Reference to the current :py:class:`~chemfit.fitter.FitInfo`

Example:

.. code-block:: python

    def print_progress(info):
        print(f"Step {info.step}: loss = {info.cur_loss:.3f}")

    fitter.register_callback(print_progress, n_steps=5)
    opt_params = fitter.fit_scipy()

You can register multiple callbacks; they are executed in order of registration.

FitInfo structure
----------------------------------

During a fit, a :py:class:`~chemfit.fitter.FitInfo` instance tracks global run statistics:

- ``initial_value``: Objective value at initial parameters
- ``final_value``: Objective value after optimization
- ``time_taken``: Total wall-clock time in seconds
- ``n_evals``: Number of objective function evaluations

This object is reset before each new fit and accessible as ``fitter.info``.

Example:

.. code-block:: python

    opt_params = fitter.fit_scipy()
    print(fitter.info.time_taken)
    print(fitter.info.n_evals)

Backend differences
----------------------------------

**SciPy (fit_scipy)**

- Uses local gradient-based optimizers such as ``L-BFGS-B``.
- Supports bounds directly.
- Suitable for smooth, differentiable objectives.

**Nevergrad (fit_nevergrad)**

- Uses global and derivative-free optimizers.
- Accepts a ``budget`` (number of evaluations) and optimizer name (e.g., ``"NgIohTuned"``).
- Useful for noisy, non-smooth, or black-box objectives.

.. code-block:: python

    opt_params = fitter.fit_nevergrad(budget=200, optimizer_str="CMA")

Lifecycle hooks
----------------------------------

Internally, each fit runs through two hooks:

- ``hook_pre_fit()`` — resets state, evaluates initial loss, starts timer.
- ``hook_post_fit()`` — finalizes logging, checks for near-bound parameters.

These can be overridden in subclasses if you want to extend or integrate the Fitter
with custom monitoring systems.

Summary
----------------------------------

- Works with any callable objective: ``f(params: dict) -> float``
- Supports nested parameter dictionaries
- Unified interface for SciPy and Nevergrad backends
- Logging and safety checks for invalid objective returns
- Callback system for progress monitoring
- Tracks timing and evaluation counts through ``FitInfo``

ChemFit's Fitter provides a simple, consistent, and reliable way to drive parameter
optimization in scientific workflows.
