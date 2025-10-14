Combined Objective Functions
=============================

The :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`
class represents a **weighted sum of multiple objective functions**.

It allows several independent objectives (each mapping ``params: dict -> float``)
to be combined into a single callable that can be minimized by a
:py:class:`~chemfit.fitter.Fitter`.

This is useful when fitting against multiple datasets, configurations, or loss
metrics simultaneously.

Basic concept
----------------------------------

Each objective function is paired with a non-negative weight.
When the combined objective is called, all sub-objectives are evaluated using
the same parameter dictionary, multiplied by their weights, and summed.

::

    combined_loss(params) = sum_i ( weight_i * objective_i(params) )


Example
----------------------------------

.. code-block:: python

    from chemfit.combined_objective_function import CombinedObjectiveFunction

    # Define two simple objectives
    def ob1(params):
        return (params["x"] - 1.0)**2

    def ob2(params):
        return (params["x"] - 3.0)**2

    combined = CombinedObjectiveFunction(
        objective_functions=[ob1, ob2],
        weights=[0.5, 1.0],
    )

    loss = combined({"x": 2.0})
    print(loss)  # 0.5*(1.0)^2 + 1.0*(1.0)^2 = 1.5


Constructor
----------------------------------

.. code-block:: python

    CombinedObjectiveFunction(objective_functions, weights=None)

Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``objective_functions``
  Sequence of callables ``f(params: dict[str, float]) -> float``.
  Each function represents one objective term.

- ``weights`` (optional)
  Sequence of non-negative floats. If omitted, all weights default to ``1.0``.

Raises
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- AssertionError if ``len(weights)`` does not match the number of objectives.
- AssertionError if any weight is negative.

Attributes
----------------------------------

- ``objective_functions``
  List of all stored objective callables.

- ``weights``
  List of non-negative weights, aligned with the objective list.


Calling the combined objective
----------------------------------

The combined objective is itself callable:

.. code-block:: python

    result = combined(params)

By default, all internal terms are included.
A subset can be selected via the optional ``idx_slice`` argument:

.. code-block:: python

    # Evaluate only the first term
    loss = combined(params, idx_slice=slice(0, 1))


Methods
----------------------------------

**n_terms()**

Returns the number of objective terms.

.. code-block:: python

    n = combined.n_terms()  # integer

----------------------------------

**add(obj_funcs, weights=1.0)**

Add one or more new objective functions (and corresponding weights) to the instance.

.. code-block:: python

    def ob3(params): return (params["x"] - 4.0)**2
    combined.add(ob3, weights=0.2)

    # Or add multiple at once
    combined.add([ob1, ob2], weights=[0.5, 0.5])

Notes:

- If ``weights`` is a single float, it is reused for each new objective.
- All new weights must be non-negative.
- The function returns ``self`` for method chaining.

----------------------------------

**add_flat(combined_objective_functions_list, weights=None)**

Class method that merges several ``CombinedObjectiveFunction`` instances into
a new, flat one.
Each sub-instance’s internal weights are scaled by its associated outer weight.

.. code-block:: python

    combined1 = CombinedObjectiveFunction([ob1, ob2], [1.0, 2.0])
    combined2 = CombinedObjectiveFunction([ob3], [0.5])

    combined_flat = CombinedObjectiveFunction.add_flat(
        [combined1, combined2],
        weights=[1.0, 0.2],
    )

    print(combined_flat.n_terms())  # 3

This is especially useful when building composite objectives programmatically
(such as combining multiple molecule or configuration terms).

----------------------------------

**get_meta_data()**

Returns basic metadata about the combined objective:

.. code-block:: python

    meta = combined.get_meta_data()
    # Example: {"n_terms": 3, "type": "CombinedObjectiveFunction"}

----------------------------------

**gather_meta_data(idx_slice=slice(None))**

Collects metadata from all sub-objectives that support
``get_meta_data()`` (for example, instances of
:py:class:`~chemfit.abstract_objective_function.ObjectiveFunctor`).

Returns a list of dictionaries or ``None`` for objectives without metadata.

.. code-block:: python

    meta_list = combined.gather_meta_data()
    for entry in meta_list:
        print(entry)


Practical use with Fitter
----------------------------------

The combined objective can be minimized directly using a Fitter:

.. code-block:: python

    from chemfit import Fitter

    fitter = Fitter(objective_function=combined, initial_params={"x": 0.0})
    opt_params = fitter.fit_scipy()

    print(opt_params)

This pattern allows you to easily combine multiple
:py:class:`~chemfit.abstract_objective_function.QuantityComputerObjectiveFunction`
terms, each corresponding to a different dataset, configuration, or physical property.

Example (sketch):

.. code-block:: python

    combined = CombinedObjectiveFunction(
        [
            energy_objective,     # e.g., fit energies
            force_objective,      # e.g., fit forces
            dipole_objective,     # e.g., fit dipole moments
        ],
        weights=[1.0, 0.2, 0.5],
    )

    fitter = Fitter(combined, initial_params=params)
    fitter.fit_nevergrad(budget=200)


Summary
----------------------------------

- Combines multiple objective functions into a single weighted sum.
- Supports arbitrary callables that accept ``params: dict[str, float]``.
- Weights can be adjusted, extended, or merged at runtime.
- Compatible with :py:class:`~chemfit.fitter.Fitter`.
- Useful for multi-objective fitting (energies, forces, properties, etc.).
- Provides metadata aggregation for downstream analysis.
