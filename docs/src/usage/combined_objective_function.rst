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

.. math::

    \text{combined_loss}(\text{params}) = \sum_i w_i \cdot ob_i(\text{params})


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
