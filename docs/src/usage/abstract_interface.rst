=============================
Generic Objective Interfaces
=============================

This page describes the abstract interfaces used to build objective functions
in the fitting package. The design separates **computing intermediate
quantities** from **evaluating a scalar loss**, and provides lightweight
introspection via ``get_meta_data()``.


High-level Concepts
===================

- **Parameters**: a ``dict[str, Any]`` of free variables your optimizer controls.
- **Quantities**: a ``dict[str, Any]`` computed from parameters (e.g., predictions,
  residuals, caches) that downstream losses can reuse.
- **Loss / Objective value**: a single ``float`` to minimize.

.. hint::

   Keep parameters and quantities explicit dictionaries. This makes functions
   easy to test, log, and serialize.


Data Flow
=========

1. Optimizer proposes ``parameters`` (a dict).
2. ``QuantityComputer(parameters)`` → ``quantities`` (a dict).
3. ``loss_function(quantities)`` → ``float`` loss.
4. Optional: call ``get_meta_data()`` on the objective to inspect the most recent
   computation (e.g., last loss, last quantities).


Usage Examples
==============

Minimal quantity computer
-------------------------

.. code-block:: python

   from typing import Any
   from chemfit.abstract_objective_function import QuantityComputer

   class LinearModelComputer(QuantityComputer):
       """
       Example that computes predictions and residuals for y = a * x + b.
       Expects parameters: {"a": float, "b": float}
       Holds fixed data x, y in the instance.
       """
       def __init__(self, x, y):
           super().__init__()
           self.x = x
           self.y = y

       def _compute(self, parameters: dict[str, Any]) -> dict[str, Any]:
           a = parameters["a"]
           b = parameters["b"]
           y_hat = [a * xi + b for xi in self.x]
           residuals = [yh - yt for yh, yt in zip(y_hat, self.y)]
           return {"y_hat": y_hat, "residuals": residuals}

A simple loss function
----------------------

.. code-block:: python

   from typing import Any

   def mse_loss(quantities: dict[str, Any]) -> float:
       r = quantities["residuals"]
       return sum(ri * ri for ri in r) / len(r)

Wiring it together as an objective
----------------------------------

.. code-block:: python

   from chemfit.abstract_objective_function import QuantityComputerObjectiveFunction

   x = [0.0, 1.0, 2.0, 3.0]
   y = [1.0, 3.1, 4.9, 7.2]

   qc = LinearModelComputer(x, y)
   objective = QuantityComputerObjectiveFunction(
       loss_function=mse_loss,
       quantity_computer=qc,
   )

   loss = objective({"a": 2.0, "b": 1.0})
   print(loss)  # -> a float

   # Introspection
   meta = objective.get_meta_data()
   # meta["last_loss"] is the last computed loss
   # meta["computer"]["last"] contains the most recent quantities dict

Loss as an ``ObjectiveFunctor`` (optional)
------------------------------------------

If your loss needs its own state/metadata, implement it as an :py:class:`~chemfit.abstract_objective_function.ObjectiveFunctor`
over quantities:

.. code-block:: python

   from typing import Any
   from chemfit.abstract_objective_function import ObjectiveFunctor, SupportsGetMetaData

   class RobustL1Loss(ObjectiveFunctor):
       def __init__(self):
           self._last: float | None = None

       def __call__(self, quantities: dict[str, Any]) -> float:
           r = quantities["residuals"]
           self._last = sum(abs(ri) for ri in r) / len(r)
           return self._last

       def get_meta_data(self) -> dict[str, Any]:
           return {"last_loss": self._last}

   robust = RobustL1Loss()
   objective = QuantityComputerObjectiveFunction(robust, qc)
   _ = objective({"a": 2.0, "b": 1.0})
   # objective.get_meta_data()["loss_function"] now includes RobustL1Loss metadata.


Design Notes & Best Practices
=============================

- **Keep losses pure** when possible: accept only ``quantities`` and return
  a ``float``. This simplifies testing and reuse.
- **Use ``QuantityComputer`` to cache** expensive intermediate results. The base
  class already stores the last computed dictionary in metadata.
- **Validate inputs early** (e.g., check required keys in ``parameters`` and
  ``quantities``) to fail fast during development.
- **Log via metadata**: expose anything useful for debugging (timings,
  convergence flags, shapes) through ``get_meta_data()``.
- **Composability**: multiple objectives can wrap the same ``QuantityComputer``
  with different losses, enabling multi-criteria exploration.
