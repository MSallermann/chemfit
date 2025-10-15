#######################
Overview
#######################

ChemFit is a modular framework for parameter optimization of scientific models
built around the Atomic Simulation Environment (ASE) and compatible tools.

It provides a clean way to construct objective functions from smaller building blocks:
computations that produce physical quantities, and loss functions that measure their
deviation from reference data.

Although originally developed for tuning the SCME 2.0 potential, ChemFit's architecture
is model- and calculator-agnostic. Its abstractions can represent any parameterized
computation, such as classical potentials, quantum chemistry wrappers, or surrogate models.


.. _overview_objective_functions:

**********************
Objective Functions
**********************

An objective function in ChemFit is any callable that maps a dictionary of parameters
to a scalar value:

.. code-block:: python

   f(params: dict) -> float

Simple objectives can be written as plain functions:

.. code-block:: python

   def ob(params: dict) -> float:
       return 2.0 * params["x"]**2

For more advanced cases, ChemFit favors the functor style, using objects that implement
``__call__`` and optionally ``get_meta_data()``:

.. code-block:: python

   class Quadratic:
       def __init__(self, factor: float):
           self.factor = factor

       def __call__(self, params: dict) -> float:
           return self.factor * params["x"]**2

   ob = Quadratic(factor=2.0)

These functors can store state, expose metadata, and integrate easily into larger
optimization workflows.

The recommended base class for such functors is
:py:class:`~chemfit.abstract_objective_function.ObjectiveFunctor`.


******************************
Minimization via the Fitter
******************************

The :py:class:`~chemfit.fitter.Fitter` class provides a unified interface to drive
optimization with different backends. Any callable objective can be passed to a
:py:class:`~chemfit.fitter.Fitter` along with an initial parameter set.

.. code-block:: python

   from chemfit import Fitter

   def ob(params):
       return 2.0 * (params["x"] - 2)**2 + 3.0 * (params["y"] + 1)**2

   fitter = Fitter(objective_function=ob, initial_params={"x": 0.0, "y": 0.0})
   optimal_params = fitter.fit_scipy()

   print(optimal_params)  # Expected: x ~ 2.0, y ~ -1.0

Available backends:

1. SciPy via :py:meth:`~chemfit.fitter.Fitter.fit_scipy`
2. Nevergrad via :py:meth:`~chemfit.fitter.Fitter.fit_nevergrad`

Both operate on the same abstract objective interface.


***********************************
Quantity Computers
***********************************

A QuantityComputer represents the computational part of an objective function.
It is a callable object that, given a parameter dictionary, produces a dictionary
of measurable quantities:

.. code-block:: python

   quants = computer(params: dict) -> dict[str, Any]

Conceptually, the data flow looks like this:

::

   parameters  ->  QuantityComputer  ->  quantities  ->  loss  ->  objective  ->  Fitter

By decoupling quantity computation from scalar loss evaluation, ChemFit allows you to:

- Reuse the same physical computation with different loss functions.
- Log and inspect intermediate quantities such as energies, forces, or distances.
- Compose multiple objectives that share the same underlying model.

To obtain a scalar objective, wrap a :py:class:`~chemfit.abstract_objective_function.QuantityComputer` with a loss function using
:py:class:`~chemfit.abstract_objective_function.QuantityComputerObjectiveFunction`:

.. code-block:: python

   from chemfit.abstract_objective_function import QuantityComputerObjectiveFunction

   objective = QuantityComputerObjectiveFunction(
       loss_function=lambda q: (q["energy"] - (-10.0))**2,
       quantity_computer=my_computer,
   )

   result = objective({"epsilon": 1.0, "sigma": 1.0})


***********************************
ASE-Based Quantity Computers
***********************************

ChemFit includes two concrete implementations of the :py:class:`~chemfit.abstract_objective_function.QuantityComputer` interface
that use the Atomic Simulation Environment (ASE) as a backend:

1. :class:`~chemfit.ase_objective_function.SinglePointASEComputer`
   Performs a single-point energy and force calculation.

2. :class:`~chemfit.ase_objective_function.MinimizationASEComputer`
   Relaxes a structure to a local minimum before evaluating quantities.

Both classes are configured through small protocol-based components:

- :py:class:`~chemfit.ase_objective_function.CalculatorFactory`: attaches an ASE calculator to an Atoms object.
- :py:class:`~chemfit.ase_objective_function.ParameterApplier`: updates calculator parameters from a dictionary.
- :py:class:`~chemfit.ase_objective_function.AtomsFactory`: creates or loads an ASE Atoms object.
- :py:class:`~chemfit.ase_objective_function.QuantityProcessor`: extracts or post-processes results after calculation.

This modular setup makes ChemFit compatible with any ASE calculator:
Lennard-Jones, DFTB, machine-learned potentials, or ab initio wrappers.

A minimal sketch:

.. code-block:: python

   from chemfit.abstract_objective_function import QuantityComputerObjectiveFunction
   from chemfit.ase_objective_function import SinglePointASEComputer, PathAtomsFactory
   from chemfit import Fitter

   def construct_calc(atoms): ...
   def apply_params(atoms, params): ...

   computer = SinglePointASEComputer(
       calc_factory=construct_calc,
       param_applier=apply_params,
       atoms_factory=PathAtomsFactory("reference.traj"),
       tag="example",
   )

   objective = QuantityComputerObjectiveFunction(
       loss_function=lambda q: (q["energy"] - (-10.0))**2,
       quantity_computer=computer,
   )

   fitter = Fitter(objective, initial_params={"epsilon": 1.0, "sigma": 1.0})
   fitter.fit_scipy()


*************************************
Composition and Extensibility
*************************************

ChemFit emphasizes composition over subclassing.

You can extend or modify behavior by supplying new factories or quantity processors
instead of inheriting from base classes. For example, you can attach a processor
that computes a bond length or RMSD without changing the core code:

.. code-block:: python

   def bond_length_processor(calc, atoms):
       quants = dict(calc.results)
       quants["bond_length"] = atoms.get_distance(0, 1)
       return quants

   computer = SinglePointASEComputer(
       calc_factory=construct_calc,
       param_applier=apply_params,
       atoms_factory=PathAtomsFactory("ref.traj"),
       quantity_processors=[bond_length_processor],
   )

   result = computer({"epsilon": 1.0, "sigma": 1.0})
   print(result["energy"], result["bond_length"])


******************************
Summary
******************************

- Objective functions map parameters to scalar losses.
- Quantity computers compute physical quantities from parameters.
- The QuantityComputer abstraction is general and backend-independent.
- ChemFit implements two ASE-based computers for single-point and relaxed calculations.
- Factories and processors define calculator behavior and data extraction.
- Composition replaces subclassing: functionality is extended by configuration.
- The Fitter class provides a unified interface for SciPy and Nevergrad optimization.
- Works with any ASE-compatible calculator or custom backend.
