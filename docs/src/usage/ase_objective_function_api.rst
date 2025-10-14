.. _ase_objective_function_api:

=====================================
ASE-Based Quantity Computers
=====================================

The :mod:`chemfit.ase_objective_function` module integrates the
**Atomic Simulation Environment (ASE)** into the generic fitting
framework defined in :mod:`chemfit.abstract_objective_function`.

These classes allow you to use ASE calculators directly as
:class:`QuantityComputer` instances, producing structured
dictionaries of results (energy, forces, stress, etc.) that
can be fed into loss functions and optimizers.

The design is modular — all steps of the computation pipeline
(calculator creation, parameter updates, atom loading, and result
processing) are fully configurable.


Overview
=========

An ASE-based computer in this framework typically performs these steps:

1. Create or load an :class:`ase.Atoms` object.
2. Optionally modify it (constraints, reorientation, scaling, etc.).
3. Attach an ASE :class:`Calculator` to the atoms.
4. Apply a dictionary of parameters to the calculator.
5. Run the calculation with ``atoms.calc.calculate(atoms)``.
6. Collect quantities (e.g., energy, forces) into a result dictionary.

This design makes it easy to couple ASE with gradient-free
optimizers, or to fit interatomic potentials, empirical
force fields, and machine-learned energy models.


Protocols
=========

Several lightweight **protocols** define the expected behavior of
components you can plug into a computer.

Each protocol is just a callable interface that defines one clear responsibility:

- **CalculatorFactory**
  Creates or attaches an ASE calculator to an :class:`ase.Atoms` object.
  It must set ``atoms.calc`` in place.

- **ParameterApplier**
  Applies a dictionary of fitting parameters (``dict[str, Any]``)
  to the calculator currently attached to ``atoms.calc``.

- **AtomsFactory**
  Creates a new :class:`ase.Atoms` object, e.g. by reading a structure file
  or generating atoms programmatically.

- **AtomsPostProcessor**
  Optionally modifies an :class:`ase.Atoms` object after it is created
  and before the calculator is attached — for example, to set constraints
  or adjust periodic boundary conditions.

- **QuantityProcessor**
  Extracts data from the finished ASE calculation and returns
  a dictionary of computed quantities.

Each protocol is checked at runtime via ``check_protocol()`` to ensure
objects passed into the computers conform to the expected interface.


Atoms Factory Example
---------------------

A concrete helper, :class:`PathAtomsFactory`, is provided to read atoms from a file:

.. code-block:: python

   from chemfit.ase_objective_function import PathAtomsFactory

   atoms_factory = PathAtomsFactory("geometry.traj", index=0)
   atoms = atoms_factory()   # returns an ase.Atoms object


SinglePointASEComputer
======================

A **single-point** computer is the simplest kind of ASE-based
quantity computer. It builds an Atoms object, attaches a calculator,
applies parameters, and performs one calculation without geometry optimization.

**Key arguments:**

- ``calc_factory`` - a function attaching a calculator to ``atoms``.
- ``param_applier`` - a function that applies a parameter dictionary.
- ``atoms_factory`` - a factory producing an ``ase.Atoms`` object.
- ``atoms_post_processor`` - optional modifier applied before calculation.
- ``quantity_processors`` - list of callables that extract results.
- ``tag`` - optional label for metadata.

The result of ``__call__(parameters)`` is a dictionary of quantities,
typically including at least ``"energy"`` and possibly ``"forces"`` or
``"stress"``.

Internally, the base class calls all registered quantity processors
to build the final result dictionary. The default processor simply
returns all entries from ``calc.results`` plus the number of atoms.

**Metadata**

``get_meta_data()`` returns a dictionary with:

- ``tag`` - user-defined label.
- ``n_atoms`` - number of atoms in the system.
- ``type`` - the class name of the computer.
- ``last`` - most recent computed quantities.

Example: Lennard-Jones Objective Term
-------------------------------------

The Lennard-Jones (LJ) unit test demonstrates how to build a full
objective from ASE-based computers.

.. code-block:: python

   import functools
   from chemfit.abstract_objective_function import QuantityComputerObjectiveFunction
   from chemfit.ase_objective_function import SinglePointASEComputer
   from chemfit.combined_objective_function import CombinedObjectiveFunction
   from chemfit.fitter import Fitter

   # Custom user-defined ASE adapters
   from conftest import LJAtomsFactory, apply_params_lj, construct_lj, e_lj

   def loss_function(quants: dict, e_ref: float):
       return (quants["energy"] - e_ref) ** 2

   def lj_ob_term(r: float, eps: float, sigma: float):
       computer = SinglePointASEComputer(
           calc_factory=construct_lj,
           param_applier=apply_params_lj,
           atoms_factory=LJAtomsFactory(r),
           tag=f"lj_{r}",
       )

       return QuantityComputerObjectiveFunction(
           loss_function=functools.partial(loss_function, e_ref=e_lj(r, eps, sigma)),
           quantity_computer=computer,
       )

   # Combine many LJ distances into one global objective
   r_list = [2.5, 3.0, 3.5]
   objective = CombinedObjectiveFunction(
       objective_functions=[lj_ob_term(r, 1.0, 1.0) for r in r_list]
   )

   fitter = Fitter(objective, initial_params={"epsilon": 2.0, "sigma": 1.5})
   optimized_params = fitter.fit_scipy()

   print(optimized_params)
   # {'epsilon': ~1.0, 'sigma': ~1.0}

This pattern generalizes to any ASE-compatible calculator.


MinimizationASEComputer
=======================

A subclass of ``SinglePointASEComputer`` that performs a **geometry relaxation**
to the nearest local minimum before running the final single-point calculation.

It uses ASE's :class:`ase.optimize.BFGS` optimizer internally.

**Initialization parameters:**

- ``dt`` - timestep for the optimizer (default: 1e-2).
- ``fmax`` - convergence threshold on maximum force (default: 1e-5).
- ``max_steps`` - maximum number of relaxation steps (default: 2000).

All other arguments are the same as for ``SinglePointASEComputer``.

**Workflow**

1. The structure is reset to its reference positions.
2. Velocities are zeroed.
3. Calculator parameters are applied.
4. A BFGS optimization is run until convergence or max steps reached.
5. The relaxed structure is used for a single-point evaluation.

This class is useful for fitting potentials to equilibrium geometries,
or for objectives that depend on relaxed energies rather than fixed configurations.


Quantity Processors
===================

After the ASE calculation, one or more **quantity processors** are called.
Each processor receives the calculator and atoms, and returns a dictionary
of key-value pairs, which are merged into the final result.

The default processor is:

.. code-block:: python

   def default_quantity_processor(calc, atoms):
       return {**calc.results, "n_atoms": len(atoms)}

You can define additional processors to add, e.g., stress tensors,
force norms, or derived physical quantities.

The Default Processor
---------------------

Every ASE-based computer automatically prepends the built-in
``default_quantity_processor`` to its list of quantity processors.

This ensures that the calculator's raw results (e.g. ``energy``, ``forces``,
and other keys in ``calc.results``) are always included in the output
dictionary, even if you supply your own custom processors.

Your processors are executed *after* the default one, allowing you to
extend or post-process those quantities without needing to repeat the
basic extraction logic.

.. code-block:: python

   def my_processor(calc, atoms):
       # calc.results already present thanks to the default processor
       quants = {"force_norm": (calc.results["forces"] ** 2).sum() ** 0.5}
       return quants

   computer = SinglePointASEComputer(
       calc_factory=construct_calc,
       param_applier=apply_params,
       atoms_factory=MyAtomsFactory(),
       quantity_processors=[my_processor],  # default comes first automatically
   )

   result = computer({"epsilon": 1.0, "sigma": 1.0})
   # result contains energy, forces, and force_norm


Extending and Customizing
=========================

The ASE computers are designed to be **composed**, not subclassed.

Whenever possible, prefer *composition* — supplying your own
factories, processors, and parameter appliers — rather than
inheriting from the base classes. This keeps behavior explicit,
reduces hidden state, and makes components easy to test and reuse
across projects.

**Recommended approach: compose behavior via constructor arguments.**

For example, to add an extra computed property without subclassing:

.. code-block:: python

   import numpy as np
   from chemfit.ase_objective_function import SinglePointASEComputer

   def rms_force_processor(calc, atoms):
       f = calc.results.get("forces")
       if f is None:
           return {}
       return {"rms_force": np.sqrt((f**2).mean())}

   computer = SinglePointASEComputer(
       calc_factory=construct_lj,
       param_applier=apply_params_lj,
       atoms_factory=LJAtomsFactory(2.5),
       quantity_processors=[rms_force_processor],
   )

   results = computer({"epsilon": 1.0, "sigma": 1.0})
   print(results["energy"], results["rms_force"])

**When to subclass**

Subclass only when you need to **extend lifecycle behavior** that cannot
be expressed through composition — for example, adding an additional
relaxation step (as in :class:`MinimizationASEComputer`) or modifying
metadata structure.

Typical extension points:

- ``_compute()`` — to customize how results are produced.
- ``create_atoms_object()`` — to alter how Atoms are built or validated.
- ``get_meta_data()`` — to expose custom metadata or diagnostic info.

**Rule of thumb:**
Start with composition. Reach for subclassing only if you truly need to
change the flow of computation itself.


Case Studies: Custom Quantities via Processors
==============================================

These examples highlight how to express flexible objectives by *composing*
a ``QuantityComputer`` with lightweight **quantity processors**—no subclassing required.

Assumptions (pseudo-helpers)
----------------------------

For illustration, assume the following small adapters exist:

- ``construct_calc(atoms)`` — attaches an ASE calculator to ``atoms.calc``.
- ``apply_params(atoms, params)`` — updates parameters on ``atoms.calc``.
- ``MyAtomsFactory(arg)`` — creates an ``ase.Atoms`` object for the given argument.

(You can think of these as the Lennard–Jones helpers used in the unit tests.)

Dimer Distance Target (with Relaxation)
---------------------------------------

A simple case is to relax a geometry and match an inter-fragment distance
to a reference. The processor augments ``calc.results`` with a custom metric
(``dimer_distance``), and the loss depends only on that quantity.

.. code-block:: python

   from chemfit.abstract_objective_function import QuantityComputerObjectiveFunction
   from chemfit.ase_objective_function import MinimizationASEComputer, PathAtomsFactory
   from chemfit.fitter import Fitter

   REF_DISTANCE = 3.2

   def compute_dimer_distance(calc, atoms):
       return {"dimer_distance" : atoms.get_distance(0, 3)}

   objective = QuantityComputerObjectiveFunction(
       loss_function=lambda q: (q["dimer_distance"] - REF_DISTANCE) ** 2,
       quantity_computer=MinimizationASEComputer(
           calc_factory=construct_calc,
           param_applier=apply_params,
           atoms_factory=PathAtomsFactory("ref.traj"),
           quantity_processors=[compute_dimer_distance],
           tag="dimer_distance",
       ),
   )

   fitter = Fitter(objective_function=objective, initial_params={"epsilon": 1.5, "sigma": 1.2})
   optimal_params = fitter.fit_scipy(tol=1e-4, options={"maxiter": 50})

This pattern demonstrates how specialized geometric quantities can be integrated
without modifying the computer class itself. The ``MinimizationASEComputer``
handles relaxation automatically before the measurement.

Kabsch RMSD Objective
---------------------

Another example aligns a relaxed structure to a reference configuration using
the Kabsch algorithm and minimizes the resulting RMSD. A custom processor caches
the reference positions and returns the rotation, translation, and RMSD as new
quantities.

.. code-block:: python

   from chemfit.abstract_objective_function import QuantityComputerObjectiveFunction
   from chemfit.ase_objective_function import MinimizationASEComputer, PathAtomsFactory, AtomsFactory
   from chemfit.fitter import Fitter

   import chemfit.kabsch as kb

   class KabschDistance:
       def __init__(self, atoms_factory: AtomsFactory):
           self.atoms_factory = atoms_factory
           self._positions_ref = None

       def __call__(self, calc, atoms):
           if self._positions_ref is None:
               self._positions_ref = self.atoms_factory().positions

           R, t = kb.kabsch(atoms.positions, self._positions_ref)
           pos_aligned = kb.apply_transform(atoms.positions, R, t)
           rmsd = kb.rmsd(pos_aligned, self._positions_ref)

           return {"kabsch_r": R, "kabsch_t": t, "kabsch_rmsd": rmsd}

   objective = QuantityComputerObjectiveFunction(
       loss_function=lambda q: q["kabsch_rmsd"],
       quantity_computer=MinimizationASEComputer(
           calc_factory=construct_calc,
           param_applier=apply_params,
           atoms_factory=PathAtomsFactory("ref.traj"),
           quantity_processors=[KabschDistance(PathAtomsFactory("ref.traj"))],
           tag="kabsch",
       ),
   )

   fitter = Fitter(objective_function=objective, initial_params={"epsilon": 1.5, "sigma": 1.2})
   optimal_params = fitter.fit_scipy(tol=1e-4, options={"maxiter": 50})


Design Notes
============

- **Composable:** all behavior is supplied via small protocol objects.
- **Transparent:** metadata always includes the most recent quantities.
- **Reproducible:** atoms are lazily created and cached per instance.
- **ASE-native:** works directly with ASE calculators and optimizers.
- **Debug-friendly:** loggers and metadata help inspect intermediate steps.

These abstractions allow the fitting layer (e.g. :class:`chemfit.fitter.Fitter`)
to remain independent of the simulation backend while still exposing
all relevant physical data through the quantity dictionaries.
