.. _ase_objective_function_api:

ASE-Based Quantity Computers
============================

The :mod:`chemfit.ase_objective_function` module provides one configurable
:py:class:`~chemfit.ase_objective_function.ASEComputer`. It separates an ASE
evaluation into three stages:

1. prepare a copy of the cached base structure and attach a fresh calculator;
2. run one evaluation procedure;
3. extract quantities from the resulting atoms and calculator.

Single-point calculations, geometry optimization, and custom ASE procedures all
use this same lifecycle.

Minimal example
---------------

.. code-block:: python

   from ase.calculators.lj import LennardJones

   from chemfit.ase_objective_function import ASEComputer, PathAtomsFactory

   def make_calculator(parameters, atoms, ctx):
       return LennardJones(
           epsilon=parameters["epsilon"],
           sigma=parameters["sigma"],
       )

   computer = ASEComputer(
       atoms_factory=PathAtomsFactory("geometry.xyz"),
       calculator=make_calculator,
   )

   quantities = computer({"epsilon": 1.0, "sigma": 1.0})
   print(quantities["energy"])

The default evaluator calls ``atoms.calc.calculate(atoms)``, so this is a
single-point calculation. The default quantity processor returns the entries
in ``calc.results`` and an additional ``"n_atoms"`` value.

Components
----------

An ASE computer contains:

- an atoms factory;
- zero or more base-atoms setup callbacks;
- one calculator factory;
- one evaluator;
- one or more quantity processors.

The context-aware callbacks use these contracts:

.. code-block:: python

   calculator(parameters, atoms, ctx) -> Calculator
   evaluator(parameters, atoms, ctx) -> None
   processor(calc, atoms, ctx) -> dict[str, object]

An atoms setup callback is structural and runs only when the cached base
geometry is initialized:

.. code-block:: python

   atoms_setup(atoms) -> None

Calculator factories
--------------------

The calculator factory receives the current parameters, the per-evaluation
atoms copy, and the current
:py:class:`~chemfit.abstract_objective_function.EvaluateContext`. It returns a
fresh configured calculator:

.. code-block:: python

   def make_calculator(parameters, atoms, ctx):
       ctx.temp.calculator_name = "LennardJones"
       return LennardJones(
           epsilon=parameters["epsilon"],
           sigma=parameters["sigma"],
       )

You can supply it to the constructor or configure it fluently:

.. code-block:: python

   computer = ASEComputer(
       atoms_factory=PathAtomsFactory("geometry.xyz"),
   ).with_calculator(make_calculator)

Additional keyword-only arguments can be bound by
:py:meth:`~chemfit.ase_objective_function.ASEComputer.with_calculator`:

.. code-block:: python

   def make_calculator(parameters, atoms, ctx, *, cutoff):
       return LennardJones(
           epsilon=parameters["epsilon"],
           sigma=parameters["sigma"],
           rc=cutoff,
       )

   computer = computer.with_calculator(make_calculator, cutoff=12.0)

Evaluators
----------

An evaluator performs the ASE calculation that must finish before quantity
extraction. Every computer has exactly one active evaluator.

The default evaluator performs a single-point calculation. Advanced workflows
can replace it with
:py:meth:`~chemfit.ase_objective_function.ASEComputer.with_evaluator`:

.. code-block:: python

   def run_md(parameters, atoms, ctx, *, timestep, steps):
       dynamics = make_dynamics(atoms, timestep=timestep)
       dynamics.run(steps)
       ctx.temp.md_steps = steps

   md_computer = computer.with_evaluator(
       run_md,
       timestep=1.0,
       steps=1000,
   )

The evaluator receives the same context used by calculator construction and
quantity extraction. These callbacks can therefore communicate through
``ctx.temp`` without storing evaluation state on the computer.

Geometry minimization
---------------------

Use :py:meth:`~chemfit.ase_objective_function.ASEComputer.minimize` for the
common ASE BFGS workflow:

.. code-block:: python

   relaxed = computer.minimize(
       fmax=1e-5,
       max_steps=2000,
   )

``minimize()`` returns another ``ASEComputer`` configured with a BFGS evaluator;
minimization is not represented by a separate subclass. Quantity processors run
after optimization and therefore see the relaxed structure and its calculator.

Both ``minimize()`` and ``with_evaluator()`` use copy-on-write configuration.
The original computer retains its previous evaluator.

Quantity processors
-------------------

A quantity processor receives the evaluated calculator, the post-evaluation
atoms, and the current context:

.. code-block:: python

   def extract_distance(calc, atoms, ctx):
       return {
           "energy": calc.results["energy"],
           "distance": atoms.get_distance(0, 1),
       }

Pass processors to the constructor when they define the complete result:

.. code-block:: python

   computer = ASEComputer(
       atoms_factory=PathAtomsFactory("dimer.xyz"),
       calculator=make_calculator,
       quantity_processors=[extract_distance],
   )

If ``quantity_processors`` is omitted,
:py:class:`~chemfit.ase_objective_function.DefaultQuantityProcessor` is used.
When an explicit iterable is supplied, it replaces that default.

Use :py:meth:`~chemfit.ase_objective_function.ASEComputer.with_processor` to
append another processor to a configured computer:

.. code-block:: python

   computer = computer.with_processor(extract_distance)

Processor results are merged in order with ``dict.update()``, so later
processors can replace values produced by earlier processors.

Atoms factories and setup
-------------------------

An atoms factory takes no arguments and returns one :py:class:`ase.Atoms`
object. :py:class:`~chemfit.ase_objective_function.PathAtomsFactory` reads a
single structure using ASE:

.. code-block:: python

   atoms_factory = PathAtomsFactory("trajectory.xyz", index=0)

If the selected ASE index resolves to multiple images, the factory raises
``ValueError``.

Use :py:meth:`~chemfit.ase_objective_function.ASEComputer.with_atoms_setup` for
structure configuration that should happen once before the base atoms are
cached:

.. code-block:: python

   from ase.constraints import FixAtoms

   def freeze_first_atom(atoms):
       atoms.set_constraint(FixAtoms(indices=[0]))

   constrained = computer.with_atoms_setup(freeze_first_atom)

Setup callbacks are applied in registration order. Adding one invalidates the
returned computer's inherited atoms cache because it changes construction of
the base structure. The source computer is unchanged.

By contrast, ``with_calculator()``, ``with_processor()``,
``with_evaluator()``, and ``minimize()`` do not alter the base geometry and
retain an already initialized cache.

Caching and parallel evaluation
-------------------------------

The base structure is initialized lazily. Initialization is protected by a
lock so concurrent threads create it only once. Each evaluation then copies the
cached atoms and creates its own calculator, preventing calculator state from
leaking between evaluations.

The initialization lock is omitted during pickling and recreated when the
computer is unpickled. This lets executor and MPI backends initialize geometry
locally in each process.

Turning a computer into an objective
------------------------------------

Like every quantity computer, an ``ASEComputer`` becomes an objective through
:py:meth:`~chemfit.abstract_objective_function.QuantityComputer.with_loss`:

.. code-block:: python

   def energy_loss(quantities, *, reference):
       return (quantities["energy"] - reference) ** 2

   objective = computer.with_loss(
       energy_loss,
       reference=-0.1,
   )

   value = objective({"epsilon": 1.0, "sigma": 1.0})

Evaluation details remain available through the supplied context:

.. code-block:: python

   from chemfit.abstract_objective_function import EvaluateContext

   ctx = EvaluateContext()
   quantities = computer({"epsilon": 1.0, "sigma": 1.0}, ctx)

   print(ctx.quantities)
   print(ctx.temp.atoms)

Custom evaluators
-----------------

An evaluator is deliberately one callable rather than an ordered workflow
engine. A custom evaluator may run molecular dynamics, invoke an ASE optimizer,
or perform several internal stages. The surrounding preparation and quantity
extraction lifecycle remains owned by ``ASEComputer``.
