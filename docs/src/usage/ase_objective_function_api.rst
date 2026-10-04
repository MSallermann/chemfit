.. _ase_objective_function_api:

ASE-Based Quantity Computers
============================

Use :func:`chemfit.ase_quantity() <chemfit.api.ase_quantity>` to create an
ASE-backed quantity computation from an ``ase.Atoms`` object, a structure
path, or a zero-argument atoms factory. It returns the configurable
:py:class:`~chemfit.ase_objective_function.ASEComputer` documented on this
page. An ``ASEComputer`` separates evaluation into three stages:

1. prepare a copy of the cached base structure and attach a fresh calculator;
2. run one evaluation procedure;
3. extract quantities from the resulting atoms and calculator.

Single-point calculations, geometry optimization, and custom ASE procedures all
use this same lifecycle.

Minimal example
---------------

.. code-block:: python

   import chemfit
   from ase.calculators.lj import LennardJones

   def make_calculator(parameters, atoms, ctx):
       return LennardJones(
           epsilon=parameters["epsilon"],
           sigma=parameters["sigma"],
       )

   computer = chemfit.ase_quantity("geometry.xyz").with_calculator(
       make_calculator
   )

   quantities = computer({"epsilon": 1.0, "sigma": 1.0})
   print(quantities["energy"])

The default evaluator calls ``atoms.calc.calculate(atoms)``, so this is a
single-point calculation. The default quantity processor returns the entries
in ``calc.results`` and an additional ``"n_atoms"`` value.

``chemfit.ase_quantity(...)`` has constructed an
:py:class:`~chemfit.ase_objective_function.ASEComputer`. The sections below
document its full calculator, evaluator, processor, setup, and caching API.

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

The evaluator is the action stage between calculator construction and quantity
extraction. For each call, ``ASEComputer``:

1. copies the cached base atoms into ``ctx.temp.atoms``;
2. creates and attaches a fresh calculator using the current parameters;
3. calls ``evaluator(parameters, atoms, ctx)``;
4. passes the resulting atoms and calculator state to the quantity processors.

The evaluator receives the evaluation-local atoms copy, with ``atoms.calc``
already set. It should perform the calculation or simulation in place and
return ``None``. Its return value is not used: calculated properties belong in
``atoms.calc.results``, structural changes belong on ``atoms``, and additional
per-evaluation state can be stored on ``ctx.temp``. Processors then read that
post-evaluation state and turn it into the quantity dictionary.

The default evaluator calls ``atoms.calc.calculate(atoms)``. A custom evaluator
can instead request particular properties, run dynamics, optimize the
structure, or combine several ASE operations. For example:

.. code-block:: python

   def calculate_energy_and_forces(parameters, atoms, ctx):
       atoms.get_potential_energy()
       forces = atoms.get_forces()
       ctx.temp.maximum_force = abs(forces).max()

   evaluated = computer.with_evaluator(calculate_energy_and_forces)

Here the ASE property methods populate ``atoms.calc.results``. A later
processor can read those results, the potentially modified atoms, and
``ctx.temp.maximum_force``.

Use
:py:meth:`~chemfit.ase_objective_function.ASEComputer.with_evaluator`:

.. code-block:: python

   from ase import units
   from ase.md.verlet import VelocityVerlet

   def run_md(parameters, atoms, ctx, *, timestep, steps):
       dynamics = VelocityVerlet(atoms, timestep=timestep * units.fs)
       dynamics.run(steps)
       ctx.temp.md_steps = steps

   md_computer = computer.with_evaluator(
       run_md,
       timestep=1.0,
       steps=1000,
   )

Additional keyword arguments are bound to the evaluator. Every computer has
exactly one active evaluator, so each ``with_evaluator()`` call replaces the
previous evaluator rather than appending another stage. It returns a configured
copy; the source computer is unchanged.

The calculator factory, evaluator, and processors all receive the same
context. They can therefore communicate through ``ctx.temp`` without storing
evaluation state on the computer. If the evaluator raises an exception,
processors are not run and the exception follows the enclosing objective's
usual error handling.

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
       calculator_factory=make_calculator,
       quantity_processors=[extract_distance],
   )

If ``quantity_processors`` is omitted or empty,
:py:class:`~chemfit.ase_objective_function.DefaultQuantityProcessor` is used as
an evaluation-time fallback. As soon as an explicit processor is registered,
only the explicitly configured processors run.

Use :py:meth:`~chemfit.ase_objective_function.ASEComputer.with_processor` to
append a processor to a configured computer:

.. code-block:: python

   computer = computer.with_processor(extract_distance)

On a computer with no explicit processors, this first registration replaces
the implicit default fallback. To retain the default quantities alongside
custom ones, register the default explicitly:

.. code-block:: python

   from chemfit.ase_objective_function import DefaultQuantityProcessor

   computer = (
       computer
       .with_processor(DefaultQuantityProcessor())
       .with_processor(extract_distance)
   )

Processor results are merged in order with ``dict.update()``, so later
processors can replace values produced by earlier processors.

Atoms factories and setup
-------------------------

An atoms factory takes no arguments and returns one :py:class:`ase.Atoms`
object. :py:class:`~chemfit.ase_objective_function.PathAtomsFactory` reads a
single structure using ASE:

.. code-block:: python

   atoms_factory = PathAtomsFactory("trajectory.xyz", index=0)

If the selected ASE index resolves to multiple images, the factory rejects the
selection because one ``ASEComputer`` requires one base structure.

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
