.. _ase_objective_function_api:

ASE-Based Quantity Computers
============================

The :mod:`chemfit.ase_objective_function` module provides quantity computers built
around ASE.

Minimal example
---------------

.. code-block:: python

   from chemfit.ase_objective_function import SinglePointASEComputer, PathAtomsFactory

   def construct_calc(atoms):
       atoms.calc = MyCalculator()

   def apply_params(atoms, params):
       atoms.calc.parameters.update(params)

   computer = SinglePointASEComputer(
       calc_factory=construct_calc,
       param_applier=apply_params,
       atoms_factory=PathAtomsFactory("geometry.xyz"),
   )

   quants = computer({"param": 1.0})  # returns dict with energy, forces, ...
   print(quants["energy"])

The example above follows the standard pattern.
You provide four pieces:

- something that creates an ``ase.Atoms`` object
- something that attaches a calculator to it
- something that applies fitting parameters to that calculator
- one or more quantity processors that turn the finished ASE calculation into a
  quantity dictionary

ChemFit takes care of the rest.

This is useful when you want to fit ASE calculators directly, either on fixed
structures or on relaxed ones.

The main classes are:

- :py:class:`~chemfit.ase_objective_function.SinglePointASEComputer`
- :py:class:`~chemfit.ase_objective_function.MinimizationASEComputer`

Implementing the customization points
-------------------------------------

The arguments passed to :py:class:`~chemfit.ase_objective_function.SinglePointASEComputer`
are just small callables with well-defined responsibilities.

In most cases, you do not need to subclass anything. You only need to provide
functions (or callable objects) with the expected behavior.

Each component does one thing.


Calculator factory
^^^^^^^^^^^^^^^^^^

A calculator factory receives an :py:class:`ase.Atoms` object and attaches a
calculator to it.

.. code-block:: python

   from ase.calculators.lj import LennardJones

   def construct_lj(atoms):
       atoms.calc = LennardJones()

The factory should not run the calculation. It should only attach the calculator.


Parameter applier
^^^^^^^^^^^^^^^^^

A parameter applier receives the atoms object and the current parameter
dictionary. Its job is to transfer fitting parameters into the attached
calculator.

.. code-block:: python

   def apply_params_lj(atoms, params):
       atoms.calc.parameters.epsilon = params["epsilon"]
       atoms.calc.parameters.sigma = params["sigma"]

This function is called for every evaluation. It should not rely on cached state.


Atoms factory
^^^^^^^^^^^^^

An atoms factory takes no arguments and returns a single
:py:class:`ase.Atoms` object.

.. code-block:: python

   from ase import Atoms

   def make_dimer():
       return Atoms("Ar2", positions=[[0, 0, 0], [3.5, 0, 0]])

You can also implement factories as callable objects with internal state
(this applies to all factories not just the atoms factory):

.. code-block:: python

   from ase.io import read

   class MyAtomsFactory:
       def __init__(self, path, index=0):
           self.path = path
           self.index = index

       def __call__(self):
           return read(self.path, index=self.index)

:py:class:`~chemfit.ase_objective_function.PathAtomsFactory` is a built-in
implementation of this pattern.

.. note::

    The simplest atoms factory shipped with ChemFit is
    :py:class:`~chemfit.ase_objective_function.PathAtomsFactory`.

    It reads a single structure from disk using ASE (:py:func:`ase.io.read`).

    .. code-block:: python

        from chemfit.ase_objective_function import PathAtomsFactory

        atoms_factory = PathAtomsFactory("geometry.traj")
        atoms = atoms_factory()

    You can also pass an ASE index:

    .. code-block:: python

        atoms_factory = PathAtomsFactory("trajectory.xyz", index=0)

    The important detail is that the factory must return exactly one
    :py:class:`ase.Atoms` object. If the index selects multiple images,
    :py:class:`~chemfit.ase_objective_function.PathAtomsFactory` raises an exception.

Atoms post-processor
^^^^^^^^^^^^^^^^^^^^

An atoms post-processor receives the atoms object once, before it is cached by
the computer instance.

Use it for structure-level setup that should be shared across evaluations.

.. code-block:: python

   from ase.constraints import FixAtoms

   def freeze_first_atom(atoms):
       atoms.set_constraint(FixAtoms(indices=[0]))

This function runs once during setup, not once per evaluation.


Remarks
^^^^^^^

Each customization point has a single responsibility:

- the atoms factory creates atoms
- the calculator factory attaches a calculator
- the parameter applier updates calculator parameters
- the quantity processor extracts quantities

All of these can be implemented either as plain functions or as callable
objects (classes implementing ``__call__``).

Using callable objects allows you to store configuration or reference data in
``__init__`` while keeping evaluation itself stateless.

Keeping these responsibilities separate makes the resulting computer easier to
test and reuse.


How a SinglePointASEComputer works
----------------------------------

:py:class:`~chemfit.ase_objective_function.SinglePointASEComputer` is the basic
ASE quantity computer.

A single evaluation does the following:

1. create the base atoms object once, using ``atoms_factory``
2. optionally modify it once, using ``atoms_post_processor``
3. copy the cached atoms object into ``ctx.temp.atoms``
4. attach a fresh calculator
5. apply the current parameter dictionary
6. call ``atoms.calc.calculate(atoms)``
7. run all quantity processors and merge their outputs

The important consequence is that the reference structure is cached on the
computer instance, while each evaluation still uses a fresh copy of the atoms
and a fresh calculator.

That means the calculator can mutate internal state without corrupting later
evaluations.


Turning an ASE computer into an objective term
----------------------------------------------

In practice, ASE computers are almost always turned into objective terms
using :py:meth:`~chemfit.abstract_objective_function.QuantityComputer.with_loss`.

.. code-block:: python

   from chemfit.ase_objective_function import SinglePointASEComputer, PathAtomsFactory

   def construct_lj(atoms):
       atoms.calc = LennardJones()

   def apply_params_lj(atoms, params):
       atoms.calc.parameters.epsilon = params["epsilon"]
       atoms.calc.parameters.sigma = params["sigma"]

   def loss_function(quants, e_ref):
       return (quants["energy"] - e_ref) ** 2

   computer = SinglePointASEComputer(
       calc_factory=construct_lj,
       param_applier=apply_params_lj,
       atoms_factory=PathAtomsFactory("dimer.xyz"),
   )

   objective_term = computer.with_loss(
       loss_function,
       e_ref=-0.1,
   )

   loss = objective_term({"epsilon": 1.0, "sigma": 1.0})

.. note::

   :py:meth:`~chemfit.abstract_objective_function.QuantityComputer.with_loss`
   automatically binds additional keyword arguments to the loss function.

   This means you can pass parameters such as ``e_ref`` directly, without
   using :py:mod:`functools.partial`.


A real Lennard-Jones example
----------------------------

The following example fits a Lennard–Jones potential.

For several interatomic distances, one objective term is created per
configuration. Each term evaluates the energy with ASE and compares it to a
reference value.

All terms are combined into a single objective and optimized to recover the
correct parameters.

.. code-block:: python

   import numpy as np

   from chemfit.ase_objective_function import SinglePointASEComputer
   from chemfit.combined_objective_function import CombinedObjectiveFunction
   from chemfit.fitter import Fitter

   from conftest import LJAtomsFactory, apply_params_lj, construct_lj

   def e_lj(r, eps, sigma):
       return 4 * eps * ((sigma / r)**12 - (sigma / r)**6)

   def loss_function(quants, e_ref):
       return (quants["energy"] - e_ref) ** 2

   def lj_ob_term(r, eps, sigma):
       computer = SinglePointASEComputer(
           calc_factory=construct_lj,
           param_applier=apply_params_lj,
           atoms_factory=LJAtomsFactory(r),
           tag=f"lj_{r}",
       )

       return computer.with_loss(
           loss_function,
           e_ref=e_lj(r, eps, sigma),
       )

   def get_objective(eps, sigma):
       r_min = 2.0 ** (1 / 6) * sigma
       r_list = np.linspace(0.925 * r_min, 3.0 * sigma)

       return CombinedObjectiveFunction(
           objective_functions=[lj_ob_term(r, eps, sigma) for r in r_list]
       )

   objective = get_objective(1.0, 1.0)

   fitter = Fitter(
       objective,
       initial_params={"epsilon": 2.0, "sigma": 1.5},
   )

   optimal_params = fitter.fit_scipy()

Quantity processors
-------------------

A quantity processor is a callable with signature

.. code-block:: python

   def processor(calc, atoms) -> dict[str, object]:
       ...

It receives the evaluated calculator and atoms object and returns a dictionary.
The outputs of all quantity processors are merged into the final quantity
dictionary.

The default quantity processor is
:py:class:`~chemfit.ase_objective_function.DefaultQuantityProcessor`.

It returns:

- everything in ``calc.results``
- ``"n_atoms"``

A small example:

.. code-block:: python

   from chemfit.ase_objective_function import (
       DefaultQuantityProcessor,
       SinglePointASEComputer,
   )

   computer = SinglePointASEComputer(
       calc_factory=construct_lj,
       param_applier=apply_params_lj,
       atoms_factory=PathAtomsFactory("dimer.xyz"),
       quantity_processors=[DefaultQuantityProcessor()],
   )

   quants = computer({"epsilon": 1.0, "sigma": 1.0})
   print(quants.keys())

One detail matters here: if you pass ``quantity_processors=None``, ChemFit uses
``[DefaultQuantityProcessor()]``.

If you pass your own list, ChemFit uses exactly that list. The default processor
is **not** prepended automatically.

So if you want both the raw calculator results and your own derived quantities,
include :py:class:`~chemfit.ase_objective_function.DefaultQuantityProcessor`
explicitly.

For example:

.. code-block:: python

   import numpy as np

   from chemfit.ase_objective_function import (
       DefaultQuantityProcessor,
       SinglePointASEComputer,
   )

   def rms_force_processor(calc, atoms):
       forces = calc.results.get("forces")
       if forces is None:
           return {}
       return {"rms_force": np.sqrt((forces**2).mean())}

   computer = SinglePointASEComputer(
       calc_factory=construct_lj,
       param_applier=apply_params_lj,
       atoms_factory=PathAtomsFactory("dimer.xyz"),
       quantity_processors=[
           DefaultQuantityProcessor(),
           rms_force_processor,
       ],
   )

   quants = computer({"epsilon": 1.0, "sigma": 1.0})
   print(quants["energy"])
   print(quants["rms_force"])

.. warning::

    Because quantity processor outputs are merged with ``dict.update()``, later
    processors can overwrite keys returned by earlier ones.

.. note::

   Quantity processors should follow the same basic rule as quantity computers:
   avoid mutating global state or instance state during evaluation.

   In practice, a quantity processor should behave like a pure function of
   ``calc`` and ``atoms``, returning a dictionary of derived quantities.

   This matters especially when evaluations may run in parallel.

Atoms post-processing
---------------------

Sometimes the structure should be modified once before any evaluations happen.

That is what ``atoms_post_processor`` is for.

It is applied to the base atoms object before that object is cached.

.. code-block:: python

   def freeze_first_atom(atoms):
       from ase.constraints import FixAtoms
       atoms.set_constraint(FixAtoms(indices=[0]))

   computer = SinglePointASEComputer(
       calc_factory=construct_lj,
       param_applier=apply_params_lj,
       atoms_factory=PathAtomsFactory("geometry.xyz"),
       atoms_post_processor=freeze_first_atom,
   )

Since the processed atoms object is cached, this is the right place for
structure-level setup that should be shared across all evaluations of the same
computer instance.

MinimizationASEComputer
-----------------------

:py:class:`~chemfit.ase_objective_function.MinimizationASEComputer` is a
subclass of :py:class:`~chemfit.ase_objective_function.SinglePointASEComputer`
that performs a local geometry optimization before extracting quantities.

Internally it uses ASE's :py:class:`ase.optimize.BFGS`.

The workflow is almost the same as for the single-point computer, except that
after ``prepare_ctx(...)`` it runs

.. code-block:: python

   optimizer = BFGS(ctx.temp.atoms, logfile=None)
   optimizer.run(fmax=self.fmax, steps=self.max_steps)

and only then calls the quantity processors.

A minimal example:

.. code-block:: python

   from chemfit.ase_objective_function import MinimizationASEComputer

   computer = MinimizationASEComputer(
       calc_factory=construct_lj,
       param_applier=apply_params_lj,
       atoms_factory=PathAtomsFactory("geometry.xyz"),
       fmax=1e-4,
       max_steps=500,
   )

   quants = computer({"epsilon": 1.0, "sigma": 1.0})

The constructor adds three parameters:

.. code-block:: python

   MinimizationASEComputer(
       ...,
       dt=1e-2,
       fmax=1e-5,
       max_steps=2000,
   )

In the current implementation, ``fmax`` and ``max_steps`` are used by BFGS.
``dt`` is stored for compatibility with older code but is currently unused.

A distance-based example after relaxation
-----------------------------------------

Here is a realistic example from the test suite. The relaxation is handled by
:py:class:`~chemfit.ase_objective_function.MinimizationASEComputer`; the custom
quantity processor adds a derived geometric quantity.

.. code-block:: python

   from chemfit.abstract_objective_function import QuantityComputerObjectiveFunction
   from chemfit.ase_objective_function import MinimizationASEComputer, PathAtomsFactory

   REF_DISTANCE = 3.2

   def compute_dimer_distance(calc, atoms):
       quants = calc.results
       quants["dimer_distance"] = atoms.get_distance(0, 3)
       return quants

   objective = QuantityComputerObjectiveFunction(
       quantity_computer=MinimizationASEComputer(
           calc_factory=construct_calc,
           param_applier=apply_params,
           atoms_factory=PathAtomsFactory("ref.traj"),
           quantity_processors=[compute_dimer_distance],
           tag="dimer_distance",
       ),
       loss_function=lambda quants: (quants["dimer_distance"] - REF_DISTANCE) ** 2,
   )

The important part here is that the quantity processor runs on the relaxed
structure, not on the original one.

A more advanced processor: Kabsch RMSD
--------------------------------------

Quantity processors do not have to be plain functions. They can also be objects.

If a processor needs fixed reference data, the right place to acquire that state
is ``__init__``. It should not be acquired lazily during ``__call__``.

This follows the same rule as in :ref:`writing_quantity_computers`: evaluation
should not mutate hidden instance state.

An example using the Kabsch RMSD against the reference configuration looks like this:

.. code-block:: python

   from typing import Any

   import numpy as np
   from ase import Atoms
   from ase.calculators.calculator import Calculator

   import chemfit.kabsch as kb
   from chemfit.ase_objective_function import AtomsFactory

   class KabschDistance:
       def __init__(self, atoms_factory: AtomsFactory):
           self._positions_ref = np.array(atoms_factory().positions, copy=True)

       def __call__(self, calc: Calculator, atoms: Atoms) -> dict[str, Any]:
           kabsch_r, kabsch_t = kb.kabsch(atoms.positions, self._positions_ref)
           positions_aligned = kb.apply_transform(atoms.positions, kabsch_r, kabsch_t)
           kabsch_rmsd = kb.rmsd(positions_aligned, self._positions_ref)

           return {
               "kabsch_t": kabsch_t,
               "kabsch_r": kabsch_r,
               "kabsch_rmsd": kabsch_rmsd,
           }

Used in a minimization-based objective:

.. code-block:: python

   objective = MinimizationASEComputer(
       calc_factory=construct_calc,
       param_applier=apply_params,
       atoms_factory=PathAtomsFactory("ref.traj"),
       quantity_processors=[
           KabschDistance(PathAtomsFactory("ref.traj"))
       ],
       tag="kabsch",
   ).with_loss(lambda quants: quants["kabsch_rmsd"])

.. note::

   If a quantity processor needs persistent reference data, acquire it in
   ``__init__`` and treat it as fixed thereafter.

   Do not lazily initialize or mutate processor state inside ``__call__``.
   That makes evaluation harder to reason about and can become problematic
   under parallel execution.


What gets stored in the context
-------------------------------

Like any quantity computer, ASE-based computers write their results into the
evaluation context.

That means you can inspect quantities after evaluation:

.. code-block:: python

   from chemfit.abstract_objective_function import EvaluateContext

   ctx = EvaluateContext()
   quants = computer({"epsilon": 1.0, "sigma": 1.0}, ctx)

   print(ctx.quantities)
   print(ctx.to_meta_data())

The ASE computer also uses ``ctx.temp.atoms`` internally during evaluation.
That is an implementation detail, but it is useful to know when debugging.

Composition vs subclassing
--------------------------

For these ASE computers, composition is usually enough.

Most customization should happen through:

- ``calc_factory``
- ``param_applier``
- ``atoms_factory``
- ``atoms_post_processor``
- ``quantity_processors``

That already covers most use cases.

Subclassing only becomes necessary when the evaluation flow itself changes. The
main built-in example is
:py:class:`~chemfit.ase_objective_function.MinimizationASEComputer`, which keeps
the single-point setup but inserts an optimization step before the quantity
processors run.

Remarks
-------

The most important practical points are these.

If you provide your own ``quantity_processors``, include
:py:class:`~chemfit.ase_objective_function.DefaultQuantityProcessor`
yourself if you still want ``calc.results`` in the output.

If you need a fixed structure plus a custom derived quantity,
:py:class:`~chemfit.ase_objective_function.SinglePointASEComputer` is usually the
right tool.

If the quantity should be measured after local relaxation, use
:py:class:`~chemfit.ase_objective_function.MinimizationASEComputer` instead.

Once wrapped in a
:py:class:`~chemfit.abstract_objective_function.QuantityComputerObjectiveFunction`,
ASE-based computers integrate with
:py:class:`~chemfit.fitter.Fitter`,
:ref:`combined_objective_functions`, and :ref:`parallel_execution`.
