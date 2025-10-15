.. ChemFit documentation master file, created by
   sphinx-quickstart on Thu Jun  5 11:32:11 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

################
ChemFit
################

**ChemFit**, is a package to support fitting the parameters of potentials described by an ASE calculator by minimizing objective functions.

-------------------------

**Features:**

- An extendable base class for ASE based objective functions (See :ref:`overview_objective_functions`)
- Implementations of ready-to-use objective functions (energy and structure based)
- A fitting module with convenient wrappers around optimization backends

**Highlights**:

- **Flexibility:** The objective functions can be combined arbitrarily with custom calculators and ways to create atoms.
- **Parallelization:** Objective functions can be parallelized over different contributing terms using ``mpi4py`` **without the headache of pickling custom calculators.** The lazy-loading mechanism ensures no superfluous file IO is performed.

-------------------------


.. _quickstart:

*************
Quickstart
*************

To fit the parameters of a potential in **ChemFit**, you need to provide three basic components:

1. A *factory* that knows how to construct the :py:class:`ase.Atoms` object.
   If your configurations are stored on disk, use :py:func:`~chemfit.ase_objective_function.PathAtomsFactory`.
2. A *factory* that knows how to construct the calculator.
3. A *function* that knows how to apply a given parameter dictionary to the calculator.

All of these are passed as callable “factory functions” into the objective function.
For details, see :ref:`ase_objective_function_api`.

.. note::

   We do not create the atoms and calculator outside the objective function.
   This is crucial for the **lazy-loading** mechanism, which ensures that—when running with MPI—
   each rank only reads the files it needs, avoiding redundant file I/O.

The following toy example demonstrates how to fit the parameters of a simple **Lennard–Jones potential**.

.. code-block:: python

   import numpy as np
   from ase import Atoms
   from ase.calculators.lj import LennardJones
   from chemfit.abstract_objective_function import QuantityComputerObjectiveFunction
   from chemfit.ase_objective_function import SinglePointASEComputer
   from chemfit.combined_objective_function import CombinedObjectiveFunction
   from chemfit.fitter import Fitter

   ########################
   # Define the factories
   ########################

   # Factory that creates an Atoms object for a dimer at distance r
   class LJAtomsFactory:
       def __init__(self, r: float):
           p0 = np.zeros(3)
           p1 = np.array([r, 0.0, 0.0])
           self.atoms = Atoms(positions=[p0, p1])

       def __call__(self):
           return self.atoms

   # Factory that attaches a Lennard–Jones calculator
   def construct_lj(atoms: Atoms):
       atoms.calc = LennardJones(rc=2000)

   # Function to apply parameters to the calculator
   def apply_params_lj(atoms: Atoms, params: dict[str, float]):
       atoms.calc.parameters.sigma = params["sigma"]
       atoms.calc.parameters.epsilon = params["epsilon"]

   ################################
   # Build the objective function
   ################################

   # Analytical reference energies
   def e_lj(r, eps, sigma):
       return 4.0 * eps * ((sigma / r) ** 6 - 1.0) * (sigma / r) ** 6

   eps = 1.0
   sigma = 1.0
   r_min = 2 ** (1.0 / 6.0) * sigma
   r_list = np.linspace(0.925 * r_min, 3.0 * sigma)

   # Create one objective term per configuration
   terms = []
   for r in r_list:
       ref_e = e_lj(r, eps, sigma)
       computer = SinglePointASEComputer(
           calc_factory=construct_lj,
           param_applier=apply_params_lj,
           atoms_factory=LJAtomsFactory(r),
           tag=f"lj_{r:.2f}",
       )
       term = QuantityComputerObjectiveFunction(
           loss_function=lambda q, e=ref_e: (q["energy"] - e) ** 2,
           quantity_computer=computer,
       )
       terms.append(term)

   # Combine all terms into a single objective
   ob = CombinedObjectiveFunction(terms)

   ################################
   # Fit the parameters
   ################################

   initial_params = {"epsilon": 2.0, "sigma": 1.5}

   fitter = Fitter(ob, initial_params=initial_params)
   opt_params = fitter.fit_scipy(options=dict(disp=True))

   print(f"Optimal parameters: {opt_params}")

-------------------------

This example uses the same conceptual building blocks that you will also use
for more complex calculators, including the SCME potential described in :ref:`example_scme`.

*************
Contents
*************

.. toctree::
   :maxdepth: 2

   src/installation
   src/usage/overview.rst
   src/usage/abstract_interface.rst
   src/usage/fitter.rst
   src/usage/ase_objective_function_api.rst
   src/usage/combined_objective_function.rst
   src/usage/mpi.rst
   src/usage/example_scme.rst
   src/development/development.rst
   src/api/modules
