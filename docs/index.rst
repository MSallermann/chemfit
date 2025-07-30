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

- **Flexibility:** The objective functions can be combined arbitarily with custom calculators and ways to create atoms.
- **Parallelization:** Objective functions can be parallelized over different contributing terms using ``mpi4py`` **without the headache of pickling custom calculators.** The lazy-loading mechanism ensures no superfluous file IO is performed.

-------------------------

.. _quickstart:

*************
Quickstart
*************

In order to use the provided objective functions you have to provide three pieces of information

- How to construct the :py:class:`ase.Atoms` object. If you saved them to a file you can simply pass the path, which under the hood uses the :py:func:`~chemfit.ase_objective_function.PathAtomsFactory`.
- How to construct the calculator. This one you will have to define yourself. It's not hard though.
- How to apply a parametrization (essentially a dictionary) to the calculator. This one you will have to define yourself as well. It provides the "link" between the dictionary of parameters to optimize and the calculator and depends on the specific calculator you are using.

All of this is specified in so-called factory functions.
For further information see :ref:`ase_objective_function_api`.

.. note::

   You might ask yourself, why we don't just create the atoms and the calculator outside of the objective function and pass them into the initializer.

   We avoid this to support the lazy-loading mechanism, which can greatly help working with the MPI parallelization, as it ensures that every rank only reads the files it absolutely has to.

The following toy example shows how to fit the parameters of a Lennard Jones potential.

.. code-block:: python

   from ase.calculators.lj import LennardJones
   from ase import Atoms
   import numpy as np
   from chemfit.multi_energy_objective_function import create_multi_energy_objective_function
   from chemfit.fitter import Fitter

   ########################
   # Define the factories
   ########################

   # This tells the objective function how to create a specific atoms object.
   # Here it places on atom at the origin and another one at a
   # distance `r` along the x-axis.s
   class LJAtomsFactory:
      def __init__(self, r: float):
         p0 = np.zeros(3)
         p1 = np.array([r, 0.0, 0.0])
         self.atoms = Atoms(positions=[p0, p1])

      def __call__(self):
         return self.atoms

   # This tells the objective function how to construct the LennardJones calculator 
   def construct_lj(atoms: Atoms):
      atoms.calc = LennardJones(rc=1000)

   # Lastly, this tells the objective function how to apply the parametrization
   def apply_params_lj(atoms: Atoms, params: dict[str, float]):
      atoms.calc.parameters.sigma = params["sigma"]
      atoms.calc.parameters.epsilon = params["epsilon"]

   ################################
   # Create the objective function
   ################################

   # This is the "target" function we use
   def e_lj(r, eps, sigma):
      return 4.0 * eps * ((sigma / r) ** 6 - 1.0) * (sigma / r) ** 6

   eps = 1.0
   sigma = 1.0

   r_min = 2 ** (1.0 / 6.0) * sigma
   r_list = np.linspace(0.925 * r_min, 3.0 * sigma)

   ob = create_multi_energy_objective_function(
      calc_factory=construct_lj,
      param_applier=apply_params_lj,
      tag_list=[f"lj_{r:.2f}" for r in r_list],
      reference_energy_list=[e_lj(r, eps, sigma) for r in r_list],
      path_or_factory_list=[LJAtomsFactory(r) for r in r_list],
   )

   ################################
   # Find the optimal parameters
   ################################

   initial_params = {"epsilon": 2.0, "sigma": 1.5}

   fitter = Fitter(ob, initial_params=initial_params)
   opt_params = fitter.fit_scipy(options=dict(disp=True))

   print(f"Optimal parameters {opt_params}")


*************
Contents
*************

.. toctree::
   :maxdepth: 2

   src/installation
   src/usage/overview.rst
   src/usage/fitter.rst
   src/usage/ase_objective_function_api.rst
   src/usage/objective_functions.rst
   src/usage/example_scme.rst
   src/usage/mpi.rst
   src/development/development.rst
   src/api/modules