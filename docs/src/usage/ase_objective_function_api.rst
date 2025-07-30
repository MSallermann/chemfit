.. _ase_objective_function_api:

############################
ASE Objective Function API
############################


This page shows how to implement and use the ASE-specific "functors" (callable objects) that plug into the ASEObjectiveFunction framework via the
:py:class:`~chemfit.ase_objective_function.CalculatorFactory`,
:py:class:`~chemfit.ase_objective_function.ParameterApplier`, and
:py:class:`~chemfit.ase_objective_function.AtomsPostProcessor` protocols.

CalculatorFactory
############################

A **CalculatorFactory** is any callable implementing the :py:class:`~chemfit.ase_objective_function.CalculatorFactory` protocol.
This means it must be callable with the signature

.. code-block:: python

    def factory(atoms: ase.Atoms) -> None
        ...

It must construct a calculator and attach it to the ``atoms.calc`` member.

Example: LJ calculator factory

.. code-block:: python

    from ase.calculators.lj import LennardJones

    def construct_lj(atoms: Atoms):
        atoms.calc = LennardJones(rc=2000)

For a more sophisticated example, see :py:class:`~chemfit.scme_factories:SCMECalculatorFactory`


ParameterApplier
############################

A **ParameterApplier** updates the attached calculator's internal parameters before each evaluation. The signature is

.. code-block:: python

    from ase import Atoms

    applier(atoms: ase.Atoms, params: dict) -> None


Example: LJ parameter applier

.. code-block:: python

    from ase import Atoms

    def apply_params_lj(atoms: Atoms, params: dict[str, float]):
        atoms.calc.parameters.sigma = params["sigma"]
        atoms.calc.parameters.epsilon = params["epsilon"]

For a more sophisticated example see :py:class:`~chemfit.scme_factories.SCMEParameterApplier`.


Optional factories
############################

In the following some optional factories, besides **ParameterApplier** and **CalculatorFactory**, are described. 
These can be used to make the **ASEObjectiveFunction** more flexible.

AtomsFactory
----------------------
In the example above, the ``ase.Atoms`` object is created from a path to a configuration file.
In some cases it might be required to have more fine grained control over the creation of the atoms object.
For such situations :py:class:`~chemfit.ase_objective_function.ASEObjectiveFunction` provides the option to pass an implementation 
of an **AtomsFactory** protocol (defined in :py:class:`~chemfit.ase_objective_function.AtomsFactory`) in the ``atoms_factory`` argument of the initializer (:py:meth:`~chemfit.ase_objective_function.ASEObjectiveFunction`).

.. note::
    Under the hood the ``path_to_reference_configuration`` argument is just a convenient way to construct a :py:class:`~chemfit.ase_objective_function.PathAtomsFactory`

.. warning::
    If both ``atoms_factory`` and ``path_to_reference_configuration`` are specified, ``atoms_factory`` takes precedence.

One example, where we might want to specify the ``atoms_factory`` explicitly is when the index of the image in the reference file is not ``0``:

.. code-block:: python

    from chemfit.ase_objective_function import EnergyObjectiveFunction, PathAtomsFactory

    # explicitly instantiate the PathAtomsFactory to read the second image in 'atoms.xyz'
    ob = EnergyObjectiveFunction( 
        # ... pass all other args
        atoms_factory = PathAtomsFactory(path="atoms.xyz", index=1) 
    )

As a more complex example, lets define a **LJAtomsFactory** to simplify the construction of the LennardJones objective function from above:

.. code-block:: python

    from ase.calculators.lj import LennardJones
    from ase import Atoms
    import numpy as np
    from chemfit.multi_energy_objective_function import create_multi_energy_objective_function
    from chemfit.fitter import Fitter


    class LJAtomsFactory:
        def __init__(self, r: float):
            p0 = np.zeros(3)
            p1 = np.array([r, 0.0, 0.0])
            self.atoms = Atoms(positions=[p0, p1])

        def __call__(self):
            return self.atoms


    def e_lj(r, eps, sigma):
        return 4.0 * eps * ((sigma / r) ** 6 - 1.0) * (sigma / r) ** 6


    def construct_lj(atoms: Atoms):
        atoms.calc = LennardJones(rc=2000)


    def apply_params_lj(atoms: Atoms, params: dict[str, float]):
        atoms.calc.parameters.sigma = params["sigma"]
        atoms.calc.parameters.epsilon = params["epsilon"]


    eps = 1.0
    sigma = 1.0
    r_min = 2 ** (1 / 6) * sigma
    r_list = np.linspace(0.925 * r_min, 3.0 * sigma)


    ob = create_multi_energy_objective_function(
        calc_factory=construct_lj,
        param_applier=apply_params_lj,
        tag_list=[f"lj_{r:.2f}" for r in r_list],
        reference_energy_list=[e_lj(r, eps, sigma) for r in r_list],
        path_or_factory_list=[LJAtomsFactory(r) for r in r_list], # <--- Now the atoms are constructed directly in memory
    )


AtomsPostProcessor
----------------------

An optional **AtomsPostProcessor** runs on the raw Atoms immediately after loading (before the calculator is attached). Its signature is

.. code-block:: python

    processor(atoms: ase.Atoms) -> None


You can use this hook to reorder atoms, apply constraints, or modify positions.

Example: trivial post-processor

.. code-block:: python

    from ase import Atoms

    def zero_center(atoms: Atoms) -> None:
        # shift center of mass to origin
        com = atoms.get_center_of_mass()
        atoms.positions -= com

It is passed to the initializer of :py:class:`~chemfit.ase_objective_function.ASEObjectiveFunction`.
