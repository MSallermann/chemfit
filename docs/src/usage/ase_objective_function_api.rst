############################
ASE Objective Function API
############################


This page shows how to implement and use the ASE-specific "functors" (callable objects) that plug into the ASEObjectiveFunction framework via the
:py:class:`scme_fitting.ase_objective_function.CalculatorFactory`,
:py:class:`scme_fitting.ase_objective_function.ParameterApplier`, and
:py:class:`scme_fitting.ase_objective_function.AtomsPostProcessor` protocols.

CalculatorFactory
############################

A **CalculatorFactory** is any callable implementing the :py:class:`scme_fitting.ase_objective_function.CalculatorFactory` protocol.
This means it must be callable with the signature

.. code-block:: python

    factory(atoms: ase.Atoms) -> None

It must construct a calculator and attach it to the ``atoms.calc`` member.

Example: LJ calculator factory

.. code-block:: python

    from ase.calculators.lj import LennardJones

    def construct_lj(atoms: Atoms):
        atoms.calc = LennardJones(rc=2000)

For a more sophisticated example, see :py:class:`scme_fitting.scme_objective_function:SCMECalculatorFactory`


ParameterApplier
############################

A **ParameterApplier** updates the attached calculator's internal parameters before each evaluation. The signature is

.. code-block:: python

    from ase import Atoms

    applier(atoms: ase.Atoms, params: dict[str,float]) -> None


Example: LJ parameter applier

.. code-block:: python

    from ase import Atoms

    def apply_params_lj(atoms: Atoms, params: dict[str, float]):
        atoms.calc.parameters.sigma = params["sigma"]
        atoms.calc.parameters.epsilon = params["epsilon"]

For a more sophisticated example see :py:class:`scme_fitting.scme_objective_function.SCMEParameterApplier`.


Putting it all together
############################

Pass these functors into an ASE-based objective function. 
The following code fits the "sigma" and "epsilon" parameters of the ``LennardJones`` calculator.
(For a working script see the ``test_lj`` unit test).

.. code-block:: python

    from ase.calculators.lj import LennardJones
    from ase import Atoms
    import numpy as np
    from pathlib import Path

    from scme_fitting.multi_energy_objective_function import MultiEnergyObjectiveFunction
    from scme_fitting.fitter import Fitter

    # Prepare data
    # ...
    # paths, tags, energies = prepare_data(r_list, output_folder, eps=eps, sigma=sigma)

    eps = 1.0
    sigma = 1.0

    def construct_lj(atoms: Atoms):
        atoms.calc = LennardJones(rc=2000)

    def apply_params_lj(atoms: Atoms, params: dict[str, float]):
        atoms.calc.parameters.sigma = params["sigma"]
        atoms.calc.parameters.epsilon = params["epsilon"]

    ob = MultiEnergyObjectiveFunction(
        calc_factory=construct_lj,
        param_applier=apply_params_lj,
        tag_list=tags,
        path_to_reference_configuration_list=paths,
        reference_energy_list=energies,
    )

    fitter = Fitter(ob)

    initial_params = {"epsilon": 2.0, "sigma": 1.5}

    opt_params = fitter.fit_scipy(initial_params, options=dict(disp=True, tol=1e-5))

    print(opt_params)


AtomsPostProcessor
############################

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

It is passed to the initializer of :py:class:`scme_fitting.ASEObjectiveFunction`.
