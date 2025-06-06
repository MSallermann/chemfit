#################
Fitting
#################


Using the ``EnergyObjectiveFunction``
########################################
To use :py:class:`scme_fitting.scme_objective_function.EnergyObjectiveFunction`, we need (i) a filepath to a reference configuration of atom positions and (ii) a target energy associated to this reference configuration. This energy might for example have been computed from an ab-initio code.

The reference atom positions should be saved in a format, which is parseable by ASE's ``io.read`` function (https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.read) function.
**Important**: If the file contains multiple "images" of atoms, the first image will be selected as the reference configuration. 

From these two pieces of information we can construct an objective function:

.. code-block:: python

    from scme_fitting.scme_objective_function import EnergyObjectiveFunction

    # assume we have the atom positions saved as `atoms.xyz`
    objective_function = EnergyObjectiveFunction(path_to_reference_configuration="atoms.xyz", reference_energy=1.0)

    # Evaluate the objective function
    val = objective_function( {"td" : 2.0} )


Using the ``CombinedObjectiveFunction``
#########################################
Using :py:class:`scme_fitting.combined_objective_function.CombinedObjectiveFunction` directly is likely not needed.
You are more likely to indirectly use it via the derived class :py:class:`scme_fitting.multi_energy_objective_function.MultiEnergyObjectiveFunction`.

Using it is relatively straight forward:

.. code-block:: python

    from scme_fitting.combined_objective_function import CombinedObjectiveFunction

    def a(p):
        return 1.0 * p["x"]**2

    def b(p):
        return 1.0 * p["y"]**2

    objective_function = CombinedObjectiveFunction([a,b], [1.0, 2.0]) # is equivalent to x**2 + 2*y**2

    # Evaluate the objective function
    val = objective_function( {"x" : 1.0, "y" : 1.0} )


Dimer binding curve example
#################################

Let's assume you want to find SCME parameters to reproduce a dimer binding curve.