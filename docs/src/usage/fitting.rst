#################
Fitting
#################


Using the ``EnergyObjectiveFunction``
########################################
To use :py::class:`scme_fitting.scme_objective_function.EnergyObjectiveFunction`, we need (i) a filepath to a reference configuration of atom positions and (ii) a target energy associated to this reference configuration. This energy might for example have been computed from an ab-initio code.

The reference atom positions should be saved in a format, which is parseable by the ASE's ``io.read`` (https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.read) function.
**Important**: If the file contains multiple "images" of atoms, the first image will be selected as the reference configuration. 

From these two pieces of information we can construct an objective function:

.. code-block:: python

    from scme_fitting.scme_objective_function import EnergyObjectiveFunction

    # assume we have the atom positions saved as `atoms.xyz`
    objective_function = EnergyObjectiveFunction(path_to_reference_configuration="atoms.xyz", reference_energy=1.0)

    # Evaluate the objective function
    val = objective_function( {"td" : 2.0} )


Using the ``CombinedObjectiveFunction``
########################################


Dimer binding curve example
#################################

Let's assume you want to find SCME parameters to reproduce a dimer binding curve.