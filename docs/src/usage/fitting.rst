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


Fitting a dimer binding curve with ``MultiEnergyObjectiveFunction``
#####################################################################

Let's assume you want to find SCME parameters to reproduce a dimer binding curve. The first step is of course to have all the reference configurations as well as the reference_energies ready.

Generally, it is a good idea to store this information (paths to the reference configurations, reference energies and some identifiers aka tags) in a file somewhere on your computer.

The :py:func:`scme_fitting.data_utils.process_csv` function provides a utility to parse this information from a csv file. 

It should be quite obvious how to use this function:

.. code-block:: python

    from scme_fitting.data_utils import process_csv
    paths, tags, energies = process_csv("./data/energies.csv")


**Example data** for a dimer binding curve can be found `here <https://github.com/MSallermann/SCMEFitting/tree/9ffdc77d2c7a5144618b55615ce6211028aedd3c/tests/test_configurations_scme>`_


Two further thing we have to decide are (i) the default parameters of the SCME to be used and (ii) which of these default parameters we want to optimize and what their initial values are (most of the time we will want to set the initial values to the default values). 

The default parameters are an instance of :py:class:`scme_fitting.scme_setup.SCMEParams` (a pydantic model which encompasses all "user facing" parameters of the SCME 2.0 code), whereas the initial parameters simply are a ``dict[str,float]``. Obviously, the initial parameters are a subset of the default parameters.

Here is how we might construct these parameters
.. code-block:: python

    from scme_fitting.scme_setup import SCMEParams

    # we can use the empty constructor to get some default-default params :)
    # change td, just because
    default_params = SCMEParams(td=2.0)
