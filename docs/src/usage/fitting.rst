#################
Fitting
#################


Using the ``EnergyObjectiveFunction``
########################################

:py:class:`scme_fitting.scme_objective_function.EnergyObjectiveFunction` represents a **single** reference configuration and energy pair. It's main use is to serve as a building block for more complex objective function. See :ref:`dimer_binding` for an example.

If we want to use this objective function in isolation, we need (i) a filepath to a reference configuration of atom positions and (ii) a target energy associated to this reference configuration. This energy might for example have been computed from an ab-initio code.

.. note::
    The reference atom positions should be saved in a format, which is parseable by ASE's ``io.read`` function (https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.read) function.

    **Important**: If the file contains multiple "images" of atoms, the **first image** will be selected as the reference configuration. 

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

It can still come in handy, when combining different objective functions. 

Its use is demonstrated in the following:

.. code-block:: python

    from scme_fitting.combined_objective_function import CombinedObjectiveFunction

    def a(p):
        return 1.0 * p["x"]**2

    def b(p):
        return 1.0 * p["y"]**2

    objective_function = CombinedObjectiveFunction([a,b], [1.0, 2.0]) # is equivalent to x**2 + 2*y**2

    # Evaluate the objective function
    val = objective_function( {"x" : 1.0, "y" : 1.0} )

.. _dimer_binding:

Fitting a dimer binding curve with ``MultiEnergyObjectiveFunction``
#####################################################################

Let's assume you want to find the optimal SCME parameters to reproduce the energies in a dimer binding curve.
The first step is of course to have all the reference configurations as well as the reference_energies ready.

Generally, it is a good idea to store this information (paths to the reference configurations, reference energies and some identifiers aka tags) in a file somewhere on your computer.

The :py:func:`scme_fitting.data_utils.process_csv` function provides a utility to parse this information from a CSV file:

.. code-block:: python

    from scme_fitting.data_utils import process_csv
    paths, tags, energies = process_csv("./data/energies.csv")

Of course, you are free to obtain the list of paths, tags and energies in any other way as well.

**Example data** for a dimer binding curve can be found `here. <https://github.com/MSallermann/SCMEFitting/tree/9ffdc77d2c7a5144618b55615ce6211028aedd3c/tests/test_configurations_scme>`_

Further we have to decide the default parameters of the SCME to be used. 
The default parameters are an instance of :py:class:`scme_fitting.scme_setup.SCMEParams` (a Pydantic model which encompasses all "user facing" parameters of the SCME 2.0 code).

Not all of the default parameters will be changed during the optimization, but even if they remain constant they need to have a value ... duh.

Here is an example of how the default params can be constructed:

.. code-block:: python

    from scme_fitting.utils import create_initial_params
    from scme_fitting.scme_setup import SCMEParams

    # We construct an SCMEParams instance and explicitly change the 'td' setting
    # (the rest of the parameters will be set to the defaults specified in SCMEParams)
    default_params = SCMEParams(td=2.0)

.. warning::
    The code snippet above relies on the defaults set in the :py:class:`scme_fitting.scme_setup.SCMEParams` class.
    It might be a good idea to (i) review these defaults and (ii) not rely on them as they might be subject to change.

We should now decide which of the parameters we want to optimize in order to approach the reference energies and what their initial value should be. 
Most of the time we will want to set the initial values to the corresponding default values - but this is not required.
If we inded want to use the default parameters as initial values we can make use of the function :py:func:`scme_fitting.utils.create_initial_params`.
The code below demonstrates its use:

.. code-block:: python

    from scme_fitting.utils import create_initial_params

    # These should match members in default_params
    adjustable_params = ["td", "te", "C6", "C8", "C10"]

    # This creates a dictionary of initial params by fetching 
    # the corresponding values from the default params.
    # It is essentially equivalent to:
    #      initial_params = {k: dict(default_params)[k] for k in adjustable_params}
    initial_params = create_initial_params(adjustable_params, default_params)

Lastly, we should decide if we want to use monomer expansions in the style of the generalized SCME code. These are supplied in the form of a path to and HDF5 file (``path_to_scme_expansions`` argument) and a corresponding key to the expansion dataset in this fil  (``parametrization_key`` argument).
If any of these are ``None``, the generalized SCME will **not** be used.

.. note::
    The parameters of the monomer expansions can not be adjusted with the SCMEFitting package.

In the code below we elect to not use the generalized SCME.

.. code-block:: python

    scme_objective_function = MultiEnergyObjectiveFunction(
        default_scme_params=default_params,
        path_to_scme_expansions=None,
        parametrization_key=None,
        path_to_reference_configuration_list=paths,
        reference_energy_list=energies,
        tag_list=tags,
    )

Armed with this objective function, we can now perform a fit using the Fitter class

.. code-block:: python

    fitter = Fitter(
        objective_function=scme_objective_function,
    )

    # All keyword arguments except `initial_parameters` get forwarded to scipy.minimize
    optimal_params = fitter.fit_scipy(
        initial_parameters=INITIAL_PARAMS, tol=1e-4, options=dict(maxiter=50, disp=True)
    )

    # After the fit, this will write some useful outputs
    scme_objective_function.write_output(
        "test_output_multi_energy",
        initial_params=INITIAL_PARAMS,
        optimal_params=optimal_params,
    )