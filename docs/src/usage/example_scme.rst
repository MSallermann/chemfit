#################################################
Fitting a dimer binding curve with the SCME Code
#################################################

.. note::

    In this example we will learn how to find the optimal SCME parameters to reproduce the binding energies of an H2O dimer.


Obtaining data
####################

The first step is, of course, to have all the reference configurations as well as the reference energies ready.

.. note::
    **Example data** for a dimer binding energies computed with the PBE functional can be found `here. <https://github.com/MSallermann/SCMEFitting/tree/9ffdc77d2c7a5144618b55615ce6211028aedd3c/tests/test_configurations_scme>`_
    To follow this example download all of these and save it in a folder called ``./data``, next to your script.

Generally, it is a good idea to store this information (that is paths to the reference configurations, reference energies and some tags) in a file somewhere on your computer. In the example data this file is called ``energies.csv``. It has three columns (if you ignore the index column): ``file``, ``reference_energy`` and ``tag``.

Today, we are very lucky since the :py:func:`scme_fitting.data_utils.process_csv` function provides a utility to parse exactly this information from a CSV file:

.. code-block:: python

    from scme_fitting.data_utils import process_csv
    paths, tags, energies = process_csv("./data/energies.csv")

Of course, you are free to obtain the list of paths, tags and energies in any other way as well.

Deciding initial parameters
#################################

Further we have to decide the default parameters of the SCME to be used. 
The default parameters are an instance of :py:class:`scme_fitting.scme_setup.SCMEParams` (a Pydantic model which encompasses all "user facing" parameters of the SCME 2.0 code).

Not all of the default parameters will be changed during the optimization, but even if they remain constant they need to have a value ... duh.

Here is an example of how the default params can be constructed:

.. code-block:: python

    from scme_fitting.utils import create_initial_params
    from scme_fitting.scme_setup import SCMEParams

    # We construct an SCMEParams instance, explicitly setting some parameters the others are set to the defaults specified in SCMEParams
    SCMEParams(
        Ar_OO=299.5695377280358,
        Br_OO=-0.14632711560656822,
        Cr_OO=-2.0071714442805715,
        r_Br=5.867230272424719,
        dms=True,
        qms=True,
    )

.. warning::
    The code snippet above relies on the defaults set in the :py:class:`scme_fitting.scme_setup.SCMEParams` class.
    It might be a good idea to **review these defaults** and to **not rely on them** as they might be subject to change.

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

Lastly, we should decide if we want to use monomer expansions in the style of the generalized SCME code. These are supplied in the form of a path to an HDF5 file (``path_to_scme_expansions`` argument) and a corresponding key to the expansion dataset in this file (``parametrization_key`` argument).

If any of these are ``None``, the generalized SCME will **not** be used.


Instantiating the factory functors
####################################

While it is completely possible to supply our own factor functions, we will use the predefined ones from the :py:mod:`scme_fitting.scme_objective_function` module:

.. code-block:: python

    from scme_fitting.scme_objective_function import SCMECalculatorFactory, SCMEParameterApplier

    # we do not use the generalized SCME in this example
    calc_factory = SCMECalculatorFactory(
                        default_scme_params=default_params,
                        path_to_scme_expansions=None, 
                        parametrization_key=None
                    )

    param_applier = SCMEParameterApplier()


Instantiating the objective function
####################################

We now simply instantiate the objective function by passing the factory functors and the lists of paths, energies and tags:

.. code-block:: python

    from scme_fitting.multi_energy_objective_function import MultiEnergyObjectiveFunction

    scme_objective_function = MultiEnergyObjectiveFunction(
        calc_factory=calc_factory,
        param_applier=param_applier,
        path_to_reference_configuration_list=paths,
        reference_energy_list=energies,
        tag_list=tags,
    )


Performing the fit
######################################

Simply pass the objective function to an instance of the ``Fitter`` class and write some outputs

.. code-block:: python

    fitter = Fitter(
        objective_function=scme_objective_function,
    )

    # All keyword arguments except `initial_parameters` get forwarded to scipy.minimize
    optimal_params = fitter.fit_scipy(
        initial_parameters=initial_params, tol=1e-4, options=dict(maxiter=50, disp=True)
    )

    # After the fit, this will write some useful outputs
    scme_objective_function.write_output(
        "output_dimer_binding",
        initial_params=initial_params,
        optimal_params=optimal_params,
    )


Expected results
######################################

After the fit there should be a plot ``plot_energy.png`` in the ``output_dimer_binding`` folder.
It should look something like 

.. image:: /src/_static/plot_dimer_binding_scme.png
   :alt: dimer_binding_scme
   :align: center
   :width: 80%