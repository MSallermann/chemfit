############################################################
Example: Fitting a dimer binding curve with the SCME Code
############################################################

.. note::

    In this example we will learn how to find the optimal SCME parameters to reproduce the binding energies of an H2O dimer.


Obtaining data
####################

The first step is, of course, to obtain all the reference configurations as well as the reference energies.

.. note::
    **Example data** for dimer binding energies, computed with the PBE functional, can be found `here. <https://github.com/MSallermann/SCMEFitting/tree/9ffdc77d2c7a5144618b55615ce6211028aedd3c/tests/test_configurations_scme>`_
    To follow this little tutorial, download all of these files and save them in a folder ``./data``.

Generally, it is a good idea to store the paths to the reference configurations, reference energies and some tags in a file somewhere on your computer.
In the example data above, this file is called ``energies.csv``. It has three columns (if you ignore the index): ``file``, ``reference_energy`` and ``tag``.

We are very lucky since the :py:func:`scme_fitting.data_utils.process_csv` function provides a utility to parse exactly this information from a CSV file:

.. code-block:: python

    from scme_fitting.data_utils import process_csv
    paths, tags, energies = process_csv("./data/energies.csv")

Of course, you are free to obtain the list of paths, tags and energies in any other way as well.


Deciding initial parameters
#################################

Further we have to decide the default parameters of the SCME to be used. 
The default parameters are an instance of :py:class:`scme_fitting.scme_setup.SCMEParams` (a Pydantic model which encompasses all "user facing" parameters of the SCME 2.0 code).

Not all of the default parameters will be changed during the optimization, but even if they remain constant they need to have a value ...

Here is an example of how the default params can be constructed:

.. code-block:: python

    from scme_fitting.utils import create_initial_params
    from scme_fitting.scme_setup import SCMEParams

    # We construct an SCMEParams instance, explicitly setting some parameters the others are set to the defaults specified in SCMEParams
    default_params = SCMEParams(
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

We should now also decide which of the parameters we want to optimize in order to approach the reference energies. 
Most of the time we will want to set the initial values of the adjustable parameters to the corresponding default values - but this is not required.
If we indeed want to use the default parameters as initial values, we can make use of the function :py:func:`scme_fitting.utils.create_initial_params`.
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

Lastly, we should decide if we want to use monomer expansions in the style of the generalized SCME code.
These are supplied in the form of a path to an HDF5 file (``path_to_scme_expansions`` argument) and a corresponding key to the expansion dataset in this file (``parametrization_key`` argument).

If any of these are ``None``, the generalized SCME will **not** be used.


Instantiating the factory functors
####################################

While it is completely possible to supply your own factory functions, we will use the predefined ones from the :py:mod:`scme_fitting.scme_factories` module:

.. code-block:: python

    from scme_fitting.scme_factories import SCMECalculatorFactory, SCMEParameterApplier

    calc_factory = SCMECalculatorFactory(
                        default_scme_params=default_params,
                        path_to_scme_expansions=None, # we do not use the generalized SCME in this example
                        parametrization_key=None
                    )

    param_applier = SCMEParameterApplier()


Instantiating the objective function
####################################

We now simply instantiate the objective function by passing the factory functors and the lists of paths, energies and tags:

.. code-block:: python

    from scme_fitting.multi_energy_objective_function import MultiEnergyObjectiveFunction

    scme_factories = MultiEnergyObjectiveFunction(
        calc_factory=calc_factory,
        param_applier=param_applier,
        path_to_reference_configuration_list=paths,
        reference_energy_list=energies,
        tag_list=tags,
    )


Performing the fit
######################################

Pass the objective function to an instance of the ``Fitter`` class and write some outputs

.. code-block:: python

    fitter = Fitter(
        objective_function=scme_factories,
        initial_params = initial_params
    )

    # All keyword arguments get forwarded to scipy.minimize
    optimal_params = fitter.fit_scipy(
        tol=1e-4, options=dict(maxiter=50, disp=True)
    )

    # After the fit, this will write some useful outputs
    scme_factories.write_output(
        "output_dimer_binding",
        initial_params=initial_params,
        optimal_params=optimal_params,
    )


Expected results
######################################

After the call to the ``write_output`` function, there should be a ``output_dimer_binding/plot_energy.png`` file.
It should look something like this

.. image:: /src/_static/plot_dimer_binding_scme.png
   :alt: dimer_binding_scme
   :align: center
   :width: 80%


The optimal parameters should be saved as a json file called ``output_dimer_binding/optimal_params.json``:

.. code-block:: javascript

    {
        "td": 1.7307507548872705,
        "te": 3.3319409063023553,
        "C6": 334.4715463605395,
        "C8": 1146.9930705691029,
        "C10": 33441.07679944017
    }

Lastly, there should be a CSV file ``output_dimer_binding/energies.csv`` containing information about each reference configuration in each row.