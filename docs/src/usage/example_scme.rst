.. _example_scme:

############################################################
Example: Fitting a dimer binding curve with the SCME Code
############################################################

.. note::

    In this example we will learn how to find the optimal SCME parameters to reproduce the binding energies of an H2O dimer.

******************
Obtaining data
******************

The first step is, of course, to obtain all the reference configurations as well as the reference energies.

.. note::
    **Example data** for dimer binding energies, computed with the PBE functional, can be found `here. <https://github.com/MSallermann/ChemFit/tree/9ffdc77d2c7a5144618b55615ce6211028aedd3c/tests/test_configurations_scme>`_
    To follow this little tutorial, download all of these files and save them in a folder ``./data``.

Generally, it is a good idea to store the paths to the reference configurations, reference energies and some tags in a file somewhere on your computer.
In the example data above, this file is called ``energies.csv``. It has three columns (if you ignore the index): ``file``, ``reference_energy`` and ``tag``.

We are very lucky since the :py:func:`~chemfit.data_utils.process_csv` function provides a utility to parse exactly this information from a CSV file:

.. code-block:: python

    from chemfit.data_utils import process_csv
    paths, tags, energies = process_csv("./data/energies.csv")

Of course, you are free to obtain the list of paths, tags and energies in any other way as well.


******************
Parameters
******************


This section describes how to specify the parameterization of the SCME.

Default Parameters
--------------------

First, we have to decide the default parameters of the SCME to be used, these are the parameters passed into the calculator upon initial construction.

The default parameters are supplied as a simple nested dictionary.

Not all of the default parameters will be changed during the optimization, but even if they remain constant they need to have a value ... specifying these constant values is the purpose of the ``default_params`` dictionary.

Here is an example of how the default params can be constructed:

.. code-block:: python

    from ase.units import Bohr

    default_params = {
        "dispersion": {
            "td": 4.7,
            "rc": 8.0 / Bohr,
            "C6": 46.4430e0,
            "C8": 1141.7000e0,
            "C10": 33441.0000e0,
        },
        "repulsion": {
            "Ar_OO": 299.5695377280358,
            "Br_OO": -0.14632711560656822,
            "Cr_OO": -2.0071714442805715,
            "r_Br": 5.867230272424719,
            "rc": 7.5 / Bohr,
        },
        "electrostatic": {
            "te": 1.2 / Bohr,
            "rc": 9.0 / Bohr,
            "NC": [1, 2, 1],
            "scf_convcrit": 1e-8,
            "max_iter_scf": 500,
        },
        "dms": True,
        "qms": True,
    }

.. warning::
    Every parameter, which is not specified in the default parameters, implicitly relies on the defaults set within the SCME code.
    It might be a good idea to **review these defaults** and to **not rely on them** as they might be subject to change.


Initial Parameters
--------------------

We should now also decide which of the parameters we want to optimize in order to approach the reference energies.
This is done by specifying a dictionary of initial parameters

.. code-block:: python

    initial_params = {
        "electrostatic": {"te": 2.0},
        "dispersion": {
            "td": 4.7,
            "C6": 46.4430e0,
            "C8": 1141.7000e0,
            "C10": 33441.0000e0,
        },
    }

.. note::

    In this example, every numeric leaf in ``initial_params`` is varied by the
    fitter, starting from the value shown above. Parameters that appear only in
    ``default_params`` remain fixed. More generally, leaf behavior depends on
    the selected optimization backend and, for Nevergrad, may be customized
    with a partial parametrization; see :ref:`fitter`.

.. note::

    If a key is found both in the `default_params` and the `initial_params`, the `initial_params` just overwrite it upon application of the parameters.


Using monomer expansions
-------------------------

Lastly, we should decide if we want to use monomer expansions in the style of the generalized SCME code.
These are supplied in the form of a path to an HDF5 file (``path_to_scme_expansions`` argument) and a corresponding key to the expansion dataset in this file (``parametrization_key`` argument).

If any of these are ``None``, the generalized SCME will **not** be used.


Instantiating the factory functors
************************************

While it is completely possible to supply your own factory functions, we will use the predefined ones from the :py:mod:`~chemfit.scme_factories` module:

.. code-block:: python

    from chemfit.scme_factories import SCMECalculatorFactory

    calc_factory = SCMECalculatorFactory(
                        default_scme_params=default_params,
                        path_to_scme_expansions=None, # we do not use the generalized SCME in this example
                        parametrization_key=None
                    )

Instantiating the objective function
******************************************************


For each configuration, we create an objective term from an
:py:class:`~chemfit.ase_objective_function.ASEComputer` and combine them in a
:py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`.

As the loss function we use

.. math::

    \text{L} = \sum_i \left( \frac{E_i - E^\text{ref}_i}{n_\text{atoms}} \right)^2.

.. code-block:: python

    from chemfit.ase_objective_function import ASEComputer, PathAtomsFactory
    from chemfit.combined_objective_function import CombinedObjectiveFunction

    def energy_loss(quantities, *, reference):
        return (
            (quantities["energy"] - reference) ** 2
            / quantities["n_atoms"] ** 2
        )

    def make_energy_term(path, reference):
        computer = ASEComputer(
            calculator_factory=calc_factory,
            atoms_factory=PathAtomsFactory(path),
        )
        return computer.with_loss(energy_loss, reference=reference)

    terms = [make_energy_term(p, e) for p, e in zip(paths, energies)]
    ob = CombinedObjectiveFunction(terms)


Performing the fit
************************************

Pass the objective function to an instance of the ``Fitter`` class and write some outputs

.. code-block:: python

    from pathlib import Path

    from chemfit.fitter import Fitter
    from chemfit.utils import dump_dict_to_file

    fitter = Fitter(
        objective_function = ob,
        initial_params = initial_params
    )

    # Additional keyword arguments are forwarded to scipy.minimize.
    optimal_params = fitter.fit_scipy(
        tol=1e-4, options=dict(maxiter=50)
    )

    # Inspect the optimized parameters and fitter contexts.
    print(f"{optimal_params = }")
    print(f"{fitter.contexts = }")

    # Optional: save the parameters to a JSON file
    dump_dict_to_file(
        Path("output_dimer_binding/optimal_params.json"),
        optimal_params,
    )


Getting per-term data
************************************

After the fit, evaluate the chosen parameters with an explicit context to
inspect per-term quantities.

.. note::

   A fitter context describes the last candidate evaluated in that candidate
   slot and separately stores its best observation. Nevergrad's recommendation
   is not necessarily the last evaluated candidate, so evaluate it explicitly
   when you need its complete child metadata.

.. code-block:: python

    from chemfit.abstract_objective_function import EvaluateContext

    ctx = EvaluateContext()
    ob(optimal_params, ctx)

    energy_fitted = [
        child["quantities"]["energy"]
        for child in ctx.meta["children"]
    ]


Expected results
************************************

If the fitted energies are plotted against the reference it should look something like this

.. image:: /src/_static/plot_dimer_binding_scme.png
   :alt: dimer_binding_scme
   :align: center
   :width: 80%

The call to :func:`~chemfit.utils.dump_dict_to_file` writes the nested optimized
parameter mapping to ``output_dimer_binding/optimal_params.json``. ChemFit does
not create an energies CSV automatically; write ``energy_fitted`` together
with the reference energies and tags using the table library of your choice.
