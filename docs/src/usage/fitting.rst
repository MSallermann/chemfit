#################
Fitting
#################


Using the ``EnergyObjectiveFunction``
########################################

The class :py:class:`scme_fitting.ase_objective_function.EnergyObjectiveFunction` represents a **single** reference configuration and energy pair.
Its main use is to serve as a building block for more complex objective functions.

This objective function has the form

.. math::

   O =  w \cdot \left| E_\text{pred}(\{r\}_\text{ref}) - E_\text{ref} \right|^2,

where :math:`w` is a weight factor, :math:`E_\text{pred}(\{r\}_\text{ref})` is the potential energy of the reference configuration predicted by the calculator and :math:`E_\text{ref}` is the reference energy.

If we want to use this objective function in isolation, we need at least
    - A filepath to a reference configuration of atom positions
    - A target energy associated to this reference configuration. This energy might for example have been computed from an ab-initio code.
    - A ``CalculatorFactory``
    - A ``ParameterApplier``
    - Optionally (but recommended) a ``tag``, which is a string identifier for bookkeeping purposes


.. note::
    The reference atom positions should be saved in a format, which is parseable by ASE's ``io.read`` function (https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.read) function.

    **Important**: If the file contains multiple "images" of atoms, the **first image** will be selected as the reference configuration. 

From these pieces of information we can construct the objective function:

.. code-block:: python

    from scme_fitting.ase_objective_function import EnergyObjectiveFunction

    from my_calculator import MyCalculator

    class MyCalculatorFactory:
        def __init__(self, some_parameter):
            self.some_parameter = some_parameter
        def __call__(self, atoms):
            atoms.calc = MyCalculator(self.some_parameter)

    class MyCalculatorParameterApplier:
        def __call__(self, atoms, params):
            atoms.calc.my_params.x = params["x"]
            atoms.calc.my_params.y = params["y"]

    # assume we have the atom positions saved as `atoms.xyz` and we know the reference energy is 1.0 eV
    objective_function = EnergyObjectiveFunction(
            calc_factory= MyCalculatorFactory(some_parameter=2), 
            param_applier = MyCalculatorParameterApplier(),
            path_to_reference_configuration = "atoms.xyz",
            tag = "my_tag",
            reference_energy = 1.0
        )

    # Evaluate the objective function at x=2.0 and y=1.0
    val = objective_function( {"x" : 2.0, "y": 1.0} )


Using the ``MultiEnergyObjectiveFunction``
##############################################

:py:class:`scme_fitting.multi_energy_objective_function.MultiEnergyObjectiveFunction` is used in much the same way as it's single energy counterpart.
The only difference is that, instead of a single configuration/energy pair, it supports multiple - leading to an objective function value of

.. math::
    O = \sum_i  w^i \cdot \left| E^i_\text{pred}(\{r^i\}_\text{ref}) - E^i_\text{ref} \right|^2,

where each index :math:`i` refers to a separate configuration/energy pair.

Consequently instead of a single ``path_to_reference_configuration`` argument the initializer takes a whole list. Fittingly (*wink wink*), called ``path_to_reference_configuration_list``.

Two other initializer arguments enjoy a similar promotion, namely: ``reference_energy_list`` and ``tag_list``.

Crucially, the objective function still takes only a single ``parameter_applier`` and ``calculator_factory``.

.. code-block:: python

    from scme_fitting.multi_energy_objective_function import MultiEnergyObjectiveFunction

    # ... assume the same definitions for `MyCalculatorFactory` and `MyCalculatorParameterApplier` from above

    objective_function = MultiEnergyObjectiveFunction(
            calc_factory = MyCalculatorFactory(some_parameter=2), 
            param_applier = MyCalculatorParameterApplier(),
            path_to_reference_configuration_list = ["atoms_1.xyz", "atoms_2.xyz"],
            tag_list = ["my_tag_1", "my_tag_2"],
            reference_energy_list = [1.0, 2.0]
        )

    # Evaluate the objective function at x=2.0 and y=1.0
    val = objective_function( {"x" : 2.0, "y": 1.0} )


The ``MultiEnergyObjectiveFunction`` has a convenience function (:py:meth:`scme_fitting.multi_energy_objective_function.MultiEnergyObjectiveFunction.write_output`) to write a "report" (various json files and plots) to an output folder after a fit has been performed.
It can be used like this

.. code-block:: python

    initial_params = {"x": 1.0, "y": 1.0}

    opt_params = fitter.fit_scipy(initial_params, options=dict(disp=True, tol=1e-5))

    objective_function.write_output(
        "test_my_calculator_output",
        initial_params=initial_params,
        optimal_params=opt_params,
    )


Using the ``CombinedObjectiveFunction``
#########################################

The :py:class:`scme_fitting.combined_objective_function.CombinedObjectiveFunction` class is used to turn a list of individual objective functions into a single objective function which is the (weighted) sum of the individual terms.

Using it directly is likely not needed. You are more likely to indirectly use it via the derived class :py:class:`scme_fitting.multi_energy_objective_function.MultiEnergyObjectiveFunction`.

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