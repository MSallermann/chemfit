.. _predefined_objective_functions:


Predefined objective functions
==================================


The abstract ``ObjectiveFunctor``
-------------------------------------

The :py:class:`~chemfit.abstract_objective_function.ObjectiveFunctor` class is an abstract base class for functor based objective functions in ChemFit.

Besides the obvious :py:meth:`~chemfit.abstract_objective_function.ObjectiveFunctor.__call__` method, which computes the value of the objective function for a given parameter set, there is only the :py:meth:`~chemfit.abstract_objective_function.ObjectiveFunctor.get_meta_data` method to be implemented.
This method is supposed to return a dictionary of meta data.

.. note::
    ``ChemFit`` also works with objective functions which do not implement the :py:meth:`~chemfit.abstract_objective_function.ObjectiveFunctor.get_meta_data` method (such as regular functions), but some functionality may be lost.


The combined objective function
-------------------------------------

The :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction` class is used to turn a list of individual functions into a combined objective function, formed by the (weighted) sum of the individual terms.

Its use is demonstrated in the following

.. code-block:: python

    from chemfit.combined_objective_function import CombinedObjectiveFunction

    def a(p):
        return 1.0 * p["x"]**2

    def b(p):
        return 1.0 * p["y"]**2

    objective_function = CombinedObjectiveFunction([a,b], [1.0, 2.0]) # is equivalent to x**2 + 2*y**2

    # Evaluate the objective function
    val = objective_function( {"x" : 1.0, "y" : 1.0} )

You can also add terms by using :py:meth:`~chemfit.combined_objective_function.CombinedObjectiveFunction.add`, like so

.. code-block:: python

    def c(p):
        return 2

    # now is equivalent to a+b+c
    objective_function.add(c)

Lastly, if you have one or more :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction` you can add them in a flat hierarchy with :py:meth:`~chemfit.combined_objective_function.CombinedObjectiveFunction.add_flat`:

.. code-block:: python

    # All of these append their terms/weights to the terms of the calling CombinedObjectiveFunction
    objective_function.add_flat(other_cob)
    objective_function.add_flat([other_cob, other_cob2])
    objective_function.add_flat([other_cob, other_cob2], [1.0, 2.0])


Why do we care?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Obviously this example makes it seem like the :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction` is not useful. After all, we could have just defined it ourselves like so

.. code-block:: python

    def objective_function(p):
        return a(b) + b(p)

There are two reasons, to use the :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`.


**Reason 1**: MPI parallelization
************************************************

Technically, :py:class:`~chemfit.mpi_wrapper_cob.MPIWrapperCOB` is also an objective function, since it implements the :py:class:`~chemfit.abstract_objective_function.ObjectiveFunctor` interface.
It can be used to make a :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction` "MPI aware". For more details see :ref:`mpi`.


**Reason 2:** Gathering meta data
************************************************

If the individual terms of the :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`, implement the ``get_meta_data`` method, we can easily collect the meta data in a list.

.. code-block:: python

    from chemfit.abstract_objective_function import ObjectiveFunctor
    from chemfit.combined_objective_function import CombinedObjectiveFunction

    class MyFunctor(ObjectiveFunctor):

        def __init__(self, f: float):
            self.f = f
            self.meta_data = {}

        def get_meta_data(self):
            return self.meta_data

        def __call__(self, params: dict) -> float:
            val = self.f * params["x"] ** 2
            self.meta_data["last_value"] = val
            return val

    def a(p):
        return p["y"] ** 2

    objective_function = CombinedObjectiveFunction(
        [a, MyFunctor(1), MyFunctor(2)]
    )  # is equivalent to y**2 + x**2 + 2.0*x**2

    # Evaluate the objective function
    val = objective_function({"x": 1.0, "y": 2.0})
    meta_data_list = objective_function.gather_meta_data()

    print(meta_data_list) # [None, {'last_value': 1.0}, {'last_value': 2.0}]

.. note::
    As you can see in the example above, ``None`` is returned if the ``get_meta_data`` method is not implemented.

.. note::
    The main use of :py:meth:`~chemfit.abstract_objective_function.ObjectiveFunctor.get_meta_data` is to gather information about individual terms in a :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction` (possibly collected from different MPI ranks).

.. tip::
    For objective functions with many terms, you can use ``pandas`` and the ``DataFrame.from_records`` method to turn a list of meta data dictionaries into a DataFrame and from there into e.g a CSV or any columnar format.

    .. code-block:: python

        import pandas as pd

        df = pd.DataFrame.from_records(meta_data_list)
        df.to_csv("meta_data.csv")


ASE based objective functions
-----------------------------------

The ASE based objective functions derive from :py:class:`chemfit.ase_objective_function.ASEObjectiveFunction`. 
They are meant for use with the "atomic simulation environment" (ASE).
All of these functions are designed for flexibility (See :ref:`ase_objective_function_api`) and can accommodate any ase calculator. 

``ChemFit`` provides a few pre-defined objective functions of that type, which are explained in the following.

**Custom ase-based objective functions** can be implemented by deriving from :py:class:`chemfit.ase_objective_function.ASEObjectiveFunction` and implementing the ``__call__(params : dict) -> float`` operator.


The energy based objective function for a single configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :py:class:`~chemfit.ase_objective_function.EnergyObjectiveFunction` represents a **single** reference configuration and energy pair.
Its main use is to serve as a building block for more complex objective functions.

This objective function has the form

.. math::
   O =  w \cdot \left| E_\text{pred}(\{r\}_\text{ref}) - E_\text{ref} \right|^2,

where :math:`w` is a weight factor, :math:`E_\text{pred}(\{r\}_\text{ref})` is the potential energy of the reference configuration predicted by the calculator and :math:`E_\text{ref}` is the reference energy.

If we want to use this objective function in isolation, we need at least
    - A filepath to a reference configuration of atom positions
    - A target energy associated to this reference configuration. This energy might for example have been computed from an ab-initio code.
    - A :py:class:`~chemfit.ase_objective_function.CalculatorFactory`
    - A :py:class:`~chemfit.ase_objective_function.ParameterApplier`
    - Optionally (but recommended) a ``tag``, which is a string identifier for book keeping purposes

.. note::
    The reference atom positions should be saved in a format, which is parseable by ASE's ``io.read`` function (https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.read) function.

    **Important**: If the file contains multiple "images" of atoms, the **first image** will be selected as the reference configuration. 


From these pieces of information we can construct the objective function:

.. code-block:: python

    from chemfit.ase_objective_function import EnergyObjectiveFunction

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


The ``MultiEnergyObjectiveFunction``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Technically, there is no separate ``MultiEnergyObjectiveFunction`` class (there used to be one).

The *function* :py:func:`~chemfit.multi_energy_objective_function.construct_multi_energy_objective_function` provides a convenient tool to construct a :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction` consisting only of :py:class:`~chemfit.ase_objective_function.EnergyObjectiveFunction`.

The objective function value is computed as

.. math::
    O = \sum_i  w^i \cdot \left| E^i_\text{pred}(\{r^i\}_\text{ref}) - E^i_\text{ref} \right|^2,

where each index :math:`i` refers to a separate configuration/energy pair.

Consequently instead of a single ``path_to_reference_configuration`` argument the initializer takes a whole list. Fittingly (*wink wink*), called ``path_to_reference_configuration_list``.

Two other initializer arguments enjoy a similar promotion, namely: ``reference_energy_list`` and ``tag_list``.

Crucially, the objective function takes only a single ``parameter_applier`` and ``calculator_factory``.

.. code-block:: python

    from chemfit.multi_energy_objective_function import construct_multi_energy_objective_function

    # ... assume the same definitions for `MyCalculatorFactory` and `MyCalculatorParameterApplier` from above

    objective_function = construct_multi_energy_objective_function(
            calc_factory = MyCalculatorFactory(some_parameter=2), 
            param_applier = MyCalculatorParameterApplier(),
            path_to_reference_configuration_list = ["atoms_1.xyz", "atoms_2.xyz"],
            tag_list = ["my_tag_1", "my_tag_2"],
            reference_energy_list = [1.0, 2.0]
        )

    # Evaluate the objective function at x=2.0 and y=1.0
    val = objective_function( {"x" : 2.0, "y": 1.0} )