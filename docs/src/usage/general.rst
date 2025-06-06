#######################
Overview
#######################


Performing a fit with this package is generally involves three steps:

#. Create an *objective function*. This may be any function or callable object, where the ``__call__`` operator has the signature ``f(params : dict[str, float]) -> float``.

#. Create an instance of a :py:class:`scme_fitting.fitter.Fitter` object, giving the objective function in the initializer

#. Use the :py:class:`scme_fitting.fitter.Fitter.fit_scipy` or :py:class:`scme_fitting.fitter.Fitter.fit_nevergrad` methods to minimize the objective function


A minimal example
#######################

In this simple example we define a quadratic objective function with two degrees of freedom and minimize it.

.. code-block:: python

    from scme_fitting import Fitter

    def ob(params):
        return 2.0 * (params["x"] - 2) ** 2 + 3.0 * (params["y"] + 1) ** 2

    fitter = Fitter(objective_function=ob)

    initial_params = dict(x=0.0, y=0.0)
    optimal_params = fitter.fit_scipy(initial_parameters=initial_params)

    # We expect x=2.0 and y=-1.0
    print(f"{optimal_params = }")


The builtin objective functions
##############################################

The core utility provided by this package are the builtin objective functions. Currently, there are three different provided objective functions:

:py:class:`scme_fitting.scme_objective_function.EnergyObjectiveFunction`:
    An objective function based on the squared energy difference, given by ``weight * (e_cur - e_target)**2`` 

:py:class:`scme_fitting.scme_objective_function.DimerDistanceObjectiveFunction`:
    An objective function based on the squared Oxygen-Oxygen bond length difference (after relaxation) in a dimer of H2O molecules. Computed as ``weight * (rOO_cur - rOO_target)**2`` 

:py:class:`scme_fitting.combined_objective_function.CombinedObjectiveFunction`:
    This is a utility class, used to add several objective functions (optionally weighing) them into a single combined objective function.

