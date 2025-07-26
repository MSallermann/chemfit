#######################
Overview
#######################

The ChemFit package helps you tune the free parameters of the "SCME 2.0" code so that ASE-computed potential energies match ab-initio reference values *as close as possible*.
**Beyond fitting just the SCME parameters**, it can drive the parameters of any ASE calculator via the same callback machinery.

.. _overview_objective_functions:

**********************
Objective functions
**********************

An objective function (for the sake of this package) is any function (or object implementing a ``__call__`` operator) which supports the call signature (``A(params : dict) -> float``).
For the objective functions built into this package see: :ref:`predefined_objective_functions`.

See the following code for two example implementations of the ``ob(x) = 2 * x^2`` objective function, first as a regular function and second as a functor object:

.. code-block:: python

    # 'A' is an objective function [ob(x) = 2.0 * x^2]
    def A(params : dict) -> float:
        return 2.0 * params["x"]**2

    class BFunctor:
        def __init__(self, f : float):
            self.f = f

        def __call__(self, params : dict) -> float:
            return self.f * params["x"]**2

    # `B` is also an objective function [ob(x) = 2.0 * x^2]
    B = BFunctor(f=2.0)

For more complex tasks the "functor" pattern is generally favoured (this is also how the objective functions in :py:mod:`chemfit.ase_objective_function` are implemented).

In principle such objective functions may be optimized with any suitable method. For convenience `ChemFit` provides the :py:class:`chemfit.fitter.Fitter` class, which wraps two different backends.

These two backends are

#. ``scipy.minimize``, used via :py:meth:`chemfit.fitter.Fitter.fit_scipy`
#. ``nevergrad``, used via :py:meth:`chemfit.fitter.Fitter.fit_nevergrad`


A minimal ``Fitter`` example
********************************

Performing a fit with this package generally involves three steps:

#. Create an *objective function*. This may be any function or callable object, where the ``__call__`` operator has the signature ``f(params : dict) -> float``.

#. Create an instance of a :py:class:`chemfit.fitter.Fitter` object, giving the objective function in the initializer

#. Use the :py:class:`chemfit.fitter.Fitter.fit_scipy` or :py:class:`chemfit.fitter.Fitter.fit_nevergrad` methods to minimize the objective function


In this simple example we define a quadratic objective function with two degrees of freedom and minimize it.

.. code-block:: python

    from chemfit import Fitter

    initial_params = dict(x=0.0, y=0.0)

    def ob(params):
        return 2.0 * (params["x"] - 2) ** 2 + 3.0 * (params["y"] + 1) ** 2

    fitter = Fitter(objective_function=ob, initial_params=initial_params)

    optimal_params = fitter.fit_scipy()

    # We expect x=2.0 and y=-1.0
    print(f"{optimal_params = }")


ASE objective functions
********************************

The core functionality is provided by the :py:class:`chemfit.ase_objective_function`. This is an abstract base class, where deriving classes have to implement the ``__call__(params : dict) -> float`` operator. This operator should compute an objective function value for one reference configuration, which in a later fit would then be minimized.

Some implementations of this objective function are provided (see the classes in :py:mod:`chemfit.ase_objective_function`).

The key point, which makes these classes flexible, is that they take two factory functions (or functors) in their initializer, which tell them how to construct a calculator object and how to apply a certain parametrization to it.

.. note::

    As long as implementations of :py:class:`chemfit.ase_objective_function.CalculatorFactory` and :py:class:`chemfit.ase_objective_function.ParameterApplier` are provided, the objective functions work with *any* ASE calculator.


SCME factory functions
***************************

Ready to use implementations of :py:class:`chemfit.ase_objective_function.CalculatorFactory` and :py:class:`chemfit.ase_objective_function.ParameterApplier` are provided in the :py:mod:`chemfit.scme_factories` module.

These should cover most use cases of fitting parameters in the SCME, but they can *of course* be extended to fit any individual task.
