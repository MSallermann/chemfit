#######################
Overview
#######################

The SCMEFitting package helps you tune the free parameters of the "SCME 2.0" code so that ASE-computed potential energies match ab-initio reference values *as close as possible*.
**Beyond fitting just the SCME parameters**, it can drive the parameters of any ASE calculator via the same callback machinery.

Objective functions
#######################

An objective function (for the sake of this package) is any function (or object implementing a ``__call__`` operator) which supports the call signature (``A(params : dict[str,float]) -> float``).

See the following code for two example implementations of the ``ob(x) = 2 * x^2`` objective function, first as a regular function and second as a functor object:

.. code-block:: python

    # 'A' is an objective function [ob(x) = 2.0 * x^2]
    def A(params : dict[str,float]) -> float:
        return 2.0 * params["x"]**2

    class BFunctor:
        def __init__(self, f : float):
            self.f = f

        def __call__(self, params : dict[str,float]) -> float:
            return self.f * params["x"]**2

    # `B` is also an objective function [ob(x) = 2.0 * x^2]
    B = BFunctor(f=2.0)

For more complex tasks the "functor" pattern is generally favoured (this is also how the objective functions in :py:mod:`scme_fitting.ase_objective_function` are implemented).

In principle such objective functions may be optimized with any suitable method. For convenience `SCMEFitting` provides the :py:class:`scme_fitting.fitter.Fitter` class, which wraps two different backends.

These two backends are

#. ``scipy.minimize``, used via :py:meth:`scme_fitting.fitter.Fitter.fit_scipy`
#. ``nevergrad``, used via :py:meth:`scme_fitting.fitter.Fitter.fit_nevergrad`


A minimal ``Fitter`` example
#############################

Performing a fit with this package generally involves three steps:

#. Create an *objective function*. This may be any function or callable object, where the ``__call__`` operator has the signature ``f(params : dict[str, float]) -> float``.

#. Create an instance of a :py:class:`scme_fitting.fitter.Fitter` object, giving the objective function in the initializer

#. Use the :py:class:`scme_fitting.fitter.Fitter.fit_scipy` or :py:class:`scme_fitting.fitter.Fitter.fit_nevergrad` methods to minimize the objective function


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


ASE objective functions
########################

The core functionality is provided by the :py:class:`scme_fitting.ase_objective_function`. This is an abstract base class, where deriving classes have to implement the ``__call__(params : dict[str,float]) -> float`` operator. This operator should compute an objective function value for one reference configuration, which in a later fit would then be minimized.

Some implementations of this objective function are provided (see the classes in :py:mod:`scme_fitting.ase_objective_function`).

The key point, which makes these classes flexible, is that they take two factory functions (or functors) in their initializer, which tell them how to construct a calculator object and how to apply a certain parametrization to it.

.. note::

    As long as implementations of :py:class:`scme_fitting.ase_objective_function.CalculatorFactory` and :py:class:`scme_fitting.ase_objective_function.ParameterApplier` are provided, the objective functions work with *any* ASE calculator.


SCME factory functions
########################

Ready to use implementations of :py:class:`scme_fitting.ase_objective_function.CalculatorFactory` and :py:class:`scme_fitting.ase_objective_function.ParameterApplier` are provided in the :py:mod:`scme_fitting.scme_objective_function` module.

These should cover most use cases of fitting parameters in the SCME, but they can *of course* be extended to fit any individual task.
