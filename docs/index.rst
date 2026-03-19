.. ChemFit documentation master file, created by
   sphinx-quickstart on Thu Jun  5 11:32:11 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

################
ChemFit
################

**ChemFit** is a framework for fitting parameters of models used in computational chemistry, molecular dynamics, and materials science.

It provides composable building blocks for constructing objective functions from many independent terms, computing intermediate quantities using simulation workflows, and optimizing model parameters. A small set of core abstractions makes it straightforward to implement custom objective functions and to parallelize their evaluation across objective terms and/or trial parameters.

Out of the box, ChemFit includes integrations for the calculators defined in the Atomic Simulation Environment (ASE) as well as file-based simulation pipelines.


**Highlights**:

- **Designed for atomistic simulations:** Build objective functions from quantities computed by ASE calculators or external simulation codes.
- **Composable objectives:** Assemble complex fitting targets from many independent objective terms.
- **Parallel by construction:** Evaluate objective terms concurrently across processes or MPI ranks without changing the objective definition.
- **Extensible architecture:** Implement custom objective functions and simulation interfaces using ChemFit's core abstractions.

-------------------------

.. _quickstart:

*************
Quickstart
*************

In ChemFit, an objective function is typically built from two layers:

1. A :class:`~chemfit.abstract_objective_function.QuantityComputer`
   computes intermediate quantities from a parameter dictionary.
2. A loss function maps those quantities to a scalar loss.

Multiple such objective terms can then be combined using
:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`
and optimized with :class:`~chemfit.fitter.Fitter`.

In practical workflows, the quantity computation may be performed by an
ASE calculator, a file-based simulation pipeline, or custom Python code.

The following minimal example simply defines a loss function

.. math::

    L(\text{params}) = (x^2 + y^2 - 2)^2,

where :math:`x^2` and :math:`y^2` are intermediate quantities:

.. testcode::

    from chemfit.wrap_funcs import to_quantity_computer

    @to_quantity_computer()
    def computer(params):
        return {"x2": params["x"] ** 2, "y2": params["y"] ** 2}

    def square_deviation(q, target):
        return (q["x2"] + q["y2"] - target)**2

    ob = computer.with_loss(square_deviation, target=2)

    PARAMS = {"x": 1.0, "y": 2.0}
    print(ob(PARAMS)) # <-- 9.0

.. testoutput::
    :hide:

    9.0

The quantity computer is defined via a simple Python function, mapping a :class:`dict` of parameters to a :class:`dict` of quantities.
The :func:`to_quantity_computer` decorator turns this function into a :class:`~chemfit.wrap_funcs.WrappedQuantityComputer` instance.

The :meth:`~chemfit.abstract_objective_function.QuantityComputer.with_loss`
method combines a quantity computer with a loss function to form a complete objective.

.. note::

    The wrapped :class:`~chemfit.wrap_funcs.WrappedQuantityComputer` differs from the plain
    function in that it can optionally accept an evaluation context
    (:class:`~chemfit.abstract_objective_function.EvaluateContext`).

    This context is used internally by ChemFit to enable parallel evaluation
    and to collect metadata during execution. In most cases, you do not need
    to interact with it directly.

    For more details, see :ref:`concepts`.

====================
Combining functions
====================

Often an objective consists of many independent contributions.
ChemFit provides :class:`~chemfit.combined_objective_function.CombinedObjectiveFunction` to combine multiple objective terms into a single loss.

In the next example, we first define a parametrized loss term

.. math::

    T(\text{params},f,\text{target}) = (f x^2 + f y^2 - \text{target})^2,

where :math:`f` is an external parameter and then we combine them into an overall loss

.. math::

    L(\text{params}) = T(\text{params},1,1) + T(\text{params},2,2).

.. testcode::

    from chemfit.wrap_funcs import to_quantity_computer
    from chemfit.combined_objective_function import CombinedObjectiveFunction

    @to_quantity_computer()
    def computer(params, f):
        return {"fx2": f * params["x"] ** 2, "fy2": f * params["y"] ** 2}

    def loss(q, target):
        return (q["fx2"] + q["fy2"] - target) ** 2

    terms = [
        computer.bind(f=1).with_loss(loss, target=1),
        computer.bind(f=2).with_loss(loss, target=2)
    ]

    PARAMS = {"x": 1, "y": 2}
    combined = CombinedObjectiveFunction(terms)
    print(combined(PARAMS)) # <-- 80.0

.. testoutput::
    :hide:

    80.0

The :class:`~chemfit.combined_objective_function.CombinedObjectiveFunction` can be customized in many ways.
Details can be found in the dedicated :ref:`combined_objective_functions` page.

The evaluation of the terms of a :class:`~chemfit.combined_objective_function.CombinedObjectiveFunction` can be parallelized in two alternate ways:

1. Executors which implement the :class:`~chemfit.abstract_objective_function.ExecutorLike` interface.
2. By launching multiple processes and using the message passing interface (MPI). See :ref:`mpi` for details.

The following demonstrates the use of the "ExecutorLike" approach

.. testcode::

    from chemfit.combined_objective_function import CombinedObjectiveFunction
    from chemfit.executor_wrapper_cob import ExecutorWrapperCOB
    from concurrent.futures import ThreadPoolExecutor

    cob = CombinedObjectiveFunction( terms ) # define the combined objective function
    cob_parallel = ExecutorWrapperCOB(cob, executor=ThreadPoolExecutor(4))
    print(cob_parallel(PARAMS)) # <-- evaluation of the wrapper parallelizes over the terms

.. testoutput::
    :hide:

    80.0

====================
Optimizing
====================

In principle, objective functions defined with ChemFit can be optimized in any way you like.
For convenience a :class:`~chemfit.fitter.Fitter` class is provided.

It can be used as follows

.. code-block:: python

    from chemfit.fitter import Fitter

    fitter = Fitter(objective_function, initial_params)

    # fit with nevergrad
    optimal_params = fitter.fit_nevergrad(100)

    # fit with scipy
    optimal_params = fitter.fit_scipy()

The :class:`~chemfit.fitter.Fitter` can be configured in many ways, for details refer to :ref:`fitter`.

*************
Contents
*************

.. toctree::
   :maxdepth: 2

   src/installation
   src/usage/concepts.rst
   src/usage/overview.rst
   src/usage/abstract_interface.rst
   src/usage/fitter.rst
   src/usage/ase_objective_function_api.rst
   src/usage/combined_objective_function.rst
   src/usage/mpi.rst
   src/usage/example_scme.rst
   src/usage/text_file_based_computer.rst
   src/usage/aynchronous_execution.rst
   src/development/development.rst
   src/api/modules
