.. ChemFit documentation master file, created by
   sphinx-quickstart on Thu Jun  5 11:32:11 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

################
ChemFit
################

**ChemFit** is a framework for fitting parameters of models used in computational chemistry, molecular dynamics, and materials science.

It provides composable building blocks for constructing objective functions from many independent terms, computing intermediate quantities using simulation workflows, and optimizing model parameters. A small set of core abstractions makes it straightforward to implement custom objective functions and to parallelize their evaluation across objective terms and/or trial parameters.

Out of the box, ChemFit includes integrations for the calculators defined in the Atomic Simulation Environment (ASE) as well as external simulation pipelines.


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

A typical ChemFit workflow is:

1. compute quantities from parameters;
2. attach a loss;
3. combine independent objective terms;
4. fit the parameters.

The top-level API expresses this workflow directly: use
:func:`chemfit.quantity() <chemfit.wrap_funcs.quantity>`, attach a loss with
``with_loss()``, combine terms with
:func:`chemfit.combine() <chemfit.api.combine>`, and optimize them with
:func:`chemfit.fit_nevergrad() <chemfit.api.fit_nevergrad>`.

Underneath, these helpers operate on ChemFit's
:class:`~chemfit.abstract_objective_function.QuantityComputer`,
:class:`~chemfit.abstract_objective_function.ObjectiveFunctor`, and
:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`
abstractions. A quantity computer maps a parameter dictionary to intermediate
quantities, and a loss maps those quantities to a scalar objective value.

In practical workflows, the quantity computation may be performed by an
ASE calculator, an external simulation pipeline, or custom Python code.

The following minimal example simply defines a loss function

.. math::

    L(\text{params}) = (x^2 + y^2 - 2)^2,

where :math:`x^2` and :math:`y^2` are intermediate quantities:

.. testcode::

    import chemfit

    @chemfit.quantity()
    def simulate(params):
        return {"x2": params["x"] ** 2, "y2": params["y"] ** 2}

    def square_deviation(q, target):
        return (q["x2"] + q["y2"] - target)**2

    ob = simulate.with_loss(square_deviation, target=2.0)

    print(ob({"x": 1.0, "y": 2.0})) # <-- 9.0

.. testoutput::
    :hide:

    9.0

The quantity computer is defined via a simple Python function, mapping a :class:`dict` of parameters to a :class:`dict` of quantities.
The :func:`~chemfit.wrap_funcs.quantity` decorator turns this function into a
:class:`~chemfit.wrap_funcs.WrappedQuantityComputer` instance.

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
ChemFit provides :func:`~chemfit.api.combine` to combine multiple objective terms into a single loss.

In the next example, we first define a parametrized loss term

.. math::

    T(\text{params},f,\text{target}) = (f x^2 + f y^2 - \text{target})^2,

where :math:`f` is an external parameter and then we combine them into an overall loss

.. math::

    L(\text{params}) = T(\text{params},1,1) + T(\text{params},2,2).

.. testcode::

    import chemfit

    @chemfit.quantity()
    def computer(params, f):
        return {"fx2": f * params["x"] ** 2, "fy2": f * params["y"] ** 2}

    def loss(q, target):
        return (q["fx2"] + q["fy2"] - target) ** 2

    combined = chemfit.combine(
        computer.bind(f=1.0).with_loss(loss, target=1),
        computer.bind(f=2.0).with_loss(loss, target=2)
    )
    print(combined({"x": 1.0, "y": 2.0})) # <-- 80.0

.. testoutput::
    :hide:

    80.0

``chemfit.combine()`` returns a
:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`.
That object provides reducers, aggregators, exception handling, child
contexts, mutation, and scheduler-backed execution; see
:ref:`combined_objective_functions` when you need those controls.

====================
Optimizing
====================

Optimize the combined objective through the same top-level interface:

.. code-block:: python

    result = chemfit.fit_nevergrad(
        combined,
        initial={"x": 1.0, "y": 2.0},
        bounds={"x": (-2.0, 2.0), "y": (-2.0, 2.0)},
        budget=100,
        batch_size=4,
    )

    print(result.best_parameters)
    print(result.best_loss)

``chemfit.fit_nevergrad()`` returns a :class:`~chemfit.api.FitResult`.
Use :class:`~chemfit.fitter.Fitter` directly for SciPy, callbacks, custom
optimizer loops, or manual lifecycle control. Use the scheduler APIs for
direct control over threads, processes, or MPI. The :ref:`common_workflows`,
:ref:`fitter`, and :ref:`parallel_execution` pages continue from here.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   src/installation
   src/usage/public_api.rst

.. toctree::
   :maxdepth: 2
   :caption: Using ChemFit

   src/usage/ase_objective_function_api.rst
   src/usage/external_computer.rst
   src/usage/combined_objective_function.rst
   src/usage/fitter.rst
   src/usage/parallel_execution.rst

.. toctree::
   :maxdepth: 2
   :caption: Understanding and advanced usage

   src/usage/concepts.rst
   src/usage/writing_quantity_computers.rst
   src/usage/objective_hooks.rst
   src/usage/mpi.rst
   src/development/development.rst

.. toctree::
   :maxdepth: 3
   :caption: API reference

   src/api/modules
