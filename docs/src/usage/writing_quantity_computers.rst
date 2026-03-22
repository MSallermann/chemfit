.. _writing_quantity_computers:

#############################
Writing Quantity Computers
#############################

This page contains practical guidance for writing your own quantity computers from scratch.

********************
Before we start...
********************

Before we start, let us clearly state that "quantity computers" are merely a useful convention.
Nothing prevents you from using ChemFit without them. That being said, you should probably use them.

A recommended first step is to check if you can use one of the built-in ways:

1. If you already have a pure python function implementing your computation, have a look at the :py:func:`~chemfit.wrap_funcs.to_quantity_computer` decorator, which also features in the :ref:`quickstart` examples.
2. If you are using an external simulation tool, like LAMMPS for example, have a look at the :py:class:`~chemfit.file_based_computer.FileBasedQuantityComputer` and its corresponding doc page: :ref:`file_based`.
3. If you are using ASE, try the :py:class:`~chemfit.ase_objective_function.SinglePointASEComputer` or :py:class:`~chemfit.ase_objective_function.MinimizationASEComputer` described in :ref:`ase_objective_function_api`.

If none of the built-in computers are to your taste, think about sub-classing them.

********************
Let's cook
********************

For a completely fresh QuantityComputer, derive from the :py:class:`~chemfit.abstract_objective_function.QuantityComputer` base class and implement the :py:meth:`~chemfit.abstract_objective_function.QuantityComputer._compute` method. That's it.

The ``compute_`` method should accept exactly two arguments: A dictionary of parameters of type :py:class:`dict[str,Any]` and an :py:class:`~chemfit.abstract_objective_function.EvaluateContext`.
It should return the dictionary of quantities.

This is probably a point at which we should familiarize ourselves with the...

**Golden Rule:**
    DO NOT MODIFY GLOBAL STATE FROM WITHIN THE COMPUTE METHOD. If you violate this rule, parallel evaluation of your quantity computer can be undefined. It does not *have* to be, but for everyone's sake let's assume it **will** be.

    Importantly, the golden rule applies to instance variables of the computer itself as well.

    Let's illustrate what **not** to do:

    .. code-block:: python

        class GoldenRuleViolator(QuantityComputer)
            # ...
            def _compute(params, ctx):
                self.bad = params["x"] # <-- bad mojo
                # ...
                return {"mojo" : self.bad}

    Now what happens if you call the same instance of ``GoldenRuleViolator`` in parallel? That's right! Bad things. The reason is that the value of ``self.bad`` could be overwritten by another thread in the middle of the compute function, which would make your ``params`` and the returned quantities mismatched.

    No you might say: "Why would I ever do something so stupid?". Let me just say that you'd be surprised how easy it is to accidentally violate the **Golden Rule**.


Therefore, if you have anything to communicate with the outside world, there are two options

1. Put it in the quantities dict and return it
2. Write to ``ctx.meta``

Let's fix the ``GoldenRuleViolator``:

.. code-block:: python

    class GoodCitizen(QuantityComputer)
        # ...
        def _compute(params, ctx):
            bad = params["x"]
            ctx.meta.bad = bad # <-- no problemo
            # ...
            return {"mojo" : bad}

Now there is no problem. All we ever do is write to ``bad`` which is local to the current function evaluation or to ``ctx.meta.bad`` which explicitly prevents any kind of race conditions.

******************************************
Calling computers from within computers
******************************************

.. note::

    This section is for fairly advanced use and, probably, most relevant if you are looking to implement your own execution wrapper for the :py:class:`~chemfit.combined_objective_function.CombinedObjectiveFunction`, besides the built-in MPI and executor wrappers.

If we want to make calls to other computers from our custom computer, the recommended approach is to make use of the child context system to supply fresh contexts to the inner computers.

Here is a simple demonstration of the idea: We have an outer computer, which accepts a parent :py:class:`~chemfit.abstract_objective_function.EvaluateContext` and then later on splits of two child contexts using the :py:meth:`~chemfit.abstract_objective_function.EvaluateContext.child_contexts` context manager.

.. code-block:: python

    class OuterComputer(QuantityComputer):

        def _compute(params, ctx):
            # ...
            with ctx.child_contexts(2) as child_contexts:
                q1 = inner_computer1(params, child_contexts[0])
                q2 = inner_computer2(params, child_contexts[1])
            # ...

The benefit of this approach is two-fold

1. We get full meta-data provenance. All of the child meta data can be found in ``ctx.meta["children"]``.
2. Since the inner computers have their own context they can also be evaluated in parallel ... although the example above does not make use of this.

.. note::

    For parallel evaluation with an executor, use the :py:func:`~chemfit.executor_utils.map_with_context` function.
    Differently from the regular executor ``map`` function, it correctly handles the ``ctx`` fields even if execution happens in different processes.
