##################
Fitter
##################

The `Fitter` class is a wrapper around optimization backends and minimizes objective functions.

Currently it supports the following backends:

    1. `Nevergrad <https://github.com/facebookresearch/nevergrad>`_ 
    2. `Scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_ 

What can be optimized with Fitter
##################################

Any python function with the signature:

.. code-block:: python

    def f(params : dict) -> float:
        # implementation
        ...

Params dict format 
=====================

The ``params`` can be any dictionary, which contains only **float values** or sub-dictionaries, which may in turn contain only float values or sub-dictionaries and so on ...

In other words if the params dict is visualized as a tree (with key-value pairs being branches), all leaf nodes must be float value.

As an example, this is allowed

.. code-block:: python

    # Allowed
    params = {
                "foo" : {
                    "bar" : {
                        "a" : 1.0
                    }
                    "b" : 2.0
                }
            }

This is **not** allowed

.. code-block:: python

    # Not allowed
    params = {
                "foo" : 2.0,
                "bar" : [1.0, 2.0] # <-- Error: "bar" is not a float value
            }

Bounds
=====================

Bounds for parameters may be specified, alongside the initial parameters, in the constructor of the fitter class, e.g.

.. code-block:: python

    fitter = Fitter(
        objective_function=obj_func,
        initial_params=initial_params,
        bounds=bounds
    )

The bounds must be in the same path as the parameters they are supposed to apply to.
The same rules as for the params dict apply, **except** that each leaf node needs to be a tuple of two floats.

.. note::

    It is allowed to **not** specify bounds for a parameter.

For example:

.. code-block:: python

    initial_params = {"foo": {"x" : 0.0}, "y": 0.0}

    # Restrict params['foo']['x'] to the interval (0.0, 1.5)
    # (leaves params['y'] unrestricted)
    bounds = {"params": { "x" : (0.0, 1.5)}
