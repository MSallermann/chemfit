#######################
Installation
#######################

====================================
Installing the SCMEFitting package
====================================

The installation of the SCMEFitting package is simple. The dependencies (**except SCME 2.0**) are installed automatically.

Execute this command in your terminal (and activate your preferred virtual environment before)

.. code-block:: bash

    pip install git+https://github.com/MSallermann/SCMEFitting.git

Alternatively, you can of course check out the repository manually and perform a local pip install.

.. warning::

    The pip install **does not** install the SCME 2.0 code for you.

.. note::
    Even **without the SCME code**, you can still use everything this package provides, except the :py:mod:`scme_fitting.scme_objective_function` module.

=============================
Installing the SCME 2.0 code
=============================

To use this package with the SCME code, you need a working installation of the SCME 2.0 code, which is presently hosted in a private GitLab repositor (at some point it will become public).

.. note::

    In practice, this means that the following import must succeed in the environment in which SCMEFitting is installed

    .. code-block:: python

        import pyscme # must succeed in order to use scme_fitting.scme_objective_function

    Further, a recent enough version must be installed.
