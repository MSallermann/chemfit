#######################
Installation
#######################

====================================
Installing the ChemFit package
====================================

It's on PyPI, so just do

.. code-block:: bash

    pip install chemfit

For the additional MPI dependency (see :ref:`mpi`) use

.. code-block:: bash

    pip install chemfit[mpi]

To install the latest development version, instead, use

.. code-block:: bash

    pip install git+https://github.com/MSallermann/ChemFit.git

.. warning::
    The pip install **does not** install the SCME 2.0 code for you.

.. note::
    Even **without the SCME code**, you can still use everything this package provides, except the :py:mod:`~chemfit.scme_factories` module.

=============================
Installing the SCME 2.0 code
=============================

To use this package with the SCME code, you need a working installation of the SCME 2.0 code, which is presently hosted in a private GitLab repository (at some point it will become public).

.. note::
    In practice, this means that the following import must succeed in the environment in which ChemFit is installed

    .. code-block:: python

        import pyscme # must succeed in order to use chemfit.scme_factories

    Further, a recent enough version must be installed.
