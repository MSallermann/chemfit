#######################
Installation
#######################

========================
Installing the SCME 2.0 code
========================

**Important:** To use this package you need a working installation of the SCME 2.0 code.

In practice, this means that the following import must succeed in the environment in which SCMEFitting is installed

.. code-block:: python

    import pyscme # must succeed in order to use SCMEFitting

Further, a recent enough version must be installed.

====================================
Installing the SCMEFitting package
====================================

The installation of the SCMEFitting package is simple. The dependencies (**except SCME 2.0**) are installed automatically.

-----------------
Quick
-----------------
Execute this command in your terminal (and activate your preferred virtual environment before)

.. code-block:: bash

    pip install git+https://github.com/MSallermann/SCMEFitting.git



-----------------
For development
-----------------
Check out the git repository and perform an editable install

.. code-block:: bash

    git clone git@github.com:MSallermann/SCMEFitting.git
    pip install -e SCMEFitting