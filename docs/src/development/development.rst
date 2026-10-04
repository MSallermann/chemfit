#######################
Development
#######################

This page contains some pointers for people looking to develop the ChemFit package.


=============================
Installation
=============================

To ease development, check out the git repository and perform an editable install

.. code-block:: bash

    git clone git@github.com:MSallermann/chemfit.git
    pip install -e chemfit


=============================
Running the unit tests
=============================

Make sure ``pytest`` is installed. To run the unit tests, run the following from the repository root

.. code-block:: bash

    pytest tests


=============================
Building the documentation
=============================

Make sure to install the Sphinx dependencies. You can do so by specifying the ``[build_docs]`` tag in the pip install (see below), but *don't forget the quotation marks*.

.. code-block:: bash

    pip install -e ".[build_docs]"


Then you can build the documentation by invoking the following from the repository root

.. code-block:: bash

    sphinx-apidoc -o ./docs/src/api ./src/chemfit -f --remove-old --separate
    sphinx-build -M html ./docs ./docs/build

Use ``sphinx-autobuild -M html ./docs ./docs/build`` instead when you want a
live-reloading development server. The repository's ``build_docs.sh`` script
starts that workflow after regenerating the API pages.
