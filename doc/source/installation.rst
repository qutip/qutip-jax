************
Installation
************

.. _quickstart:

Qutip-Jax needs the development version of Qutip and installation is only possible from source

.. code-block:: bash

    pip install qutip --pre
    pip install git+https://github.com/qutip/qutip-jax.git


.. _prerequisites:

Prerequisites
=============
This package use the development version of QuTiP, Jax, equinox and diffrax.

The following to packages are used for plotting and testing:

.. code-block:: bash

    matplotlib pytest

In addition

.. code-block:: bash

    sphinx numpydoc sphinx sphinx_rtd_theme

are used to build and test the documentation.
