************
Installation
************

.. _quickstart:

Qutip-Jax can be installed with pip:

.. code-block:: bash

    pip install qutip>=5.1.0 qutip-jax

However qutip-jax is currently being developed and a lot of feature are only available when installed from source:

.. code-block:: bash

    pip install git+https://github.com/qutip/qutip.git
    pip install git+https://github.com/qutip/qutip-jax.git

The latest changes are also avaiable as pre release on TestPyPI:

    pip install --pre --index-url https://test.pypi.org/simple/ qutip-jax

.. _prerequisites:

Prerequisites
=============
This package uses QuTiP, Jax, equinox and diffrax.

We use ``pytest`` for testing.

See doc/requirementes.txt for a list of package needed to build the documentation.
