import jax
import jax.numpy as jnp
from jax import jit

import numpy as np
from numpy.testing import assert_array_almost_equal


import qutip_jax
from qutip_jax.jaxarray import JaxArray

import qutip


def test_create():
    """Tests creation of JaxArrays from NumPy and JAX-Numpy arrays"""
    assert isinstance(JaxArray(jnp.linspace(0, 3, 11)), JaxArray)
    assert isinstance(JaxArray(np.linspace(0, 3, 11)), JaxArray)


def test_jit():
    """Tests JIT of JaxArray methods"""
    arr = JaxArray(jnp.linspace(0, 3, 11))

    # Some function of JaxArray that we would like to JIT.
    @jit
    def func(arr):
        return arr.trace()

    tr = func(arr)
    assert isinstance(tr, jax.interpreters.xla.DeviceArray)


def test_convert():
    """Tests if the conversions from Qobj to JaxArray work"""
    ones = jnp.ones((3, 3))
    qobj = qutip.Qobj(ones)
    prod = qobj * jnp.array([0.5])
    assert_array_almost_equal(prod.data.to_array(), ones * jnp.array([0.5]))

    sx = qutip.sigmax()
    assert(isinstance(sx.data, qutip.core.data.CSR))
    assert(isinstance(sx.to('jax').data, JaxArray))

