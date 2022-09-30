import jax
import jax.numpy as jnp
from jax import jit

import numpy as np

from qutip_jax.jaxarray import JaxArray


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
