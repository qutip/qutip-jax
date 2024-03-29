import qutip
from .jaxarray import JaxArray
import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["is_jax_array", "jax_from_dense", "dense_from_jax"]

# Conversion function
def jax_from_dense(dense):
    return JaxArray(dense.to_array(), copy=False)


def dense_from_jax(jax_array):
    return qutip.data.Dense(jax_array.to_array(), copy=False)


def is_jax_array(data):
    return isinstance(data, jax.Array)
