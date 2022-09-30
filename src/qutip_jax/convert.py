import qutip
from .jaxarray import JaxArray
import jax
import jax.numpy as jnp
import numpy as np

# Conversion function
def jax_array_from_dense(dense):
    return JaxArray(dense.data.to_array())


def jax_array_to_dense(jax_array):
    return qutip.data.Dense(np.array(jax_array), copy=False)


def is_jax_array(data):
    return (
        isinstance(data, jax.interpreters.xla.DeviceArray)
        and data.dtype is jnp.complex128
    )
