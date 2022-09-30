import qutip
from .jaxarray import JaxArray
import jax
import jax.numpy as jnp
import numpy as np

# Conversion function
def jax_from_dense(dense):
    return qutip.data.Dense(dense.to_array(), copy=False)

def dense_from_jax(jax_array):
    return qutip.data.Dense(np.array(jax_array), copy=False)


def is_jax_array(data):
    return (
        isinstance(data, jax.interpreters.xla.DeviceArray)
        and data.dtype is jnp.complex128
    )
