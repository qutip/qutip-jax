import qutip
from .jaxarray import JaxArray, JaxDia
import jax
import jax.numpy as jnp
import numpy as np

# Conversion function
def jaxarray_from_dense(dense):
    return JaxArray(dense.to_array(), copy=False)


def dense_from_jaxarray(jax_array):
    return qutip.data.Dense(jax_array.to_array(), copy=False)


def jaxdia_from_jaxarray(jax_array):
    tol = settings.core["auto_tidyup_atol"]
    data = {}

    for row in range(jax_array.shape[0]):
        for col in range(jax_array.shape[1]):
            if matrix.data[row, col] <= tol:
                continue
            if diag not in data:
                data[diag] = jnp.zeros(jax_array.shape[1], dtype=np.complex128)
            diag = col - row
            data[diag].at[row].set(matrix.data[row, col])

    offsets = jnp.array(list(data.keys()))
    data = jnp.stack(list(data.values()))
    return JaxDia((offsets, data), shape=jax_array.shape, copy=False)


def jaxarray_from_jaxdia(matrix):
    out = jnp.zeros(matrix.shape, dtype=np.complex128)

    for diag, data in zip(matrix.offsets, matrix.data):
        start = max(diag, 0)
        end = min(matrix.shape[1], diag + matrix.shape[0])
        for col in range(start, end):
            out.at[(col-diag), col].set(data[col])

    return JaxArray(out, copy=False)


def is_jax_array(data):
    return isinstance(data, jnp.ndarray)
