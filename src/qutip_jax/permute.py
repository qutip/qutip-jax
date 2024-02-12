import qutip
from .jaxarray import JaxArray
from jax import jit
import numpy as np

__all__ = []


def indices_jaxarray(matrix, row_perm=None, col_perm=None):
    if row_perm is None and col_perm is None:
        return matrix.copy()
    data = matrix._jxa
    if row_perm is not None:
        data = data[np.argsort(row_perm), :]
    if col_perm is not None:
        data = data[:, np.argsort(col_perm)]
    return JaxArray(data)


def dimensions_jaxarray(matrix, dimensions, order):
    dims = np.array(dimensions, dtype=int)
    order = np.array(order, dtype=int)
    N = len(dimensions)
    size = matrix.shape[0]
    if size == 1:
        size = matrix.shape[1]

    if np.any(0 >= dims):
        raise ValueError("invalid dimensions")
    if np.any(0 > order) or np.any(order >= N):
        raise ValueError(f"invalid order, {order}")
    if len(np.unique(order)) != N:
        raise ValueError("duplicate order element")
    if len(order) != N:
        raise ValueError("invalid order: wrong number of elements")
    if np.prod(dims) != size:
        raise ValueError("dimensions does not match the shape")

    cumprod = np.ones(N, dtype=int)
    for i in range(N - 1):
        cumprod[i+1] = cumprod[i] * dims[order][-i-1]
    cumprod = cumprod[::-1][np.argsort(order)]
    idx = np.arange(size)
    permute = np.zeros(size, dtype=int)
    for dim, step in zip(dims[::-1], cumprod[::-1]):
        permute += step * (idx % dim)
        idx //= dim
    row_perm, col_perm = None, None
    if matrix.shape[0] == size:
        row_perm = permute
    if matrix.shape[1] == size:
        col_perm = permute
    return indices_jaxarray(matrix, row_perm, col_perm)


qutip.data.permute.indices.add_specialisations([
    (JaxArray, JaxArray, indices_jaxarray),
])


qutip.data.permute.dimensions.add_specialisations([
    (JaxArray, JaxArray, dimensions_jaxarray),
])
