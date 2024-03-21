import jax.numpy as jnp
from .jaxarray import JaxArray
from .jaxdia import JaxDia, clean_dia
import qutip
from jax import jit
from functools import partial
import numpy as np


__all__ = [
    "isherm_jaxarray",
    "isdiag_jaxarray",
    "iszero_jaxarray",
    "isherm_jaxdia",
    "isdiag_jaxdia",
    "iszero_jaxdia",
]


@partial(jit, static_argnames=["tol"])
def _isherm(matrix, tol):
    return jnp.allclose(matrix, matrix.T.conj(), atol=tol, rtol=0)


# jitting this makes it 100x slower for nonsquare.
# splitting it like this seems ideal.
def isherm_jaxarray(matrix, tol=None):
    if matrix.shape[0] != matrix.shape[1]:
        return False
    if tol is None:
        tol = qutip.settings.core["atol"]
    return _isherm(matrix._jxa, tol)


def _is_zero(vec, tol):
    return jnp.allclose(vec, 0.0, atol=tol, rtol=0)


def _is_conj(vec1, vec2, tol):
    return jnp.allclose(vec1, vec2.conj(), atol=tol, rtol=0)


def isherm_jaxdia(matrix, tol=None):
    if matrix.shape[0] != matrix.shape[1]:
        return False
    tol = tol or qutip.settings.core["atol"]
    done = []
    for offset, data in zip(matrix.offsets, matrix.data):
        if offset in done:
            continue
        start = max(0, offset)
        end = min(matrix.shape[1], matrix.shape[0] + offset)
        if -offset not in matrix.offsets:
            if not _is_zero(data[start:end], tol):
                return False
        else:
            idx = matrix.offsets.index(-offset)
            done.append(-offset)
            st = max(0, -offset)
            et = min(matrix.shape[1], matrix.shape[0] - offset)
            if not _is_conj(data[start:end], matrix.data[idx, st:et], tol):
                return False
    return True


@jit
def isdiag_jaxarray(matrix):
    mat_abs = jnp.abs(matrix._jxa)
    return jnp.trace(mat_abs) == jnp.sum(mat_abs)


def isdiag_jaxdia(matrix):
    if matrix.num_diags == 0 or (
        matrix.num_diags == 1 and matrix.offsets[0] == 0
    ):
        return True
    for offset, data in zip(matrix.offsets, matrix.data):
        if offset == 0:
            continue
        start = max(0, offset)
        end = min(matrix.shape[1], matrix.shape[0] + offset)
        if not jnp.all(data[start:end] == 0):
            return False
    return True


def iszero_jaxarray(matrix, tol=None):
    if tol is None:
        tol = qutip.settings.core["atol"]
    return jnp.allclose(matrix._jxa, 0.0, atol=tol)


def iszero_jaxdia(matrix, tol=None):
    if tol is None:
        tol = qutip.settings.core["atol"]
    if matrix.num_diags == 0:
        return True
    # We must ensure the values outside the dims are not included
    return jnp.allclose(clean_dia(matrix).data, 0.0, atol=tol)


qutip.data.isherm.add_specialisations(
    [
        (JaxArray, isherm_jaxarray),
        (JaxDia, isherm_jaxdia),
    ]
)


qutip.data.iszero.add_specialisations(
    [
        (JaxArray, iszero_jaxarray),
        (JaxDia, iszero_jaxdia),
    ]
)


qutip.data.isdiag.add_specialisations(
    [
        (JaxArray, isdiag_jaxarray),
        (JaxDia, isdiag_jaxdia),
    ]
)
