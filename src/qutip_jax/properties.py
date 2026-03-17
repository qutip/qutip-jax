import jax
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

@partial(jit, static_argnames=["shape", "tol"])
def _isherm_dia_jit(offsets, data, shape, tol):
    num_rows, num_cols = shape
    num_diags = offsets.shape[0]

    # Return index of partner diagonal:
    def find_partner_idx(target_offset):
        mask = (offsets == -target_offset)
        idx = jnp.argmax(mask)
        return jnp.where(mask[idx], idx, -1)
    
    def check_one_diagonal(i):
        offset = offsets[i]
        diag_data = data[i]
        
        col_indices = jnp.arange(num_cols)
        row_indices = col_indices - offset
        
        mask = (row_indices >= 0) & (row_indices < num_rows)

        partner_idx = find_partner_idx(offset)

        def handle_no_partner():
            return jnp.all(jnp.where(mask, jnp.abs(diag_data) <= tol, True))
        
        def handle_with_partner():
            partner_diag_data = data[partner_idx]
            partner_aligned = jnp.take(partner_diag_data, row_indices, mode='clip')
            diff = jnp.abs(diag_data - partner_aligned.conj())
            return jnp.all(jnp.where(mask, diff <= tol, True))
        
        return jnp.where(partner_idx == -1, handle_no_partner(), handle_with_partner())
    
    results = jax.vmap(check_one_diagonal)(jnp.arange(num_diags))
    return jnp.all(results)

def isherm_jaxdia(matrix, tol=None):
    if matrix.shape[0] != matrix.shape[1]:
        return False
    if tol is None:
        tol = qutip.settings.core["atol"]

    offset_array = jnp.array(matrix.offsets)
    return _isherm_dia_jit(offset_array, matrix.data, matrix.shape, tol)


@jit
def isdiag_jaxarray(matrix):
    mat_abs = jnp.abs(matrix._jxa)
    return jnp.trace(mat_abs) == jnp.sum(mat_abs)

@partial(jit, static_argnames=["shape"])
def _isdiag_dia_jit(offsets, data, shape):
    num_rows, num_cols = shape
    num_diags = offsets.shape[0]

    def check_one_diagonal(i):
        offset = offsets[i]
        diag_data = data[i]

        col_indices = jnp.arange(num_cols)
        row_indices = col_indices - offset
        mask = (row_indices >= 0) & (row_indices < num_rows)
        
        all_zero = jnp.all(jnp.where(mask, diag_data == 0, True))
        skip = (offset == 0)

        return jnp.where(skip, True, all_zero)
    
    results = jax.vmap(check_one_diagonal)(jnp.arange(num_diags))
    return jnp.all(results)

def isdiag_jaxdia(matrix):
    if matrix.num_diags == 0 or (matrix.num_diags == 1 and matrix.offsets[0] == 0):
        return True

    offset_array = jnp.array(matrix.offsets)
    return _isdiag_dia_jit(offset_array, matrix.data, matrix.shape)


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
