import qutip
from .jaxarray import JaxArray
from .jaxdia import JaxDia
from .binops import mul_jaxarray, mul_jaxdia
import jax.scipy.linalg as linalg
from jax import jit
import numpy as np
import jax.numpy as jnp

__all__ = [
    "neg_jaxarray",
    "neg_jaxdia",
    "adjoint_jaxarray",
    "adjoint_jaxdia",
    "transpose_jaxarray",
    "transpose_jaxdia",
    "conj_jaxarray",
    "conj_jaxdia",
    "inv_jaxarray",
    "expm_jaxarray",
    "sqrtm_jaxarray",
    "project_jaxarray",
]


def _check_square_shape(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Can only be performed for square matrix. "
            f"This matrix has shape={matrix.shape}"
        )


def neg_jaxarray(matrix):
    """Unary element-wise negation of a matrix."""
    return mul_jaxarray(matrix, -1)


@jit
def neg_jaxdia(matrix):
    """Unary element-wise negation of a matrix."""
    return mul_jaxdia(matrix, -1)


@jit
def adjoint_jaxarray(matrix):
    """Hermitian adjoint (matrix conjugate transpose)."""
    return JaxArray(matrix._jxa.T.conj())


def transpose_jaxarray(matrix):
    """Transpose of a matrix."""
    return JaxArray(matrix._jxa.T)


def conj_jaxarray(matrix):
    """Element-wise conjugation of a matrix."""
    return JaxArray._fast_constructor(matrix._jxa.conj(), matrix.shape)


@jit
def conj_jaxdia(matrix):
    """Element-wise conjugation of a matrix."""
    return JaxDia._fast_constructor(
        matrix.offsets, matrix.data.conj(), matrix.shape
    )


@jit
def transpose_jaxdia(matrix):
    """Transpose of a matrix."""
    new_offset = tuple(-diag for diag in matrix.offsets[::-1])
    new_data = jnp.zeros(
        (matrix.data.shape[0], matrix.shape[0]), dtype=jnp.complex128
    )
    for i, diag in enumerate(matrix.offsets):
        old_start = max(0, diag)
        old_end = min(matrix.shape[1], matrix.shape[0] + diag)
        new_start = max(0, -diag)
        new_end = min(matrix.shape[0], matrix.shape[1] - diag)
        new_data = new_data.at[-i - 1, new_start:new_end].set(
            matrix.data[i, old_start:old_end]
        )
    return JaxDia._fast_constructor(new_offset, new_data, matrix.shape[::-1])


@jit
def adjoint_jaxdia(matrix):
    """Hermitian adjoint (matrix conjugate transpose)."""
    new_offset = tuple(-diag for diag in matrix.offsets[::-1])
    new_data = jnp.zeros(
        (matrix.data.shape[0], matrix.shape[0]), dtype=jnp.complex128
    )
    for i, diag in enumerate(matrix.offsets):
        old_start = max(0, diag)
        old_end = min(matrix.shape[1], matrix.shape[0] + diag)
        new_start = max(0, -diag)
        new_end = min(matrix.shape[0], matrix.shape[1] - diag)
        new_data = new_data.at[-i - 1, new_start:new_end].set(
            matrix.data[i, old_start:old_end].conj()
        )
    return JaxDia._fast_constructor(new_offset, new_data, matrix.shape[::-1])


def expm_jaxarray(matrix):
    """Matrix exponential `e**A` for a matrix `A`."""
    _check_square_shape(matrix)
    return JaxArray._fast_constructor(linalg.expm(matrix._jxa), matrix.shape)


def inv_jaxarray(matrix):
    """Matrix inverse `A**-1` for a matrix `A`."""
    _check_square_shape(matrix)
    return JaxArray._fast_constructor(linalg.inv(matrix._jxa), matrix.shape)


def sqrtm_jaxarray(matrix):
    """Matrix square root `sqrt(A)` for a matrix `A`."""
    _check_square_shape(matrix)
    return JaxArray._fast_constructor(linalg.sqrtm(matrix._jxa), matrix.shape)


@jit
def _project_ket(array):
    return array @ array.T.conj()


@jit
def _project_bra(array):
    return array.T.conj() @ array


def project_jaxarray(state):
    """
    Get the projector of a state with itself.  Mathematically, if passed an
    object `|a>` or `<a|`, then return the matrix `|a><a|`.
    """
    if state.shape[1] == 1:
        # Is ket
        out = _project_ket(state._jxa)
    elif state.shape[0] == 1:
        # Is bra
        out = _project_bra(state._jxa)
    else:
        raise ValueError("state must be a ket or a bra.")
    return JaxArray(out)


qutip.data.neg.add_specialisations(
    [
        (JaxArray, JaxArray, neg_jaxarray),
        (JaxDia, JaxDia, neg_jaxdia),
    ]
)


qutip.data.adjoint.add_specialisations(
    [
        (JaxArray, JaxArray, adjoint_jaxarray),
        (JaxDia, JaxDia, adjoint_jaxdia),
    ]
)


qutip.data.transpose.add_specialisations(
    [
        (JaxArray, JaxArray, transpose_jaxarray),
        (JaxDia, JaxDia, transpose_jaxdia),
    ]
)


qutip.data.conj.add_specialisations(
    [
        (JaxArray, JaxArray, conj_jaxarray),
        (JaxDia, JaxDia, conj_jaxdia),
    ]
)


qutip.data.expm.add_specialisations(
    [
        (JaxArray, JaxArray, expm_jaxarray),
    ]
)


qutip.data.inv.add_specialisations(
    [
        (JaxArray, JaxArray, inv_jaxarray),
    ]
)


qutip.data.sqrtm.add_specialisations(
    [(JaxArray, JaxArray, sqrtm_jaxarray),]
)


qutip.data.project.add_specialisations(
    [
        (JaxArray, JaxArray, project_jaxarray),
    ]
)
