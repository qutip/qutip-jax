import qutip
from .jaxarray import JaxArray
from .binops import mul_jaxarray
import jax.scipy.linalg as linalg
from jax import jit

__all__ = [
    "neg_jaxarray",
    "adjoint_jaxarray",
    "transpose_jaxarray",
    "conj_jaxarray",
    "inv_jaxarray",
    "expm_jaxarray",
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
def adjoint_jaxarray(matrix):
    """Hermitian adjoint (matrix conjugate transpose)."""
    return JaxArray(matrix._jxa.T.conj())


def transpose_jaxarray(matrix):
    """Transpose of a matrix."""
    return JaxArray(matrix._jxa.T)


def conj_jaxarray(matrix):
    """Element-wise conjugation of a matrix."""
    return JaxArray._fast_constructor(matrix._jxa.conj(), matrix.shape)


def expm_jaxarray(matrix):
    """Matrix exponential `e**A` for a matrix `A`."""
    _check_square_shape(matrix)
    return JaxArray._fast_constructor(linalg.expm(matrix._jxa), matrix.shape)


def inv_jaxarray(matrix):
    """Matrix inverse `A**-1` for a matrix `A`."""
    _check_square_shape(matrix)
    return JaxArray._fast_constructor(linalg.inv(matrix._jxa), matrix.shape)


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
    ]
)


qutip.data.adjoint.add_specialisations(
    [
        (JaxArray, JaxArray, adjoint_jaxarray),
    ]
)


qutip.data.transpose.add_specialisations(
    [
        (JaxArray, JaxArray, transpose_jaxarray),
    ]
)


qutip.data.conj.add_specialisations(
    [
        (JaxArray, JaxArray, conj_jaxarray),
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


qutip.data.project.add_specialisations(
    [
        (JaxArray, JaxArray, project_jaxarray),
    ]
)
