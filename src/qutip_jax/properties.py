import jax.numpy as jnp
from .jaxarray import JaxArray
import qutip
from jax import jit
from functools import partial


__all__ = ["isherm_jaxarray", "isdiag_jaxarray", "iszero_jaxarray"]


@partial(jit, static_argnames=["tol"])
def _isherm(matrix, tol):
    return jnp.allclose(matrix, matrix.T.conj(), atol=tol, rtol=0)


# jitting this makes it 100x slower for nonsquare.
# splitting it like this seems ideal.
def isherm_jaxarray(matrix, tol=None):
    if matrix.shape[0] != matrix.shape[1]:
        return False
    tol = tol or qutip.settings.core["atol"]
    return _isherm(matrix._jxa, tol)


@jit
def isdiag_jaxarray(matrix):
    mat_abs = jnp.abs(matrix._jxa)
    return jnp.trace(mat_abs) == jnp.sum(mat_abs)


def iszero_jaxarray(matrix, tol):
    tol = tol or qutip.settings.core["atol"]
    return jnp.allclose(matrix._jxa, 0.0, atol=tol)


qutip.data.isdiag.add_specialisations(
    [
        (JaxArray, isdiag_jaxarray),
    ]
)


qutip.data.iszero.add_specialisations(
    [
        (JaxArray, iszero_jaxarray),
    ]
)


qutip.data.isdiag.add_specialisations(
    [
        (JaxArray, isdiag_jaxarray),
    ]
)
