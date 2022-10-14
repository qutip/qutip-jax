import jax.numpy as jnp
import jax.scipy.linalg as linalg
from .jaxarray import JaxArray
import qutip
from jax import jit


__all__ = []


def frobenius_jaxarray(matrix):
    """Computes the Frobenius norm.

    Parameters
    ----------
    matrix :class:`qutip.Qobj`
        Quantum object from which a JAX array can be accessed.

    Returns:
        float: The Frobenius norm.
    """
    return jnp.linalg.norm(matrix._jxa)


def l2_jaxarray(matrix):
    """Computes the L2 norm.

    Parameters
    ----------
    matrix :class:`qutip.Qobj`
        Quantum object from which a JAX array can be accessed.

    Returns:
        float: The L2 norm.
    """
    if matrix._jxa.shape[0] != 1 and matrix._jxa.shape[1] != 1:
        raise ValueError("L2 norm is only defined on vectors")
    return jnp.linalg.norm(matrix._jxa)


@jit
def trace_jaxarray(matrix):
    """Computes the trace norm.

    Parameters
    ----------
    matrix :class:`qutip.Qobj`
        Quantum object from which a JAX array can be accessed.

    Returns:
        float: The trace norm.
    """
    if matrix._jxa.shape[0] == 1 or matrix._jxa.shape[1] == 1:
        return jnp.linalg.norm(matrix._jxa)
    else:
        out = matrix._jxa @ matrix._jxa.T.conj()
        out = linalg.sqrtm(out)
        out = jnp.trace(out)
        return out


def one_jaxarray(matrix):
    """Computes the one norm.

    Parameters
    ----------
    matrix :class:`qutip.Qobj`
        Quantum object from which a JAX array can be accessed.

    Returns:
        float: The one norm.
    """
    return jnp.linalg.norm(matrix._jxa, ord=1)


def max_jaxarray(matrix):
    """Computes the max absolute value.

    Parameters
    ----------
    matrix :class:`qutip.Qobj`
        Quantum object from which a JAX array can be accessed.

    Returns:
        float: The max absolute value.
    """
    return jnp.max(jnp.abs(matrix._jxa))


qutip.data.norm.frobenius.add_specialisations(
    [
        (JaxArray, frobenius_jaxarray),
    ]
)


qutip.data.norm.l2.add_specialisations(
    [
        (JaxArray, l2_jaxarray),
    ]
)


qutip.data.norm.trace.add_specialisations(
    [
        (JaxArray, trace_jaxarray),
    ]
)


qutip.data.norm.max.add_specialisations(
    [
        (JaxArray, max_jaxarray),
    ]
)


qutip.data.norm.one.add_specialisations(
    [
        (JaxArray, one_jaxarray),
    ]
)
