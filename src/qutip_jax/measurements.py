import jax.numpy as jnp
from .jaxarray import JaxArray
import qutip
from jax import jit
from functools import partial


__all__ = [
    "expect_jaxarray",
    "expect_super_jaxarray",
    "inner_jaxarray",
    "inner_op_jaxarray",
    "trace_jaxarray",
    "trace_oper_ket_jaxarray",
]


@partial(jit, static_argnames=["scalar_is_ket"])
def inner_jaxarray(left, right, scalar_is_ket=False):
    """Computes the inner product between left and right objects assuming they
    are kets.

    The parameter `scalar_is_ket` is only intended for the case where `left`
    and `right` are both of shape (1, 1).  In this case, `left` will be assumed
    to be a ket unless `scalar_is_ket` is False.  This parameter is ignored at
    all other times.

    Parameters
    ----------
    left, right : :class:`qutip.Qobj`
        Quantum objects from which the underlying JAX array can be accessed.
    scalar_is_ket : bool
        Specifies if the inputs have shape (1, 1).

    Returns
    -------
    out : jax.Array
        The complex valued output.
    """
    if (left._jxa.shape[0] != 1 and left._jxa.shape[1] != 1) or right._jxa.shape[
        1
    ] != 1:
        raise ValueError(
            "incompatible matrix shapes " + str(left.shape) + " and " + str(right.shape)
        )
    if (
        left._jxa.shape[1] == left._jxa.shape[0]
        and left._jxa.shape[0] == 1
        and scalar_is_ket
    ):
        left = left.conj()
    if left._jxa.shape[1] == right._jxa.shape[0]:
        out = left._jxa @ right._jxa
    else:
        out = left._jxa.T.conj() @ right._jxa
    return out[0, 0]


@partial(jit, static_argnames=["scalar_is_ket"])
def inner_op_jaxarray(left, op, right, scalar_is_ket=False):
    """Computes the inner product between left and right objects assuming they
    are operators.

    The parameter `scalar_is_ket` is only intended for the case where `left`
    and `right` are both of shape (1, 1).  In this case, `left` will be assumed
    to be a ket unless `scalar_is_ket` is False.  This parameter is ignored at
    all other times.

    Parameters
    ----------
    left, right : :class:`qutip.Qobj`
        Quantum objects from which the underlying JAX array can be accessed.
    scalar_is_ket : bool
        Specifies if the inputs have shape (1, 1).

    Returns
    -------
    out : jax.Array
        The complex valued output.
    """
    left_shape = left._jxa.shape[0] == 1 or left._jxa.shape[1] == 1
    left_op = (left._jxa.shape[0] == 1 and left._jxa.shape[1] == op._jxa.shape[0]) or (
        left._jxa.shape[1] == 1 and left._jxa.shape[0] == op._jxa.shape[0]
    )
    op_right = op._jxa.shape[1] == right._jxa.shape[0]
    right_shape = right._jxa.shape[1] == 1
    if not (left_shape and left_op and op_right and right_shape):
        raise ValueError(
            "".join(
                [
                    "incompatible matrix shapes ",
                    str(left.shape),
                    ", ",
                    str(op.shape),
                    " and ",
                    str(right.shape),
                ]
            )
        )
    if (
        left._jxa.shape[0] == 1
        and left._jxa.shape[1] == left._jxa.shape[0]
        and scalar_is_ket
    ):
        left = left.conj()
    if left._jxa.shape[1] == op._jxa.shape[0]:
        out = left._jxa @ op._jxa @ right._jxa
    else:
        out = left._jxa.T.conj() @ op._jxa @ right._jxa
    return out[0, 0]


@jit
def expect_jaxarray(op, state):
    """Computes the expectation value between op and state assuming they are
    operators and state representations (density matrix/ket).

    Parameters
    ----------
    op, state : :class:`qutip.Qobj`
        Quantum objects from which the underlying JAX array can be accessed.

    Returns
    -------
    out : jax.Array
        The complex valued output.
    """
    if (
        op._jxa.shape[0] != op._jxa.shape[1]
        or op._jxa.shape[1] != state._jxa.shape[0]
        or not (state._jxa.shape[1] == 1 or state._jxa.shape[0] == state._jxa.shape[1])
    ):
        raise ValueError(
            "incompatible matrix shapes " + str(op.shape) + " and " + str(state.shape)
        )
    if state._jxa.shape[0] == state._jxa.shape[1]:
        out = jnp.sum(op._jxa * state._jxa.T)
    else:
        out = (state._jxa.T.conj() @ op._jxa @ state._jxa)[0, 0]
    return out


@jit
def expect_super_jaxarray(op, state):
    """Computes the expectation value between op and state assuming they
    represent a superoperator and a state (vectorized).

    Parameters
    ----------
    op, state : :class:`qutip.Qobj`
        Quantum objects from which the underlying JAX array can be accessed.

    Returns
    -------
    out : jax.Array
        The complex valued output.
    """
    if state._jxa.shape[1] != 1:
        raise ValueError("expected a column-stacked matrix")
    if not (
        op._jxa.shape[0] == op._jxa.shape[1] and op._jxa.shape[1] == state._jxa.shape[0]
    ):
        raise ValueError(
            "incompatible matrix shapes " + str(op.shape) + " and " + str(state.shape)
        )

    N = int(state._jxa.shape[0] ** 0.5)
    return jnp.sum((op._jxa @ state._jxa)[:: N + 1])


@jit
def trace_jaxarray(matrix):
    """Compute the trace (sum of digaonal elements) of a square matrix."""
    if matrix._jxa.shape[0] != matrix._jxa.shape[1]:
        raise ValueError("".join([
            "matrix ", str(matrix.shape), " is not a stacked square matrix."
        ]))
    return jnp.trace(matrix._jxa)


@jit
def trace_oper_ket_jaxarray(matrix):
    """
    Compute the trace (sum of digaonal elements) of a stacked square matrix .
    """
    N = int(matrix.shape[0] ** 0.5)
    if matrix.shape[0] != N * N or matrix.shape[1] != 1:
        raise ValueError("".join([
            "matrix ", str(matrix.shape), " is not a stacked square matrix."
        ]))
    return jnp.sum(matrix._jxa[:: N + 1])



qutip.data.inner.add_specialisations(
    [
        (JaxArray, JaxArray, inner_jaxarray),
    ]
)


qutip.data.inner_op.add_specialisations(
    [
        (JaxArray, JaxArray, JaxArray, inner_op_jaxarray),
    ]
)


qutip.data.expect.add_specialisations(
    [
        (JaxArray, JaxArray, expect_jaxarray),
    ]
)


qutip.data.expect_super.add_specialisations(
    [
        (JaxArray, JaxArray, expect_super_jaxarray),
    ]
)


qutip.data.trace.add_specialisations(
    [
        (JaxArray, trace_jaxarray),
    ]
)


qutip.data.trace_oper_ket.add_specialisations(
    [
        (JaxArray, trace_oper_ket_jaxarray),
    ]
)
