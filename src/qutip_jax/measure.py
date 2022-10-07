import jax.numpy as jnp
from .jaxarray import JaxArray
import qutip
from jax import jit


__all__ = [
    "expect_jaxarray",
    "expect_super_jaxarray",
    "inner_jaxarray",
    "inner_op_jaxarray",
]


def _check_shape_inner(left, right):
    if (
        (left.shape[0] != 1 and left.shape[1] != 1)
        or right.shape[1] != 1
    ):
        raise ValueError(
            "incompatible matrix shapes "
            + str(left.shape)
            + " and "
            + str(right.shape)
        )

def _check_shape_inner_op(left, op, right):
    left_shape = left.shape[0] == 1 or left.shape[1] == 1
    left_op = (
        (left.shape[0] == 1 and left.shape[1] == op.shape[0])
        or (left.shape[1] == 1 and left.shape[0] == op.shape[0])
    )
    op_right = op.shape[1] == right.shape[0]
    right_shape = right.shape[1] == 1
    if not (left_shape and left_op and op_right and right_shape):
        raise ValueError("".join([
            "incompatible matrix shapes ",
            str(left.shape),
            ", ",
            str(op.shape),
            " and ",
            str(right.shape),
        ]))


def inner_jaxarray(left, right):
    _check_shape_inner(left, right)
    if left.shape[1] == right.shape[0]:
        out = left @ right
    else:
        out = left.T.conj() @ right
    return out[0, 0]


def inner_op_jaxarray(left, right):
    _check_shape_inner_op(left, op, right)
    if left.shape[1] == op.shape[0]:
        out = left @ op @ right
    else:
        out = left.T.conj() @ op @ right
    return out[0, 0]


def expect_jaxarray(op, state):
    if left.shape[0] == left.shape[1]:
        out = jnp.trace(op @ state)
    else:
        out = (state.T.conj() @ op @ state)[0, 0]
    return out


def expect_super_jaxarray(op, state):
    N = int(state.shape[0]**0.5)
    return jnp.sum((op @ state)[::N+1])
