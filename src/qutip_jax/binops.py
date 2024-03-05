import qutip
from .jaxarray import JaxArray
import jax.numpy as jnp

__all__ = [
    "add_jaxarray",
    "sub_jaxarray",
    "mul_jaxarray",
    "matmul_jaxarray",
    "multiply_jaxarray",
    "kron_jaxarray",
    "pow_jaxarray",
]


def _check_same_shape(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"""Incompatible shapes for addition of two matrices:
                         left={left.shape} and right={right.shape}"""
        )


def _check_matmul_shape(left, right, out):
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            "incompatible matrix shapes " + str(left.shape)
            + " and " + str(right.shape)
        )
    if (
        out is not None
        and out.shape[0] != left.shape[0]
        and out.shape[1] != right.shape[1]
    ):
        raise ValueError(
            "incompatible output shape, got "
            + str(out.shape)
            + " but needed "
            + str((left.shape[0], right.shape[1]))
        )


def add_jaxarray(left, right, scale=1):
    """
    Perform the operation
        left + scale*right
    where `left` and `right` are matrices, and `scale` is an optional complex
    scalar.
    """
    _check_same_shape(left, right)

    if scale == 1 and isinstance(scale, int):
        out = JaxArray._fast_constructor(
            left._jxa + right._jxa, shape=left.shape
        )
    else:
        out = JaxArray._fast_constructor(
            left._jxa + scale * right._jxa, shape=left.shape
        )
    return out


def sub_jaxarray(left, right):
    """
    Perform the operation
        left - right
    where `left` and `right` are matrices.
    """
    return add_jaxarray(left, right, -1)


def mul_jaxarray(matrix, value):
    """Multiply a matrix element-wise by a scalar."""
    # We don't want to check values type in case jax pass a tracer etc.
    # But we want to ensure the output is a matrix, thus don't use the
    # fast constructor.
    return JaxArray._fast_constructor(matrix._jxa * value, shape=matrix.shape)


def matmul_jaxarray(left, right, scale=1, out=None):
    """
    Compute the matrix multiplication of two matrices, with the operation
        scale * (left @ right)
    where `scale` is (optionally) a scalar, and `left` and `right` are
    matrices.

    Arguments
    ---------
    left : Data
        The left operand as either a bra or a ket matrix.

    right : Data
        The right operand as a ket matrix.

    scale : complex, optional
        The scalar to multiply the output by.
    """
    _check_matmul_shape(left, right, out)
    shape = (left.shape[0], right.shape[1])

    result = left._jxa @ right._jxa

    result *= scale

    if out is None:
        return JaxArray._fast_constructor(result, shape=shape)
    else:
        out._jxa = result + out._jxa


def multiply_jaxarray(left, right):
    """Element-wise multiplication of matrices."""
    _check_same_shape(left, right)
    return JaxArray._fast_constructor(left._jxa * right._jxa, shape=left.shape)


def kron_jaxarray(left, right):
    """
    Compute the Kronecker product of two matrices.  This is used to represent
    quantum tensor products of vector spaces.
    """
    return JaxArray._fast_constructor(jnp.kron(left._jxa, right._jxa))


def pow_jaxarray(matrix, n):
    """
    Compute the integer matrix power of the square input matrix.  The power
    must be an integer >= 0.  `A ** 0` is defined to be the identity matrix of
    the same shape.

    Arguments
    ---------
    matrix : Data
        Input matrix to take the power of.

    n : non-negative integer
        The power to which to raise the matrix.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix power only works with square matrices")
    return JaxArray._fast_constructor(jnp.linalg.matrix_power(matrix._jxa, n))


qutip.data.add.add_specialisations(
    [(JaxArray, JaxArray, JaxArray, add_jaxarray),]
)

qutip.data.sub.add_specialisations(
    [(JaxArray, JaxArray, JaxArray, sub_jaxarray),]
)

qutip.data.mul.add_specialisations(
    [(JaxArray, JaxArray, mul_jaxarray),]
)

qutip.data.matmul.add_specialisations(
    [(JaxArray, JaxArray, JaxArray, matmul_jaxarray),]
)

qutip.data.multiply.add_specialisations(
    [(JaxArray, JaxArray, JaxArray, multiply_jaxarray),]
)

qutip.data.kron.add_specialisations(
    [(JaxArray, JaxArray, JaxArray, kron_jaxarray),]
)

qutip.data.pow.add_specialisations(
    [(JaxArray, JaxArray, pow_jaxarray),]
)
