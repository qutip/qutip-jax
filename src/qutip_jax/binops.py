import qutip
from .jaxarray import JaxArray
from .jaxdia import JaxDia, clean_dia
import jax.numpy as jnp
import jax
from jax import vmap, jit
from functools import partial

__all__ = [
    "add_jaxarray",
    "add_jaxdia",
    "sub_jaxarray",
    "sub_jaxdia",
    "mul_jaxarray",
    "mul_jaxdia",
    "matmul_jaxarray",
    "matmul_jaxdia",
    "matmul_jaxdia_jaxarray_jaxarray",
    "matmul_jaxarray_jaxdia_jaxarray",
    "multiply_jaxarray",
    "multiply_jaxdia",
    "kron_jaxarray",
    "kron_jaxdia",
    "pow_jaxarray",
]


def _check_same_shape(left, right):
    if left.shape != right.shape:
        raise ValueError(
            "Incompatible shapes for addition of two matrices: "
            f"left={left.shape} and right={right.shape}"
            ""
        )


def _check_matmul_shape(left, right, out):
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            f"incompatible matrix shapes {left.shape} and {right.shape}"
        )
    if (
        out is not None
        and out.shape[0] != left.shape[0]
        and out.shape[1] != right.shape[1]
    ):
        raise ValueError(
            f"incompatible output shape, got {out.shape}, but needed "
            + str((left.shape[0], right.shape[1]))
        )


@jit
def add_jaxarray(left, right, scale=None):
    """
    Perform the operation
        left + scale*right
    where `left` and `right` are matrices, and `scale` is an optional complex
    scalar.
    """
    _check_same_shape(left, right)

    if scale == None:
        out = JaxArray._fast_constructor(
            left._jxa + right._jxa, shape=left.shape
        )
    else:
        out = JaxArray._fast_constructor(
            left._jxa + scale * right._jxa, shape=left.shape
        )
    return out


@jit
def add_jaxdia(left, right, scale=None):
    """
    Perform the operation
        left + scale*right
    where `left` and `right` are matrices, and `scale` is an optional complex
    scalar.
    """
    _check_same_shape(left, right)
    diag_left = 0
    diag_right = 0
    data = []
    offsets = []

    all_diag = set(left.offsets) | set(right.offsets)

    for diag in all_diag:
        if diag in left.offsets and diag in right.offsets:
            diag_left = left.offsets.index(diag)
            diag_right = right.offsets.index(diag)
            offsets.append(diag)
            if scale is None:
                data.append(left.data[diag_left, :] + right.data[diag_right, :])
            else:
                data.append(
                    left.data[diag_left, :] + right.data[diag_right, :] * scale
                )

        elif diag in left.offsets:
            diag_left = left.offsets.index(diag)
            offsets.append(diag)
            data.append(left.data[diag_left, :])

        elif diag in right.offsets:
            diag_right = right.offsets.index(diag)
            offsets.append(diag)
            if scale is None:
                data.append(right.data[diag_right, :])
            else:
                data.append(right.data[diag_right, :] * scale)

    return JaxDia((jnp.array(data), tuple(offsets),), left.shape, False)


@jit
def sub_jaxarray(left, right):
    """
    Perform the operation
        left - right
    where `left` and `right` are matrices.
    """
    return add_jaxarray(left, right, -1)


@jit
def sub_jaxdia(left, right):
    """
    Perform the operation
        left - right
    where `left` and `right` are matrices.
    """
    return add_jaxdia(left, right, -1)


@jit
def mul_jaxarray(matrix, value):
    """Multiply a matrix element-wise by a scalar."""
    # We don't want to check values type in case jax pass a tracer etc.
    # But we want to ensure the output is a matrix, thus don't use the
    # fast constructor.
    return JaxArray(matrix._jxa * value)


@partial(jit, donate_argnums=[0])
def imul_jaxarray(matrix, value):
    """Multiply a matrix element-wise by a scalar."""
    # We don't want to check values type in case jax pass a tracer etc.
    # But we want to ensure the output is a matrix, thus don't use the
    # fast constructor.
    return JaxArray(matrix._jxa * value)


@jit
def mul_jaxdia(matrix, value):
    """Multiply a matrix element-wise by a scalar."""
    return JaxDia._fast_constructor(
        matrix.offsets, matrix.data * value, matrix.shape
    )


@partial(jit, donate_argnums=(3,))
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


@partial(jit, donate_argnums=(3,))
def matmul_jaxdia(left, right, scale=1.0, out=None):
    _check_matmul_shape(left, right, out)
    out_dict = {}

    for diag_left in range(left.num_diags):
        for diag_right in range(right.num_diags):
            off_out = left.offsets[diag_left] + right.offsets[diag_right]
            if off_out <= -left.shape[0] or off_out >= right.shape[1]:
                continue

            start_left = (
                max(0, left.offsets[diag_left]) + right.offsets[diag_right]
            )
            start_right = max(0, right.offsets[diag_right])
            start_out = max(0, off_out)
            start = max(start_left, start_right, start_out)

            end_left = (
                min(left.shape[1], left.shape[0] + left.offsets[diag_left])
                + right.offsets[diag_right]
            )
            end_right = min(
                right.shape[1], right.shape[0] + right.offsets[diag_right]
            )
            end_out = min(right.shape[1], left.shape[0] + off_out)
            end = min(end_left, end_right, end_out)

            left_shift = -right.offsets[diag_right]
            data = jnp.zeros(right.shape[1], dtype=jnp.complex128)
            data = data.at[start:end].set(
                scale
                * left.data[diag_left, left_shift + start : left_shift + end]
                * right.data[diag_right, start:end]
            )

            if off_out in out_dict:
                out_dict[off_out] = out_dict[off_out] + data
            else:
                out_dict[off_out] = data

    out_dia = JaxDia._fast_constructor(
        tuple(out_dict.keys()),
        jnp.array(list(out_dict.values())),
        (left.shape[0], right.shape[1]),
    )
    if out is not None:
        out_dia = add_jaxdia(out, out_dia)
    return out_dia


@partial(jit, donate_argnums=(3,))
def matmul_jaxdia_jaxarray_jaxarray(left, right, scale=None, out=None):
    _check_matmul_shape(left, right, out)
    mul = vmap(jnp.multiply, (0, 0))
    if out is None:
        out = jnp.zeros((left.shape[0], right.shape[1]), dtype=jnp.complex128)
    else:
        out = out._jxa

    for offset, data in zip(left.offsets, left.data):
        start = max(0, offset)
        end = min(left.shape[1], left.shape[0] + offset)
        top = max(0, -offset)
        bottom = top + end - start

        if scale is not None:
            out = out.at[top:bottom, :].add(
                mul(data[start:end], right._jxa[start:end, :]) * scale
            )
        else:
            out = out.at[top:bottom, :].add(
                mul(data[start:end], right._jxa[start:end, :])
            )

    return JaxArray(out, shape=(left.shape[0], right.shape[1]), copy=False)


@partial(jit, donate_argnums=(3,))
def matmul_jaxarray_jaxdia_jaxarray(left, right, scale=1.0, out=None):
    _check_matmul_shape(left, right, out)
    mul = vmap(jnp.multiply, (1, 0))
    if out is None:
        out = jnp.zeros((left.shape[0], right.shape[1]), dtype=jnp.complex128)
    else:
        out = out._jxa

    for offset, data in zip(right.offsets, right.data):
        start = max(0, offset)
        end = min(right.shape[1], right.shape[0] + offset)
        top = max(0, -offset)
        bottom = top + end - start

        out = out.at[:, start:end].add(
            mul(left._jxa[:, top:bottom], data[start:end]).T * scale
        )

    return JaxArray(out, shape=(left.shape[0], right.shape[1]), copy=False)


@jit
def multiply_jaxarray(left, right):
    """Element-wise multiplication of matrices."""
    _check_same_shape(left, right)
    return JaxArray._fast_constructor(left._jxa * right._jxa, shape=left.shape)


@jit
def multiply_jaxdia(left, right):
    """Element-wise multiplication of matrices."""
    _check_same_shape(left, right)
    diag_left = 0
    diag_right = 0
    data = []
    offsets = []

    for i, diag in enumerate(left.offsets):
        if diag not in right.offsets:
            continue
        j = right.offsets.index(diag)
        offsets.append(diag)
        data.append(left.data[i, :] * right.data[j, :])

    out = JaxDia._fast_constructor(tuple(offsets), jnp.array(data), left.shape)

    return out


@jit
def kron_jaxarray(left, right):
    """
    Compute the Kronecker product of two matrices.  This is used to represent
    quantum tensor products of vector spaces.
    """
    return JaxArray(jnp.kron(left._jxa, right._jxa))


@jit
def _multiply_outer(left, right):
    return vmap(vmap(jnp.multiply, (None, 0)), (0, None))(left, right).ravel()


@jit
def kron_jaxdia(left, right):
    """
    Compute the Kronecker product of two matrices.  This is used to represent
    quantum tensor products of vector spaces.
    """
    nrows = left.shape[0] * right.shape[0]
    ncols = left.shape[1] * right.shape[1]
    left = clean_dia(left)
    right = clean_dia(right)
    out = {}

    if right.shape[0] == right.shape[1]:
        for diag_left in range(left.num_diags):
            for diag_right in range(right.num_diags):
                out_diag = (
                    left.offsets[diag_left] * right.shape[0]
                    + right.offsets[diag_right]
                )
                out_data = _multiply_outer(
                    left.data[diag_left], right.data[diag_right]
                )
                if out_diag in out:
                    out[out_diag] = out[out_diag] + out_data
                else:
                    out[out_diag] = out_data

    else:
        delta = right.shape[0] - right.shape[1]
        for diag_left in range(left.num_diags):
            start_left = max(0, left.offsets[diag_left])
            end_left = min(
                left.shape[1], left.shape[0] + left.offsets[diag_left]
            )
            for diag_right in range(right.num_diags):
                start_right = max(0, right.offsets[diag_right])
                end_right = min(
                    right.shape[1], right.shape[0] + right.offsets[diag_right]
                )

                for col_left in range(start_left, end_left):
                    out_diag = (
                        left.offsets[diag_left] * right.shape[0]
                        + right.offsets[diag_right]
                        - col_left * delta
                    )
                    data = jnp.zeros(ncols, dtype=jnp.complex128)
                    data = data.at[
                        col_left * right.shape[1] : col_left * right.shape[1]
                        + right.shape[1]
                    ].set(
                        left.data[diag_left, col_left] * right.data[diag_right]
                    )

                    if out_diag in out:
                        out[out_diag] = out[out_diag] + data
                    else:
                        out[out_diag] = data

    out = JaxDia(
        (jnp.array(list(out.values())), tuple(out.keys())),
        shape=(nrows, ncols)
    )
    out = clean_dia(out)
    return out


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
    return JaxArray(jnp.linalg.matrix_power(matrix._jxa, n))


qutip.data.add.add_specialisations(
    [
        (JaxArray, JaxArray, JaxArray, add_jaxarray),
        (JaxDia, JaxDia, JaxDia, add_jaxdia),
    ]
)

qutip.data.sub.add_specialisations(
    [
        (JaxArray, JaxArray, JaxArray, sub_jaxarray),
        (JaxDia, JaxDia, JaxDia, sub_jaxdia),
    ]
)

qutip.data.imul.add_specialisations(
    [
        (JaxArray, JaxArray, imul_jaxarray),
    ]
)

qutip.data.mul.add_specialisations(
    [
        (JaxArray, JaxArray, mul_jaxarray),
        (JaxDia, JaxDia, mul_jaxdia),
    ]
)

qutip.data.matmul.add_specialisations(
    [
        (JaxArray, JaxArray, JaxArray, matmul_jaxarray),
        (JaxDia, JaxDia, JaxDia, matmul_jaxdia),
        (JaxDia, JaxArray, JaxArray, matmul_jaxdia_jaxarray_jaxarray),
        (JaxArray, JaxDia, JaxArray, matmul_jaxarray_jaxdia_jaxarray),
    ]
)

qutip.data.multiply.add_specialisations(
    [
        (JaxArray, JaxArray, JaxArray, multiply_jaxarray),
        (JaxDia, JaxDia, JaxDia, multiply_jaxdia),
    ]
)

qutip.data.kron.add_specialisations(
    [
        (JaxArray, JaxArray, JaxArray, kron_jaxarray),
        (JaxDia, JaxDia, JaxDia, kron_jaxdia),
    ]
)

qutip.data.pow.add_specialisations(
    [
        (JaxArray, JaxArray, pow_jaxarray),
    ]
)
