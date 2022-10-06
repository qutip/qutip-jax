import qutip
from .jaxarray import JaxArray

__all__ = [
    "add_jaxarray",
    "sub_jaxarray",
    "matmul_jaxarray",
    "multiply_jaxarray"
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
    return add_jaxarray(left, right, -1)


def matmul_jaxarray(left, right, scale=1, out=None):
    _check_matmul_shape(left, right, out)
    shape = (left.shape[0], right.shape[1])

    result = left._jxa @ right._jxa

    if scale != 1 or not isinstance(scale, int):
        result *= scale

    if out is None:
        return JaxArray._fast_constructor(result, shape=shape)
    else:
        out._jxa = result + out._jxa


def multiply_jaxarray(left, right):
    _check_same_shape(left, right)
    return JaxArray._fast_constructor(left._jxa * right._jxa, shape=left.shape)


qutip.data.matmul.add_specialisations(
    [(JaxArray, JaxArray, JaxArray, matmul_jaxarray),]
)

qutip.data.add.add_specialisations(
    [(JaxArray, JaxArray, JaxArray, add_jaxarray),]
)

qutip.data.sub.add_specialisations(
    [(JaxArray, JaxArray, JaxArray, sub_jaxarray),]
)

qutip.data.multiply.add_specialisations(
    [(JaxArray, JaxArray, JaxArray, multiply_jaxarray),]
)
