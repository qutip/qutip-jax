import jax.numpy as jnp
import numpy as np
from jax import tree_util, jit, config
from qutip.core.data.extract import extract
import qutip.core.data as _data
import numpy as np
from qutip.core.data.base import Data
import numbers


config.update("jax_enable_x64", True)

__all__ = ["JaxDia"]


class JaxDia(Data):
    data: jnp.ndarray
    offsets: tuple
    shape: tuple

    def __init__(self, arg, shape=None, copy=None):
        data, offsets = arg
        offsets = tuple(np.atleast_1d(offsets).astype(jnp.int64))
        data = jnp.atleast_2d(data).astype(jnp.complex128)

        if not (
            isinstance(shape, tuple)
            and len(shape) == 2
            and isinstance(shape[0], numbers.Integral)
            and isinstance(shape[1], numbers.Integral)
            and shape[0] > 0
            and shape[1] > 0
        ):
            raise ValueError(
                """Shape must be a 2-tuple of positive ints, but is """
                + repr(shape)
            )

        self.data = data
        self.offsets = offsets
        self.num_diags = len(offsets)
        super().__init__(shape)

    def copy(self):
        return self.__class__((self.data, self.offsets), self.shape, copy=True)

    def to_array(self):
        from .convert import jaxarray_from_jaxdia

        return jaxarray_from_jaxdia(self).to_array()

    def trace(self):
        from .measurements import trace_jaxdia

        return trace_jaxdia(self)

    def conj(self):
        from .unary import conj_jaxdia

        return conj_jaxdia(self)

    def transpose(self):
        from .unary import transpose_jaxdia

        return transpose_jaxdia(self)

    def adjoint(self):
        from .unary import adjoint_jaxdia

        return adjoint_jaxdia(self)

    @classmethod
    def _fast_constructor(cls, offsets, data, shape):
        out = cls.__new__(cls)
        Data.__init__(out, shape)
        out.data = data
        out.offsets = offsets
        out.num_diags = len(out.offsets)
        return out

    def _tree_flatten(self):
        children = (self.data,)  # arrays / dynamic values
        aux_data = {
            "shape": self.shape,
            "offsets": self.offsets,
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        # unflatten should not check data validity
        # jax can pass tracer, object, etc.
        out = cls.__new__(cls)
        out.data = children[0]
        out.offsets = aux_data["offsets"]
        out.num_diags = len(out.offsets)
        shape = aux_data["shape"]
        Data.__init__(out, shape)
        return out


tree_util.register_pytree_node(
    JaxDia, JaxDia._tree_flatten, JaxDia._tree_unflatten
)


@jit
def clean_dia(matrix):
    idx = np.argsort(matrix.offsets)
    new_offset = tuple(matrix.offsets[i] for i in idx)
    new_data = matrix.data[idx, :]

    for i in range(len(new_offset)):
        start = max(0, new_offset[i])
        end = min(matrix.shape[1], matrix.shape[0] + new_offset[i])
        new_data = new_data.at[i, :start].set(0)
        new_data = new_data.at[i, end:].set(0)

    return JaxDia._fast_constructor(new_offset, new_data, matrix.shape)


def tidyup_jaxdia(matrix, tol, _=None):
    matrix = clean_dia(matrix)
    new_offset = []
    new_data = []
    for offset, data in zip(matrix.offsets, matrix.data):
        real = data.real
        mask_r = real < tol
        imag = data.imag
        mask_i = imag < tol
        if jnp.all(mask_r) and jnp.all(mask_i):
            continue
        data = real.at[mask_r].set(0) + 1j * imag.at[mask_i].set(0)
        new_offset.append(offset)
        new_data.append(data)
    new_offset = tuple(new_offset)
    new_data = jnp.array(new_data)
    return JaxDia._fast_constructor(new_offset, new_data, matrix.shape)


_data.tidyup.add_specialisations([(JaxDia, tidyup_jaxdia)], _defer=True)


def extract_jaxdia(matrix, format=None, _=None):
    """
    Return ``jaxdia_matrix`` as a pair of offsets and diagonals.

    It can be extracted as either a dict of the offset to the diagonal or a
    tuple of ``(offsets, diagonals)``.
    The diagonal are the lenght of the number of columns.
    Each entry is at the position of the column.

    The element ``A[3, 5]`` is at ``extract_jaxdia(A, "dict")[5-3][5]``.

    Parameters
    ----------
    matrix : Data
        The matrix to convert to common type.

    format : str, {"dict"}
        Type of the output.
    """
    if format in ["dict", None]:
        out = {}
        for offset, data in zip(matrix.offsets, matrix.data):
            out[offset] = data

    elif format in ["tuple"]:
        out = (matrix.offsets, matrix.data)
    else:
        raise ValueError("Dia can only be extracted to 'dict' or 'tuple'")
    return out


extract.add_specialisations([(JaxDia, extract_jaxdia)], _defer=True)
