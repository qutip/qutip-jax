import jax.numpy as jnp
import numpy as np
from jax import tree_util
from jax.config import config

config.update("jax_enable_x64", True)

import numpy as np

from qutip.core.data.base import Data

import numbers


__all__ = ["JaxDia"]


class JaxDia(Data):
    data: jnp.ndarray
    offsets: tuple
    shape: tuple

    def __init__(self, arg, shape=None, copy=None):
        offsets, data = arg
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
        return self.__class__((self.offsets, self.data), self.shape, copy=True)

    def to_array(self):
        from .convert import jaxarray_from_jaxdia
        return jaxarray_from_jaxdia(self).to_array()

    @classmethod
    def _fast_constructor(cls, offsets, data, shape):
        out = cls.__new__(cls)
        Data.__init__(out, shape)
        out.data = data
        out.offsets = offsets
        out.num_diags = len(out.offsets)
        return out

    def _tree_flatten(self):
        children = (self.data, )  # arrays / dynamic values
        aux_data = {"shape": self.shape, "offsets": self.offsets}  # static values
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


def clean_diag(matrix):
    idx = np.argsort(matrix.offsets)
    new_offset = tuple(matrix.offsets[i] for i in idx)
    new_data = matrix.data[idx, :]

    for i in range(len(new_offset)):
        start = max(0, new_offset[i])
        end = min(matrix.shape[1], matrix.shape[0] + new_offset[i])
        new_data = new_data.at[i, :start].set(0)
        new_data = new_data.at[i, end:].set(0)

    return JaxDia._fast_constructor(new_offset, new_data, matrix.shape)
