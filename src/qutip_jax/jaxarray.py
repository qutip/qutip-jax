import jax.numpy as jnp
from jax import tree_util
from jax.config import config

config.update("jax_enable_x64", True)

import numpy as np

from qutip.core.data.base import Data

import numbers


__all__ = ["JaxArray"]


class JaxArray(Data):
    def __init__(self, data, shape=None, copy=None):
        jxa = jnp.array(data, dtype=jnp.complex128)

        if shape is None:
            shape = data.shape
            if len(shape) == 0:
                shape = (1, 1)
            if len(shape) == 1:
                shape = (shape[0], 1)

        if not (
            isinstance(shape, tuple)
            and len(shape) == 2
            and isinstance(shape[0], numbers.Integral)
            and isinstance(shape[1], numbers.Integral)
            and shape[0] > 0
            and shape[1] > 0
        ):
            raise ValueError(
                """Shape must be a 2-tuple of positive ints, but is """ + repr(shape)
            )

        if np.prod(shape) != np.prod(data.shape):
            raise ValueError("Shape of data does not match argument.")

        # if copy:
        #     # Since jax's arrays are immutable, we could probably skip this.
        #     data = data.copy()
        self._jxa = jxa.reshape(shape)
        super().__init__(shape)

    def copy(self):
        return self.__class__(self._jxa, copy=True)

    def to_array(self):
        return np.array(self._jxa)

    def conj(self):
        return self.__class__(self._jxa.conj())

    def transpose(self):
        return self.__class__(self._jxa.T)

    def adjoint(self):
        return self.__class__(self._jxa.T.conj())

    def trace(self):
        return jnp.trace(self._jxa)

    def _tree_flatten(self):
        children = (self._jxa,)  # arrays / dynamic values
        aux_data = {"shape": self.shape}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(
    JaxArray, JaxArray._tree_flatten, JaxArray._tree_unflatten
)
