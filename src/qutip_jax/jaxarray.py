import jax.numpy as jnp
from jax import tree_util, config, jit
import numbers
import numpy as np
from functools import partial

config.update("jax_enable_x64", True)

from qutip.core.data.base import Data
from qutip.core.data.extract import extract


__all__ = ["JaxArray"]


class JaxArray(Data):
    _jxa: jnp.ndarray
    shape: tuple

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
                """Shape must be a 2-tuple of positive ints, but is """
                + repr(shape)
            )
        if np.prod(shape) != np.prod(data.shape):
            raise ValueError("Shape of data does not match argument.")

        self._jxa = jxa.reshape(shape)
        Data.__init__(self, shape)

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

    def __add__(self, other):
        if isinstance(other, JaxArray):
            out = self._jxa + other._jxa
            return JaxArray._fast_constructor(out, out.shape)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, JaxArray):
            out = self._jxa - other._jxa
            return JaxArray._fast_constructor(out, out.shape)
        return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, JaxArray):
            out = self._jxa @ other._jxa
            return JaxArray._fast_constructor(out, out.shape)
        return NotImplemented

    @classmethod
    @partial(jit, static_argnames=["cls", "shape"])
    def _fast_constructor(cls, array, shape):
        out = cls.__new__(cls)
        Data.__init__(out, shape)
        out._jxa = array
        return out

    def _tree_flatten(self):
        children = (self._jxa,)  # arrays / dynamic values
        aux_data = {}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        # unflatten should not check data validity
        # jax can pass tracer, object, etc.
        out = cls.__new__(cls)
        out._jxa = children[0]
        shape = getattr(out._jxa, "shape", (1,1))
        Data.__init__(out, shape)
        return out


tree_util.register_pytree_node(
    JaxArray, JaxArray._tree_flatten, JaxArray._tree_unflatten
)


def extract_jaxarray(matrix, format=None, _=None):
    """
    Return ``jaxarray`` as a jax Array.

    Parameters
    ----------
    matrix : Data
        The matrix to convert to common type.

    format : str, {"Array"}
        Type of the output.
    """
    if format not in [None, "Array", "JaxArray"]:
        raise ValueError("JaxArray can only be extracted to 'JaxArray'")
    return matrix._jxa


extract.add_specialisations([(JaxArray, extract_jaxarray)], _defer=True)
