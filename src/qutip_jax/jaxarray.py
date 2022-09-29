from qutip.core.data.base import Data
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

class JaxArray(Data):
    def __init__(self, data, shape=None, copy=None):
        data = jnp.array(data, dtype=jnp.complex128)

        if shape is None:
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

        if prod(shape) != prod(data.shape):
            raise ValueError("Shape of data does not match argument.")

        if copy:
            # Since jax's arrays are immutable, we could probably skip this.
            data = data.copy()
        self.data = jnp.reshape(data, shape)
        super.__init__(shape)

    def copy(self):
        return self.__class__(self.data, copy=True)

    def to_array(self):
        return np.array(self.data)

    def conj(self):
        return self.__class__(self.data.conj())

    def transpose(self):
        return self.__class__(self.data.T)

    def adjoint(self):
        return self.__class__(self.data.T.conj())

    def trace(self):
        return jnp.trace(self.data)
