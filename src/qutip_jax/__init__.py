import qutip
from qutip_jax.jaxarray import JaxArray, JaxDia

from .convert import (
    is_jax_array,
    jaxarray_from_dense,
    dense_from_jaxarray,
    jaxarray_from_jaxdia,
    jaxdia_from_jaxarray,
)
from .version import version as __version__


# Register the data layer for JAX
qutip.data.to.add_conversions(
    [
        (JaxArray, qutip.data.Dense, jaxarray_from_dense),
        (qutip.data.Dense, JaxArray, dense_from_jaxarray),
        (JaxArray, JaxDia, jaxarray_from_jaxdia),
        (JaxDia, JaxArray, jaxdia_from_jaxarray),
    ]
)

# User friendly name for conversion with `to` or Qobj creation functions:
qutip.data.to.register_aliases(["jax", "JaxArray"], JaxArray)
qutip.data.to.register_aliases(["jaxdia", "JaxDia"], JaxDia)

qutip.data.create.add_creators(
    [
        (is_jax_array, JaxArray, 85),
    ]
)


from .binops import *
from .unary import *
from .permute import *
from .reshape import *
from . import norm
from .measurements import *
from .properties import *
from .create import *
