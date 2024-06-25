import qutip
from .jaxarray import JaxArray
from .jaxdia import JaxDia

from .convert import *
from .version import version as __version__


# Register the data layer for JAX
qutip.data.to.add_conversions(
    [
        (JaxArray, qutip.data.Dense, jaxarray_from_dense),
        (qutip.data.Dense, JaxArray, dense_from_jaxarray, 2),
        (JaxArray, JaxDia, jaxarray_from_jaxdia),
        (JaxDia, JaxArray, jaxdia_from_jaxarray, 1.2),
        (qutip.data.Dia, JaxDia, dia_from_jaxdia, 2),
        (JaxDia, qutip.data.Dia, jaxdia_from_dia),
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
del is_jax_array

from .binops import *
from .unary import *
from .permute import *
from .reshape import *
from . import norm
from .settings import *
from .measurements import *
from .properties import *
from .linalg import *
from .create import *
from .qobjevo import *
from .ode import *
from .qutip_trees import *
