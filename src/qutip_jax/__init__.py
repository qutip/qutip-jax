import qutip
from qutip_jax.jaxarray import JaxArray

from .convert import is_jax_array, jax_from_dense, dense_from_jax
from .version import version as __version__


# Register the data layer for JAX
qutip.data.to.add_conversions(
    [
        (JaxArray, qutip.data.Dense, jax_from_dense),
        (qutip.data.Dense, JaxArray, dense_from_jax),
    ]
)

# User friendly name for conversion with `to` or Qobj creation functions:
qutip.data.to.register_aliases(["jax", "JaxArray"], JaxArray)

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
from .qobjevo import *
from .ode import *
from .qutip_trees import *
