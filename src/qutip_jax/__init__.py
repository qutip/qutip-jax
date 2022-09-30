import qutip
from qutip_jax.jaxarray import JaxArray

from .convert import jax_array_from_dense, jax_array_to_dense, is_jax_array
from .version import version as __version__


# Register the data layer for JAX
qutip.data.to.add_conversions(
    [
        (JaxArray, qutip.data.Dense, jax_array_from_dense),
        (qutip.data.Dense, JaxArray, jax_array_to_dense),
    ]
)

# User friendly name for conversion with `to` or Qobj creation functions:
qutip.data.to.register_aliases(["JaxArray"], JaxArray)

qutip.data.create.add_creators(
    [
        (is_jax_array, JaxArray, 85),
    ]
)
