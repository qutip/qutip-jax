from qutip_jax.jaxarray import JaxArray
import jax.numpy as jnp

def test_create():
    assert isinstance(JaxArray(jnp.linspace(0, 3, 11)), JaxArray)
