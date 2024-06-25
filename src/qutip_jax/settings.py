import jax.numpy as jnp
from qutip import settings

__all__ = ["use_jax_backend"]

def use_jax_backend():
    settings.core['numpy_backend'] = jnp
