from jax import random
import qutip_jax

key = random.PRNGKey(1234)

def _random_cplx(shape):
    return qutip_jax.JaxArray._fast_constructor(
        random.normal(key, shape) + 1j*random.normal(key, shape)
    )
