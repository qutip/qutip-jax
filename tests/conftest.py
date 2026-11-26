import qutip_jax
import jax
import numpy as np
import pytest
import qutip.testing.mixin as testing
from qutip.testing.random_data import random_diag

@pytest.fixture
def random_generator():
    return np.random.default_rng(1234)

def _jax_key(rng):
    # Derive a JAX PRNG key from the seeded numpy generator, so the tests
    # stay reproducible while the data is drawn from jax.random.
    return jax.random.key(int(rng.integers(2**31)))

def _random_cplx(shape, rng):
    key_real, key_imag = jax.random.split(_jax_key(rng))
    return qutip_jax.JaxArray(
        jax.random.normal(key_real, shape)
        + 1j * jax.random.normal(key_imag, shape)
    )

def _random_dia(shape, rng):
    num_offsets = shape[0] + shape[1] - 1
    density = min(0.5, 3 / num_offsets)
    matrix = random_diag(shape, density=density, gen=rng)
    return qutip_jax.jaxdia_from_dia(matrix)

testing.CORRECT_CASES.update({
    qutip_jax.JaxArray: lambda shape: [lambda rng: _random_cplx(shape, rng)],
    qutip_jax.JaxDia: lambda shape: [lambda rng: _random_dia(shape, rng)],
})
testing.WRONG_CASES.update({
    qutip_jax.JaxArray: lambda shape: [lambda rng: _random_cplx(shape, rng)],
    qutip_jax.JaxDia: lambda shape: [lambda rng: _random_dia(shape, rng)],
})
