import qutip_jax
import numpy as np
import pytest
import qutip.testing.mixin as testing
from qutip.testing.random_data import random_diag

@pytest.fixture
def random_generator():
    return np.random.default_rng(1234)

def _random_cplx(shape, rng):
    return qutip_jax.JaxArray(
        rng.normal(size=shape) + 1j * rng.normal(size=shape)
    )

def _random_dia(shape, rng):
    offsets = np.arange(-shape[0] + 1, shape[1])
    offsets = tuple(offsets[: min(3, shape[0] + shape[1] - 1)])
    density = len(offsets) / (shape[0] + shape[1] - 1)
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
