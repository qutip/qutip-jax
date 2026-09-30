from jax import random
import qutip_jax
import numpy as np
from qutip.tests.core.data.conftest import CORRECT_CASES, WRONG_CASES
from qutip.tests.conftest import random_generator

key = random.PRNGKey(1234)


def _random_cplx(shape):
    return qutip_jax.JaxArray(
        random.normal(key, shape) + 1j * random.normal(key, shape)
    )


def _random_dia(shape):
    offsets = np.arange(-shape[0] + 1, shape[1])
    np.random.shuffle(offsets)
    offsets = tuple(offsets[: min(3, shape[0] + shape[1] - 1)])
    data_shape = len(offsets), shape[1]
    data = np.random.rand(*data_shape) + 1j * np.random.rand(*data_shape)
    return qutip_jax.JaxDia((data, offsets), shape=shape)

CORRECT_CASES.update({
    qutip_jax.JaxArray: lambda shape: [lambda rng: _random_cplx(shape)],
    qutip_jax.JaxDia: lambda shape: [lambda rng: _random_dia(shape)],
})
WRONG_CASES.update({
    qutip_jax.JaxArray: lambda shape: [lambda rng: _random_cplx(shape)],
    qutip_jax.JaxDia: lambda shape: [lambda rng: _random_dia(shape)],
})
