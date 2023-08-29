import jax.numpy as jnp
from jax import jit
import numpy as np
import pytest
from qutip_jax.jaxdia import JaxDia, tidyup_jaxdia, clean_dia
import qutip


@pytest.mark.parametrize(
    "backend",
    [pytest.param(jnp, id="jnp"), pytest.param(np, id="np")],
)
@pytest.mark.parametrize("shape", [(1, 1), (10, 1), (3, 3), (1, 10)])
@pytest.mark.parametrize("dtype", [int, float, complex])
def test_init(backend, shape, dtype):
    """Tests creation of JaxArrays from NumPy and JAX-Numpy arrays"""
    array = np.array(np.random.rand(1, shape[1]), dtype=dtype)
    array = backend.array(array)
    jax_a = JaxDia((array, (0,)), shape=shape)
    assert isinstance(jax_a, JaxDia)
    assert jax_a.data.dtype == jnp.complex128
    assert jax_a.shape == shape


def test_jit():
    """Tests JIT of JaxArray methods"""
    arr = JaxDia((jnp.arange(3), (0,)), shape=(3, 3))

    # Some function of that we would like to JIT.
    @jit
    def func(arr):
        return arr.trace()

    tr = func(arr)
    assert isinstance(tr, jnp.ndarray)


def test_tidyup():
    big = JaxDia((jnp.arange(3), (0,)), shape=(3, 3))
    small = JaxDia((jnp.arange(3) * 1e-10, (1,)), shape=(3, 3))
    data = big + small
    assert data.num_diags == 2
    assert tidyup_jaxdia(data, 1e-5).num_diags == 1


def test_clean():
    data = clean_dia(JaxDia((jnp.ones((2, 3)), (0, -1)), shape=(3, 3)))
    assert data.offsets == (-1, 0)
    assert data.data[0, 2] == 0.0
