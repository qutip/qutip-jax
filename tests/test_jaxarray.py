import jax.numpy as jnp
from jax import jit
import numpy as np
import pytest
from qutip_jax.jaxarray import JaxArray
import qutip


@pytest.mark.parametrize(
    "backend",
    [pytest.param(jnp, id="jnp"), pytest.param(np, id="np")],
)
@pytest.mark.parametrize("shape", [(1, 1), (10,), (3, 3), (1, 10)])
@pytest.mark.parametrize("dtype", [int, float, complex])
def test_init(backend, shape, dtype):
    """Tests creation of JaxArrays from NumPy and JAX-Numpy arrays"""
    array = np.array(np.random.rand(*shape), dtype=dtype)
    array = backend.array(array)
    jax_a = JaxArray(array)
    assert isinstance(jax_a, JaxArray)
    assert jax_a._jxa.dtype == jnp.complex128
    if len(shape) == 1:
        shape = shape + (1,)
    assert jax_a.shape == shape


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(qutip.Qobj, id="Qobj"),
        pytest.param(qutip.data.create, id="create"),
    ],
)
def test_create(build):
    """Tests creation of JaxArrays when using JAX-Numpy arrays as Qobj input"""
    data = build(jnp.linspace(0, 3, 11))
    if isinstance(data, qutip.Qobj):
        data = data.data
    assert isinstance(data, JaxArray)


def test_jit():
    """Tests JIT of JaxArray methods"""
    arr = JaxArray(jnp.linspace(0, 3, 11))

    # Some function of JaxArray that we would like to JIT.
    @jit
    def func(arr):
        return arr.trace()

    tr = func(arr)
    assert isinstance(tr, jnp.ndarray)
