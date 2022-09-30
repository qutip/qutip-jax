import jax
import jax.numpy as jnp
from jax import jit

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

import qutip_jax
from qutip_jax.jaxarray import JaxArray
import qutip


@pytest.mark.parametrize(
    "backend",
    [pytest.param(jnp, id="jnp"), pytest.param(np, id="np")],
)
@pytest.mark.parametrize("shape", [(1,1), (10,), (3, 3), (1, 10)])
@pytest.mark.parametrize("dtype", [int, float, complex])
def test_init(backend, shape, dtype):
    """Tests creation of JaxArrays from NumPy and JAX-Numpy arrays"""
    array = np.array(np.random.rand(*shape), dtype=dtype)
    array = backend.array(array)
    jax_a = JaxArray(array)
    assert isinstance(jax_a, JaxArray)
    assert jax_a._jxa.dtype == jax.numpy.complex128
    if len(shape) == 1:
        shape = shape + (1,)
    assert jax_a.shape == shape


@pytest.mark.parametrize("build",
    [
        pytest.param(qutip.Qobj, id="Qobj"),
        pytest.param(qutip.data.create, id="create")
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
    assert isinstance(tr, jax.interpreters.xla.DeviceArray)


@pytest.mark.parametrize("to_",
    [
        pytest.param(qutip.data.Dense, id="to Dense type"),
        pytest.param(qutip.data.CSR, id="to CSR type"),
    ],
)
@pytest.mark.parametrize("back_",
    [
        pytest.param("jax", id="from str (1)"),
        pytest.param("JaxArray", id="from str (2)"),
        pytest.param(JaxArray, id="from type"),
    ],
)
def test_convert_explicit(to_, back_):
    """ Test that it can convert to and from other types """
    arr = JaxArray(jnp.linspace(0, 3, 11))
    converted = qutip.data.to(to_, arr)
    assert isinstance(converted, to_)
    back = qutip.data.to[back_](converted)
    assert isinstance(back, JaxArray)


def test_convert():
    """Tests if the conversions from Qobj to JaxArray work"""
    ones = jnp.ones((3, 3))
    qobj = qutip.Qobj(ones)
    prod = qobj * jnp.array([0.5])
    assert_array_almost_equal(prod.data.to_array(), ones * jnp.array([0.5]))

    sx = qutip.qeye(5, dtype="csr")
    assert isinstance(sx.data, qutip.core.data.CSR)
    assert isinstance(sx.to('jax').data, JaxArray)

    sx = qutip.qeye(5, dtype="JaxArray")
    assert isinstance(sx.data, JaxArray)
