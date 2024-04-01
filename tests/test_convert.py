import jax
import jax.numpy as jnp
from jax import jit

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

import qutip_jax
from qutip_jax import JaxArray, JaxDia
import qutip


@pytest.mark.parametrize(
    "to_",
    [
        pytest.param(qutip.data.Dense, id="to Dense type"),
        pytest.param(qutip.data.CSR, id="to CSR type"),
        pytest.param(JaxDia, id="to JaxDia type"),
    ],
)
@pytest.mark.parametrize(
    "back_",
    [
        pytest.param("jax", id="from str (1)"),
        pytest.param("JaxArray", id="from str (2)"),
        pytest.param(JaxArray, id="from type"),
    ],
)
def test_convert_explicit_jaxarray(to_, back_):
    """Test that it can convert to and from other types"""
    arr = JaxArray(jnp.arange(0, 3, 11))
    converted = qutip.data.to(to_, arr)
    assert isinstance(converted, to_)
    back = qutip.data.to[back_](converted)
    assert isinstance(back, JaxArray)
    assert back == arr


@pytest.mark.parametrize(
    "to_",
    [
        pytest.param(qutip.data.Dense, id="to Dense type"),
        pytest.param(qutip.data.CSR, id="to CSR type"),
        pytest.param(qutip.data.Dia, id="to Dia type"),
        pytest.param(JaxArray, id="to JaxArray type"),
    ],
)
@pytest.mark.parametrize(
    "back_",
    [
        pytest.param("JaxDia", id="from str"),
        pytest.param(JaxDia, id="from type"),
    ],
)
def test_convert_explicit_jaxdia(to_, back_):
    """Test that it can convert to and from other types"""
    arr = JaxDia((jnp.arange(3), (0,)), shape=(3, 3))
    converted = qutip.data.to(to_, arr)
    assert isinstance(converted, to_)
    back = qutip.data.to[back_](converted)
    assert isinstance(back, JaxDia)
    assert back == arr


def test_convert():
    """Tests if the conversions from Qobj to JaxArray work"""
    ones = jnp.ones((3, 3))
    qobj = qutip.Qobj(ones)
    prod = qobj * jnp.array(0.5)
    assert_array_almost_equal(prod.data.to_array(), ones * jnp.array([0.5]))

    sx = qutip.qeye(5, dtype="csr")
    assert isinstance(sx.data, qutip.core.data.CSR)
    assert isinstance(sx.to("jax").data, JaxArray)

    sx = qutip.qeye(5, dtype="JaxArray")
    assert isinstance(sx.data, JaxArray)

    sx = qutip.qeye(5, dtype="JaxDia")
    assert isinstance(sx.data, JaxDia)
