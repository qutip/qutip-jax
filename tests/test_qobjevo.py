import jax
import pytest
import jax.numpy as jnp
import numpy as np
from qutip_jax.qobjevo import JaxJitCoeff,JaxQobjEvo
from qutip import QobjEvo, coefficient, create, num

def _coeff(value):
    return JaxJitCoeff(jax.jit(lambda t, **_: value))


def test_jitcoeff_mul():
    a = _coeff(2.0)
    b = _coeff(3.0)
    assert (a * b)(0.0) == pytest.approx(6.0)


def test_jitcoeff_add():
    a = _coeff(2.0)
    b = _coeff(3.0)
    assert (a + b)(0.0) == pytest.approx(5.0)

@jax.jit
def _fp(t, w):
    return jnp.exp(1j * t * w)


def test_jaxqobjevo_call_constant():
    H = QobjEvo(num(3, dtype="jax"))
    ev = JaxQobjEvo(H)
    np.testing.assert_allclose(ev(0.7).full(), H(0.7).full(), atol=1e-6)


def test_jaxqobjevo_call_arguments():
    H = QobjEvo(
        [num(3, dtype="jax"),
         [create(3, dtype="jax"), coefficient(_fp, args={"w": 3.1415})]]
    )
    ev = JaxQobjEvo(H)
    # default argument
    
    np.testing.assert_allclose(ev(0.7).full(), H(0.7).full(), atol=1e-6)
    # argument given as keyword
    
    np.testing.assert_allclose(
        ev(0.7, w=0.5).full(), H(0.7, w=0.5).full(), atol=1e-6
    )
    
    # argument given as a dict
    np.testing.assert_allclose(
        ev(0.7, {"w": 0.5}).full(), H(0.7, w=0.5).full(), atol=1e-6
    )
