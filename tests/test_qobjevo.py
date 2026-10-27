import jax
import pytest
from qutip_jax.qobjevo import JaxJitCoeff


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
