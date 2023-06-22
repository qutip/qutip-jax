from qutip import (
    coefficient, num, destroy, create, sesolve, MESolver, basis, settings, QobjEvo, Qobj
)
import qutip_jax
from qutip_jax.qobjevo import JaxQobjEvo
from qutip_jax.ode import DiffraxIntegrator

import pytest
import jax
import jax.numpy as jnp
import numpy as np

settings.core["default_dtype"] = "jax"


@jax.jit
def fp(t, w):
    return jax.numpy.exp(1j * t * w)


@jax.jit
def fm(t, w):
    return jax.numpy.exp(-1j * t * w)


@jax.jit
def pulse(t, A, u, sigma):
    return A * jax.numpy.exp(-(t-u)**2 / sigma) / (sigma * np.pi)**0.5


@jax.jit
def cte(t, A):
    return A


def test_ode_run():
    H = (
        num(3)
        + create(3) * coefficient(fp, args={"w": 3.1415})
        + destroy(3) * coefficient(fm, args={"w": 3.1415})
    )

    ket = basis(3)

    result = sesolve(
        H, ket, [0, 1, 2], e_ops=[num(3)], options={"method": "diffrax"}
    )
    expected = sesolve(
        H, ket, [0, 1, 2], e_ops=[num(3)], options={"method": "adams"}
    )

    np.testing.assert_allclose(result.expect[0], expected.expect[0], atol=1e-6)


def test_ode_step():
    H = (
        num(3)
        + create(3) * coefficient(fp, args={"w": 3.1415})
        + destroy(3) * coefficient(fm, args={"w": 3.1415})
    )

    c_ops = [destroy(3)]

    ket = basis(3)

    solver = MESolver(H, c_ops, options={"method": "diffrax"})
    ref_solver = MESolver(H, c_ops, options={"method": "adams"})

    solver.start(ket, 0)
    ref_solver.start(ket, 0)

    assert (solver.step(1) - ref_solver.step(1)).norm() <= 1e-6


def test_ode_grad():
    H = num(10)
    c_ops = [QobjEvo([destroy(10), cte], args={"A": 1.0})]

    options = {"method": "diffrax", "normalize_output": False}
    solver = MESolver(H, c_ops, options=options)

    def f(solver, t, A):
        result = solver.run(basis(10, 9), [0, t], e_ops=num(10), args={"A": A})
        return result.e_data[0][-1].real

    df = jax.value_and_grad(f, argnums=[1, 2])

    val, (dt, dA) = df(solver, 0.2, 0.5)

    assert val == pytest.approx(9 * np.exp(- 0.2 * 0.5))
    assert dt == pytest.approx(9 * np.exp(- 0.2 * 0.5) * -0.5)
    assert dA == pytest.approx(9 * np.exp(- 0.2 * 0.5) * -0.2)


def test_non_cplx128_JaxQobjEvo():
    op1 = Qobj(qutip_jax.zeros_jaxarray(3, 3, dtype=jnp.float64))
    op2 = Qobj(
        qutip_jax.one_element_jaxarray((3, 3), (0, 0), dtype=jnp.float64)
    )
    op3 = Qobj(qutip_jax.identity_jaxarray(3, dtype=jnp.float64))
    qevo = QobjEvo(
        [op1, [op2, pulse], [op3, cte]],
        args={"A":1.0, "u":0.1, "sigma":0.5}
    )
    jqevo = JaxQobjEvo(qevo)
    assert jqevo.batched_data.dtype == jnp.float64


def test_non_real_Diffrax():
    op1 = Qobj(qutip_jax.zeros_jaxarray(3, 3, dtype=jnp.float64))
    op2 = Qobj(
        qutip_jax.one_element_jaxarray((3, 3), (0, 0), dtype=jnp.float64)
    )
    op3 = Qobj(qutip_jax.identity_jaxarray(3, dtype=jnp.float64))
    qevo = QobjEvo(
        [op1, [op2, pulse], [op3, cte]],
        args={"A":1.0, "u":0.1, "sigma":0.5}
    )
    
    ode = DiffraxIntegrator(qevo, {})
    ode.set_state(
        0, 
        qutip_jax.one_element_jaxarray((3, 1), (2, 0), dtype=jnp.float64)
    )
    t, out = ode.integrate(0.1)
    assert out._jxa.dtype == jnp.float64
    
