import pytest
import jax.numpy as jnp
from jax import jit, grad
from qutip import basis
from qutip.core.metrics import (fidelity, tracedist, bures_dist, bures_angle, 
                           hellinger_dist, hilbert_dist)
import qutip.settings
import qutip_jax

qutip.settings.core["auto_real_casting"] = False
qutip_jax.set_as_default()
tol = 1e-6  # Tolerance for assertion

with qutip.CoreOptions(default_dtype="jax"):
    rho1 = qutip.rand_dm(dimensions=5)
    rho2 = qutip.rand_dm(dimensions=5)
    ket_state = basis(2, 0)
    oper_state = qutip.rand_dm(2)

@pytest.mark.parametrize("func, name, args", [
    (fidelity, "fidelity", (rho1, rho2)),
    (tracedist, "tracedist", (rho1, rho2)),
    (bures_dist, "bures_dist", (rho1, rho2)),
    (bures_angle, "bures_angle", (rho1, rho2)),
    (hellinger_dist, "hellinger_dist", (rho1, rho2)),
    (hilbert_dist, "hilbert_dist", (rho1, rho2)),
])
def test_jit(func, name, args):
    func_jit = jit(func)
    result = func(*args)
    result_jit = func_jit(*args)
    assert jnp.abs(result - result_jit) < tol

@pytest.mark.parametrize("func, name, args", [
    (fidelity, "fidelity", (ket_state, oper_state)),
    (tracedist, "tracedist", (rho1, rho2)),
    (hellinger_dist, "hellinger_dist", (ket_state, oper_state)),
])
def test_grad(func, name, args):
    func_grad = grad(func)
    result = func(*args)
    result_grad = func_grad(*args)
    assert result_grad is not None
