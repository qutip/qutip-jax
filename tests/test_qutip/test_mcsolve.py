import pytest
import jax
import jax.numpy as jnp
import qutip as qt
import qutip_jax as qjax
from qutip import mcsolve
from functools import partial

# Use JAX backend for QuTiP
qjax.set_as_default()

# Define time-dependent functions
@jax.jit
def H_1_coeff(t, omega):
    return 2.0 * jnp.pi * 0.25 * jnp.cos(2.0 * omega * t)

# Test setup for gradient calculation
def setup_system(size=2):
    a = qt.tensor(qt.destroy(size), qt.qeye(2)).to('jaxdia')
    sm = qt.qeye(size).to('jaxdia') & qt.sigmax().to('jaxdia')

    # Define the Hamiltonian
    H_0 = 2.0 * jnp.pi * a.dag() * a + 2.0 * jnp.pi * sm.dag() * sm
    H_1_op = sm * a.dag() + sm.dag() * a

    H = [H_0, [H_1_op, qt.coefficient(H_1_coeff, args={"omega": 1.0})]]

    state = qt.basis(size, size - 1).to('jax') & qt.basis(2, 1).to('jax')

    # Define collapse operators and observables
    c_ops = [jnp.sqrt(0.1) * a]
    e_ops = [a.dag() * a, sm.dag() * sm]

    # Time list
    tlist = jnp.linspace(0.0, 1.0, 101)

    return H, state, tlist, c_ops, e_ops

# Function for which we want to compute the gradient
def f(omega, H, state, tlist, c_ops, e_ops):
    result = mcsolve(
        H, state, tlist, c_ops, e_ops=e_ops, ntraj=10,
        args={"omega": omega},
        options={"method": "diffrax"}
    )

    return result.expect[0][-1].real

# Pytest test case for gradient computation
@pytest.mark.parametrize("omega_val", [2.0])
def test_gradient_mcsolve(omega_val):
    H, state, tlist, c_ops, e_ops = setup_system(size=10)

    # Compute the gradient with respect to omega
    grad_func = jax.grad(lambda omega: f(omega, H, state, tlist, c_ops, e_ops))
    gradient = grad_func(omega_val)

    # Check if the gradient is not None and has the correct shape
    assert gradient is not None
    assert gradient.shape == ()
    assert jnp.isfinite(gradient)
