import pytest
import jax.numpy as jnp
from jax import jit, grad
from qutip import bell_state
from qutip.entropy import (entropy_vn, entropy_linear, entropy_mutual, concurrence, 
                        entropy_conditional, participation_ratio)
import qutip.settings
import qutip_jax

qutip.settings.core["auto_real_casting"] = False
qutip_jax.set_as_default()
tol = 1e-6  # Tolerance for assertion

with qutip.CoreOptions(default_dtype="jax"):
    bell_state = bell_state("10")
    bell_dm = bell_state * bell_state.dag()
    dm = qutip.rand_dm([5, 5], distribution="pure")

@pytest.mark.parametrize("func, name, args", [
    (entropy_vn, "entropy_vn", (bell_dm,)),
    (entropy_linear, "entropy_linear", (bell_dm,)),
    (concurrence, "concurrence", (bell_dm,)),
    (participation_ratio, "participation_ratio", (bell_dm,))
])

def test_jit(func, name, args):
    func_jit = jit(func)
    result = func(*args)
    result_jit = func_jit(*args)
    assert jnp.abs(result - result_jit) < tol

@pytest.mark.parametrize("func, name, args", [
    (entropy_vn, "entropy_vn", (bell_dm,)),
    (entropy_linear, "entropy_linear", (bell_dm,)),
    (entropy_mutual, "entropy_mutual", (dm, [0], [1])),  
    (concurrence, "concurrence", (bell_dm,)),
    (entropy_conditional, "entropy_conditional", (bell_dm, 0)),
])
def test_grad(func, name, args):
    func_grad = grad(func)
    result_grad = func_grad(*args)
    assert result_grad is not None


