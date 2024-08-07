import pytest
import jax.numpy as jnp
from jax import jit, grad
from qutip.entropy import entropy_vn, entropy_linear, entropy_mutual, concurrence, entropy_conditional, entangling_power, participation_ratio
import qutip.settings
import qutip_jax

qutip.settings.core["auto_real_casting"] = False
qutip_jax.use_jax_backend()
tol = 1e-6  # Tolerance for assertion

with qutip.CoreOptions(default_dtype="jax"):
    basis_0 = qutip.basis(2, 0)
    basis_1 = qutip.basis(2, 1)
    bell_state = (qutip.tensor(basis_0, basis_1) + qutip.tensor(basis_1, basis_0)).unit()
    density_matrix = bell_state * bell_state.dag()
    dm = qutip.rand_dm([5, 5], distribution="pure")

with qutip.CoreOptions(default_dtype="jax"):
    X = qutip.sigmax()
    I = qutip.qeye(2)
    CNOT = qutip.tensor(qutip.basis(2, 0) * qutip.basis(2, 0).dag(), I) + qutip.tensor(qutip.basis(2, 1) * qutip.basis(2, 1).dag(), X)

@pytest.mark.parametrize("func, name, args", [
    (entropy_vn, "entropy_vn", (density_matrix,)),
    (entropy_linear, "entropy_linear", (density_matrix,)),
    (concurrence, "concurrence", (density_matrix,)),
    (participation_ratio, "participation_ratio", (density_matrix,))
])

def test_jit(func, name, args):
    func_jit = jit(func)
    result = func(*args)
    result_jit = func_jit(*args)
    print(f"{name} (original):", result)
    print(f"{name} (JIT):", result_jit)
    assert jnp.abs(result - result_jit) < tol

@pytest.mark.parametrize("func, name, args", [
    (entropy_vn, "entropy_vn", (density_matrix,)),
    (entropy_linear, "entropy_linear", (density_matrix,)),
    (entropy_mutual, "entropy_mutual", (dm, [0], [1])),  
    (concurrence, "concurrence", (density_matrix,)),
    (entropy_conditional, "entropy_conditional", (density_matrix, 0)),
    #(entangling_power, "entangling_power", (CNOT)),
])
def test_grad(func, name, args):
    func_grad = grad(func)
    result = func(*args)
    result_grad = func_grad(*args)
    print(f"{name} (GRAD):", result_grad)
    assert result_grad is not None


