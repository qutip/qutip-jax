import qutip.tests.core.data.test_mathematics as testing
import qutip_jax
import pytest

from . import conftest


testing._ALL_CASES = {
    qutip_jax.JaxArray: lambda shape: [lambda: conftest._random_cplx(shape)]
}
testing._RANDOM = {
    qutip_jax.JaxArray: lambda shape: [lambda: conftest._random_cplx(shape)]
}


class TestAdd(testing.TestAdd):
    specialisations = [
        pytest.param(
            qutip_jax.add_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray
        )
    ]

    def test_jit_from_jxa(self):
        """Test JIT of add using the _jxa array"""

        @jax.jit
        def func():
            return sigmax().to("jax").data._jxa + sigmay().to("jax").data._jxa

        assert isinstance(func(), jax.interpreters.xla.DeviceArray)

    def test_jit_from_qobj(self):
        """Test JIT of add directly using Qobj"""

        @jax.jit
        def func():
            return sigmax() + sigmay()

        assert isinstance(func(), jax.interpreters.xla.DeviceArray)


class TestSub(testing.TestSub):
    specialisations = [
        pytest.param(
            qutip_jax.sub_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray
        )
    ]

class TestMul(testing.TestMul):
    specialisations = [
        pytest.param(
            qutip_jax.mul_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray
        )
    ]

class TestMatmul(testing.TestMatmul):
    specialisations = [
        pytest.param(
            qutip_jax.matmul_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray
        )
    ]

class TestMultiply(testing.TestMultiply):
    specialisations = [
        pytest.param(
            qutip_jax.multiply_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray
        )
    ]

class TestKron(testing.TestKron):
    specialisations = [
        pytest.param(
            qutip_jax.kron_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray
        )
    ]

class TestPow(testing.TestPow):
    specialisations = [
        pytest.param(
            qutip_jax.pow_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray
        )
    ]
