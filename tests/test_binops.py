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
