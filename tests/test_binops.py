import qutip.tests.core.data.test_mathematics as testing
import qutip_jax
from qutip_jax import JaxArray, JaxDia
import pytest

from . import conftest


testing._ALL_CASES = {
    JaxArray: lambda shape: [lambda: conftest._random_cplx(shape)],
    JaxDia: lambda shape: [lambda: conftest._random_dia(shape)],
}
testing._RANDOM = {
    JaxArray: lambda shape: [lambda: conftest._random_cplx(shape)],
    JaxDia: lambda shape: [lambda: conftest._random_dia(shape)],
}


class TestAdd(testing.TestAdd):
    specialisations = [
        pytest.param(
            qutip_jax.add_jaxarray,
            JaxArray,
            JaxArray,
            JaxArray,
        ),
        pytest.param(
            qutip_jax.add_jaxdia,
            JaxDia,
            JaxDia,
            JaxDia,
        ),
    ]


class TestSub(testing.TestSub):
    specialisations = [
        pytest.param(
            qutip_jax.sub_jaxarray,
            JaxArray,
            JaxArray,
            JaxArray,
        ),
        pytest.param(
            qutip_jax.sub_jaxdia,
            JaxDia,
            JaxDia,
            JaxDia,
        ),
    ]


class TestMul(testing.TestMul):
    specialisations = [
        pytest.param(qutip_jax.mul_jaxarray, JaxArray, JaxArray),
        pytest.param(qutip_jax.mul_jaxdia, JaxDia, JaxDia),
    ]


class TestMatmul(testing.TestMatmul):
    specialisations = [
        pytest.param(
            qutip_jax.matmul_jaxarray,
            JaxArray,
            JaxArray,
            JaxArray,
        ),
        pytest.param(
            qutip_jax.matmul_jaxdia,
            JaxDia,
            JaxDia,
            JaxDia,
        ),
        pytest.param(
            qutip_jax.matmul_jaxdia_jaxarray_jaxarray,
            JaxDia,
            JaxArray,
            JaxArray,
        ),
        pytest.param(
            qutip_jax.matmul_jaxarray_jaxdia_jaxarray,
            JaxArray,
            JaxDia,
            JaxArray,
        ),
    ]


class TestMultiply(testing.TestMultiply):
    specialisations = [
        pytest.param(
            qutip_jax.multiply_jaxarray,
            JaxArray,
            JaxArray,
            JaxArray,
        ),
        pytest.param(
            qutip_jax.multiply_jaxdia,
            JaxDia,
            JaxDia,
            JaxDia,
        ),
    ]


class TestKron(testing.TestKron):
    specialisations = [
        pytest.param(
            qutip_jax.kron_jaxarray,
            JaxArray,
            JaxArray,
            JaxArray,
        ),
        pytest.param(
            qutip_jax.kron_jaxdia,
            JaxDia,
            JaxDia,
            JaxDia,
        ),
    ]


class TestPow(testing.TestPow):
    specialisations = [pytest.param(qutip_jax.pow_jaxarray, JaxArray, JaxArray)]
