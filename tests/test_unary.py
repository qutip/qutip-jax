import qutip.tests.core.data.test_mathematics as testing
import qutip_jax
from qutip_jax import JaxArray, JaxDia
import pytest
from qutip.core import data

from . import conftest


testing._ALL_CASES = {
    JaxArray: lambda shape: [lambda: conftest._random_cplx(shape)],
    JaxDia: lambda shape: [lambda: conftest._random_dia(shape)],
}
testing._RANDOM = {
    JaxArray: lambda shape: [lambda: conftest._random_cplx(shape)],
    JaxDia: lambda shape: [lambda: conftest._random_dia(shape)],
}


class TestNeg(testing.TestNeg):
    specialisations = [
        pytest.param(qutip_jax.neg_jaxarray, JaxArray, JaxArray),
        pytest.param(qutip_jax.neg_jaxdia, JaxDia, JaxDia),
    ]


class TestAdjoint(testing.TestAdjoint):
    specialisations = [
        pytest.param(qutip_jax.adjoint_jaxarray, JaxArray, JaxArray),
        pytest.param(lambda mat: mat.adjoint(), JaxArray, JaxArray),
        pytest.param(qutip_jax.adjoint_jaxdia, JaxDia, JaxDia),
    ]


class TestConj(testing.TestConj):
    specialisations = [
        pytest.param(qutip_jax.conj_jaxarray, JaxArray, JaxArray),
        pytest.param(lambda mat: mat.conj(), JaxArray, JaxArray),
        pytest.param(qutip_jax.conj_jaxdia, JaxDia, JaxDia),
    ]


class TestTranspose(testing.TestTranspose):
    specialisations = [
        pytest.param(qutip_jax.transpose_jaxarray, JaxArray, JaxArray),
        pytest.param(lambda mat: mat.transpose(), JaxArray, JaxArray),
        pytest.param(qutip_jax.transpose_jaxdia, JaxDia, JaxDia),
    ]


class TestExpm(testing.TestExpm):
    specialisations = [
        pytest.param(qutip_jax.expm_jaxarray, JaxArray, JaxArray)
    ]


def _inv_jax(matrix):
    # Add a diagonal so `matrix` is not singular
    return qutip_jax.inv_jaxarray(
        data.add(
            matrix,
            data.diag(
                [1.1] * matrix.shape[0], shape=matrix.shape, dtype="JaxArray"
            ),
        )
    )


class TestInv(testing.TestInv):
    specialisations = [pytest.param(_inv_jax, JaxArray, JaxArray)]


class TestSqrtm(testing.TestSqrtm):
    specialisations = [
        pytest.param(qutip_jax.sqrtm_jaxarray, JaxArray, JaxArray)
    ]


class TestProject(testing.TestProject):
    specialisations = [
        pytest.param(qutip_jax.project_jaxarray, JaxArray, JaxArray)
    ]
