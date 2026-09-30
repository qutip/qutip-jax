import qutip.tests.core.data.test_mathematics as testing
import qutip_jax
from qutip_jax import JaxArray, JaxDia
import pytest
from qutip.core import data

from . import conftest

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


def _invertible_jaxarray(shape):
    # Add a diagonal so `matrix` is not singular
    matrix = conftest._random_cplx(shape)
    return data.add(
        matrix,
        data.diag([2.0 * shape[0]] * shape[0], shape=shape, dtype="JaxArray"),
    )


class TestInv(testing.TestInv):
    specialisations = [pytest.param(qutip_jax.inv_jaxarray, JaxArray, JaxArray)]
    correct_cases = {
        JaxArray: lambda shape: [lambda rng: _invertible_jaxarray(shape)],
    }
    wrong_cases = {
        JaxArray: lambda shape: [lambda rng: conftest._random_cplx(shape)],
    }


class TestSqrtm(testing.TestSqrtm):
    specialisations = [
        pytest.param(qutip_jax.sqrtm_jaxarray, JaxArray, JaxArray)
    ]


class TestProject(testing.TestProject):
    specialisations = [
        pytest.param(qutip_jax.project_jaxarray, JaxArray, JaxArray)
    ]
