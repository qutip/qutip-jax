import qutip.tests.core.data.test_mathematics as testing
import qutip_jax
from qutip_jax import JaxArray
import pytest
from qutip.core import data

from . import conftest


testing._ALL_CASES = {
    qutip_jax.JaxArray: lambda shape: [lambda: conftest._random_cplx(shape)]
}
testing._RANDOM = {
    qutip_jax.JaxArray: lambda shape: [lambda: conftest._random_cplx(shape)]
}


class TestNeg(testing.TestNeg):
    specialisations = [
        pytest.param(qutip_jax.neg_jaxarray, JaxArray, JaxArray),
    ]


class TestAdjoint(testing.TestAdjoint):
    specialisations = [
        pytest.param(qutip_jax.adjoint_jaxarray, JaxArray, JaxArray)
    ]


class TestConj(testing.TestConj):
    specialisations = [
        pytest.param(qutip_jax.conj_jaxarray, JaxArray, JaxArray)
    ]


class TestTranspose(testing.TestTranspose):
    specialisations = [
        pytest.param(
            qutip_jax.transpose_jaxarray, JaxArray, JaxArray)
    ]


class TestExpm(testing.TestExpm):
    specialisations = [
        pytest.param(
            qutip_jax.expm_jaxarray, JaxArray, JaxArray)
    ]


def _inv_jax(matrix):
    # Add a diagonal so `matrix` is not singular
    return qutip_jax.inv_jaxarray(
        data.add(
            matrix,
            data.diag([1.1]*matrix.shape[0], shape=matrix.shape, dtype='JaxArray')
        )
    )

class TestInv(testing.TestInv):
    specialisations = [
        pytest.param(
            _inv_jax, JaxArray, JaxArray)
    ]


class TestProject(testing.TestProject):
    specialisations = [
        pytest.param(
            qutip_jax.project_jaxarray, JaxArray, JaxArray)
    ]
