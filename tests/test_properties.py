import qutip_jax
import pytest
import numbers
import numpy as np

from . import conftest


@pytest.mark.parametrize("N", (1, 10))
def test_isherm(N):
    A = conftest._random_cplx((N, N))
    A = A + A.adjoint()
    assert qutip_jax.isherm_jaxarray(A)


@pytest.mark.parametrize("shape",
    [(1, 10), (10, 1), (2, 5), (5, 2)]
)
def test_isherm_non_square(shape):
    A = conftest._random_cplx(shape)
    assert not qutip_jax.isherm_jaxarray(A)


def test_isherm_cplxdiag():
    A = conftest._random_cplx((10, 10))
    A = A + qutip_jax.JaxArray(np.diag(np.arange(10)*1j))
    assert not qutip_jax.isherm_jaxarray(A)


def test_isherm_nonherm():
    A = conftest._random_cplx((10, 10))
    A = A + A.adjoint()
    A = A + qutip_jax.JaxArray(np.diag(np.arange(9), 1))
    assert not qutip_jax.isherm_jaxarray(A)


def test_isherm_tol():
    A = conftest._random_cplx((10, 10))
    A = A + A.adjoint()
    A = A + conftest._random_cplx((10, 10)) * 1e-10
    assert qutip_jax.isherm_jaxarray(A, 1e-5)
    assert not qutip_jax.isherm_jaxarray(A, 1e-15)


@pytest.mark.parametrize("shape",
    [(1, 10), (10, 1), (2, 5), (5, 2), (5, 5)]
)
def test_iszero(shape):
    A = conftest._random_cplx(shape) * 1e-10
    assert qutip_jax.iszero_jaxarray(A, 1e-5)
    assert not qutip_jax.isherm_jaxarray(A, 1e-15)


@pytest.mark.parametrize("shape",
    [(10, 1), (2, 5), (5, 2), (5, 5)]
)
def test_isdiag(shape):
    mat = np.zeros(shape)
    # empty matrices are diagonal
    assert qutip_jax.isdiag_jaxarray(qutip_jax.JaxArray(mat))

    mat[0, 0] = 1
    assert qutip_jax.isdiag_jaxarray(qutip_jax.JaxArray(mat))

    mat[1, 0] = 1
    assert not qutip_jax.isdiag_jaxarray(qutip_jax.JaxArray(mat))
