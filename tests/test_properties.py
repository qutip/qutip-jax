import qutip.tests.core.data.test_mathematics as testing
import qutip_jax
import pytest
import numbers
import numpy as np
import qutip.core.data as _data

from . import conftest


testing._ALL_CASES = {
    qutip_jax.JaxArray: lambda shape: [lambda: conftest._random_cplx(shape)],
    qutip_jax.JaxDia: lambda shape: [lambda: conftest._random_dia(shape)],
}
testing._RANDOM = {
    qutip_jax.JaxArray: lambda shape: [lambda: conftest._random_cplx(shape)],
    qutip_jax.JaxDia: lambda shape: [lambda: conftest._random_dia(shape)],
}


@pytest.mark.parametrize("N", (1, 10))
@pytest.mark.parametrize(
    ["func", "maker"],
    [
        (qutip_jax.isherm_jaxarray, conftest._random_cplx),
        (qutip_jax.isherm_jaxdia, conftest._random_dia),
    ],
)
def test_isherm(func, maker, N):
    A = maker((N, N))
    A = A + A.adjoint()
    assert func(A)


@pytest.mark.parametrize("shape", [(1, 10), (10, 1), (2, 5), (5, 2)])
@pytest.mark.parametrize(
    ["func", "maker"],
    [
        (qutip_jax.isherm_jaxarray, conftest._random_cplx),
        (qutip_jax.isherm_jaxdia, conftest._random_dia),
    ],
)
def test_isherm_non_square(func, maker, shape):
    A = maker(shape)
    assert not func(A)


def test_isherm_nonherm():
    A = conftest._random_cplx((10, 10))
    A = A + qutip_jax.JaxArray(np.diag(np.arange(10) * 1j))
    assert not qutip_jax.isherm_jaxarray(A)


def test_isherm_nonherm_dia():
    A = conftest._random_cplx((10, 10))
    A = A + qutip_jax.identity_jaxdia(10) * 1j
    assert not qutip_jax.isherm_jaxarray(A)


@pytest.mark.parametrize(
    ["func", "maker"],
    [
        (qutip_jax.isherm_jaxarray, conftest._random_cplx),
        (qutip_jax.isherm_jaxdia, conftest._random_dia),
    ],
)
def test_isherm_tol(func, maker):
    A = maker((10, 10))
    A = A + A.adjoint()
    A = A + maker((10, 10)) * 1e-10
    assert func(A, 1e-5)
    assert not func(A, 1e-15)


@pytest.mark.parametrize("shape", [(1, 10), (10, 1), (2, 5), (5, 2), (5, 5)])
@pytest.mark.parametrize(
    ["func", "maker"],
    [
        (qutip_jax.iszero_jaxarray, conftest._random_cplx),
        (qutip_jax.iszero_jaxdia, conftest._random_dia),
    ],
)
def test_iszero(func, maker, shape):
    A = maker(shape) * 1e-10
    assert func(A, 1e-5)
    assert not func(A, 1e-15)


@pytest.mark.parametrize("shape", [(10, 1), (2, 5), (5, 2), (5, 5)])
def test_isdiag(shape):
    mat = np.zeros(shape)
    # empty matrices are diagonal
    assert qutip_jax.isdiag_jaxarray(qutip_jax.JaxArray(mat))

    mat[0, 0] = 1
    assert qutip_jax.isdiag_jaxarray(qutip_jax.JaxArray(mat))

    mat[1, 0] = 1
    assert not qutip_jax.isdiag_jaxarray(qutip_jax.JaxArray(mat))


@pytest.mark.parametrize("shape", [(10, 1), (2, 5), (5, 2), (5, 5)])
def test_isdiag_dia(shape):
    mat = np.zeros(shape)
    # empty matrices are diagonal
    assert qutip_jax.isdiag_jaxdia(_data.to("jaxdia", qutip_jax.JaxArray(mat)))

    mat[0, 0] = 1
    assert qutip_jax.isdiag_jaxdia(_data.to("jaxdia", qutip_jax.JaxArray(mat)))

    mat[1, 0] = 1
    assert not qutip_jax.isdiag_jaxdia(
        _data.to("jaxdia", qutip_jax.JaxArray(mat))
    )


class TestTrace(testing.TestTrace):
    specialisations = [
        pytest.param(
            qutip_jax.trace_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray,
            object,
        ),
        pytest.param(
            qutip_jax.trace_jaxdia,
            qutip_jax.JaxDia,
            qutip_jax.JaxDia,
            object,
        ),
    ]


class TestTrace_oper_ket(testing.TestTrace_oper_ket):
    specialisations = [
        pytest.param(
            qutip_jax.trace_oper_ket_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray,
            object,
        )
    ]
