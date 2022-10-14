import qutip.tests.core.data.test_mathematics as test_mathematics
import qutip.tests.core.data.test_reshape as test_reshape
import qutip.tests.core.data.test_ptrace as test_ptrace
import qutip_jax
import pytest

from . import conftest


test_mathematics._ALL_CASES = {
    qutip_jax.JaxArray: lambda shape: [lambda: conftest._random_cplx(shape)]
}
test_mathematics._RANDOM = {
    qutip_jax.JaxArray: lambda shape: [lambda: conftest._random_cplx(shape)]
}


class TestSplitColumns(test_reshape.TestSplitColumns):
    specialisations = [
        pytest.param(
            qutip_jax.split_columns_jaxarray,
            qutip_jax.JaxArray,
            list,
        )
    ]


class TestColumnStack(test_reshape.TestColumnStack):
    specialisations = [
        pytest.param(
            qutip_jax.column_stack_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray,
        )
    ]


class TestColumnUnstack(test_reshape.TestColumnUnstack):
    specialisations = [
        pytest.param(
            qutip_jax.column_unstack_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray,
        )
    ]


class TestReshape(test_reshape.TestReshape):
    specialisations = [
        pytest.param(
            qutip_jax.reshape_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray,
        )
    ]


class TestPtrace(test_ptrace.TestPtrace):
    specialisations = [
        pytest.param(
            qutip_jax.ptrace_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray,
        )
    ]
