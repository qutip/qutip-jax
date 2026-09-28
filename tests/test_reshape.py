import qutip.tests.core.data.test_mathematics as test_mathematics
import qutip.tests.core.data.test_reshape as test_reshape
import qutip_jax
import pytest

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


class TestPtrace(test_mathematics.TestPtrace):
    specialisations = [
        pytest.param(
            qutip_jax.ptrace_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray,
        )
    ]
