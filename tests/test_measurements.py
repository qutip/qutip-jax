import qutip.testing.mixin as testing
import qutip_jax
import pytest
import numbers

class TestExpect(testing.TestExpect):
    specialisations = [
        pytest.param(
            qutip_jax.expect_jaxdia_jaxarray,
            qutip_jax.JaxDia,
            qutip_jax.JaxArray,
            object,
        ),
        pytest.param(
            qutip_jax.expect_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray,
            object,
        ),
    ]


class TestExpectSuper(testing.TestExpectSuper):
    specialisations = [
        pytest.param(
            qutip_jax.expect_super_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray,
            object,
        ),
        pytest.param(
            qutip_jax.expect_super_jaxdia_jaxarray,
            qutip_jax.JaxDia,
            qutip_jax.JaxArray,
            object,
        ),
    ]


class TestInner(testing.TestInner):
    specialisations = [
        pytest.param(
            qutip_jax.inner_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray,
            object,
        )
    ]


class TestInnerOp(testing.TestInnerOp):
    specialisations = [
        pytest.param(
            qutip_jax.inner_op_jaxarray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray,
            qutip_jax.JaxArray,
            object,
        )
    ]


class TestTrace(testing.TestTrace):
    specialisations = [
        pytest.param(
            qutip_jax.trace_jaxarray,
            qutip_jax.JaxArray,
            object,
        )
    ]


class TestTrace_oper_ket(testing.TestTrace_oper_ket):
    specialisations = [
        pytest.param(
            qutip_jax.trace_oper_ket_jaxarray,
            qutip_jax.JaxArray,
            object,
        )
    ]
