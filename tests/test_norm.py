import qutip.testing.stat_mixin as testing_norm
import qutip_jax
import pytest
import numbers

class TestOneNorm(testing_norm.TestOneNorm):
    specialisations = [
        pytest.param(
            qutip_jax.norm.one_jaxarray,
            qutip_jax.JaxArray,
            object,
        )
    ]


class TestFrobeniusNorm(testing_norm.TestFrobeniusNorm):
    specialisations = [
        pytest.param(
            qutip_jax.norm.frobenius_jaxarray,
            qutip_jax.JaxArray,
            object,
        )
    ]


class TestMaxNorm(testing_norm.TestMaxNorm):
    specialisations = [
        pytest.param(
            qutip_jax.norm.max_jaxarray,
            qutip_jax.JaxArray,
            object,
        )
    ]


class TestL2Norm(testing_norm.TestL2Norm):
    specialisations = [
        pytest.param(
            qutip_jax.norm.l2_jaxarray,
            qutip_jax.JaxArray,
            object,
        )
    ]


class TestTraceNorm(testing_norm.TestTraceNorm):
    specialisations = [
        pytest.param(
            qutip_jax.norm.trace_jaxarray,
            qutip_jax.JaxArray,
            object,
        )
    ]
