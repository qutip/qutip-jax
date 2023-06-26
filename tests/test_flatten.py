from qutip import coefficient, num, destroy, create, rand_ket, expect
import qutip_jax
import pytest
import jax


def test_Qobj_flatten():
    @jax.jit
    def func(qobj):
        return qobj.dag()

    qobj = destroy(3, dtype="jax")
    # This call require jit to be able to jit Qobjs
    assert func(qobj) == qobj.dag()


def test_QobjEvo_flatten():
    @jax.jit
    def fp(t, w):
        return jax.numpy.exp(1j * t * w)

    @jax.jit
    def fm(t, w):
        return jax.numpy.exp(-1j * t * w)

    H = (
        num(3, dtype="jax")
        + create(3, dtype="jax") * coefficient(fp, args={"w": 3.1415})
        + destroy(3, dtype="jax") * coefficient(fm, args={"w": 3.1415})
    )

    state = rand_ket(3, dtype="jax")

    @jax.jit
    def func(t, H, state):
        return H.expect(t, state, check_real=False)

    # This call require jit to be able to jit QobjEvos
    assert func(0.5, H, state) == pytest.approx(expect(H(0.5), state))
    assert func(1.5, H, state) == pytest.approx(expect(H(1.5), state))
