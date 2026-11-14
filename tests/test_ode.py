from qutip import (
    SESolver,
    coefficient,
    num,
    destroy,
    create,
    sesolve,
    MESolver,
    basis,
    settings,
    QobjEvo,
    CoreOptions,
)
import qutip_jax
import pytest
import jax
import numpy as np


@jax.jit
def fp(t, w):
    return jax.numpy.exp(1j * t * w)


@jax.jit
def fm(t, w):
    return jax.numpy.exp(-1j * t * w)


@jax.jit
def pulse(t, A, u, sigma):
    return A * jax.numpy.exp(-((t - u) ** 2) / sigma) / (sigma * np.pi) ** 0.5


@jax.jit
def cte(t, A):
    return A


# diffrax use clip with deprecated parameters
@pytest.mark.filterwarnings("ignore:Passing arguments 'a'")
@pytest.mark.parametrize("dtype", ("jax", "jaxdia"))
def test_ode_run(dtype):
    with CoreOptions(default_dtype=dtype):
        H = (
            num(3)
            + create(3) * coefficient(fp, args={"w": 3.1415})
            + destroy(3) * coefficient(fm, args={"w": 3.1415})
        )

    ket = basis(3, dtype="jax")

    result = sesolve(
        H, ket, [0, 1, 2], e_ops=[num(3)], options={"method": "diffrax"}
    )
    expected = sesolve(
        H, ket, [0, 1, 2], e_ops=[num(3)], options={"method": "adams"}
    )

    np.testing.assert_allclose(result.expect[0], expected.expect[0], atol=1e-6)


@pytest.mark.filterwarnings("ignore:Passing arguments 'a'")
@pytest.mark.parametrize("dtype", ("jax", "jaxdia"))
def test_ode_step(dtype):
    with CoreOptions(default_dtype=dtype):
        H = (
            num(3)
            + create(3) * coefficient(fp, args={"w": 3.1415})
            + destroy(3) * coefficient(fm, args={"w": 3.1415})
        )

        c_ops = [destroy(3)]

    ket = basis(3, dtype="jax")

    solver = MESolver(H, c_ops, options={"method": "diffrax"})
    ref_solver = MESolver(H.to("Dense"), c_ops, options={"method": "adams"})

    solver.start(ket, 0)
    ref_solver.start(ket, 0)

    assert (solver.step(1) - ref_solver.step(1)).norm() <= 1e-6


@pytest.mark.filterwarnings("ignore:Passing arguments 'a'")
@pytest.mark.parametrize("dtype", ("jax", "jaxdia"))
def test_ode_grad(dtype):
    with CoreOptions(default_dtype=dtype):
        H = num(10)
        c_ops = [QobjEvo([destroy(10), cte], args={"A": 1.0})]

    options = {"method": "diffrax", "normalize_output": False}
    solver = MESolver(H, c_ops, options=options)

    def f(solver, t, A):
        result = solver.run(
            basis(10, 9, dtype="jax"),
            [0, t],
            e_ops=num(10, dtype="jaxdia"),
            args={"A": A}
        )
        return result.e_data[0][-1].real

    df = jax.value_and_grad(f, argnums=[1, 2])

    val, (dt, dA) = df(solver, 0.2, 0.5)

    assert val == pytest.approx(9 * np.exp(-0.2 * 0.5**2))
    assert dt == pytest.approx(9 * np.exp(-0.2 * 0.5**2) * -0.5**2)
    assert dA == pytest.approx(9 * np.exp(-0.2 * 0.5**2) * -0.2)


@pytest.mark.filterwarnings("ignore:Passing arguments 'a'")
def test_ode_integrator_flatten():
    solver = SESolver(num(3), options={"method": "diffrax"})
    integrator = solver._integrator
    integrator.set_state(0., basis(3, 2, dtype="jax").data)

    leaves, treedef = jax.tree_util.tree_flatten(integrator)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    # The rebuilt integrator must still be able to integrate.
    t, state = integrator.integrate(0.5)
    t_rebuilt, state_rebuilt = rebuilt.integrate(0.5)

    assert t_rebuilt == t
    np.testing.assert_allclose(
        state_rebuilt.to_array(), state.to_array(), atol=1e-6
    )
