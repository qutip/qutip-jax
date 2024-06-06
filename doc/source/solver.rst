.. _qtjax_solver:

*****************************
Jax Backend in QuTiP's Solver
*****************************


.. _mesolve:

Using Jax in ``mesolve``
========================


To use JAX's arrays in QuTiP's solvers such as ``mesolve``, two conditions must be met:

1. All opertors and states must use JAX arrays.
2. An ODE integrator supporting Jax must be used.
   Currently, an ODE solver from the diffrax project is made available when importing qutip-jax.
   
The following code shows an example of how to use JAX:

.. code-block::

    import qutip
    import qutip_jax

    with qutip.CoreOptions(default_dtype="jax"):
        H = qutip.rand_herm(5)
        c_ops = [qutip.destroy(5)]
        rho0 = qutip.basis(5, 4)

    result = qutip.mesolve(H, rho0, [0, 1], c_ops=c_ops, options={"method": "diffrax"})

Note that while running the above code on the GPU, the ``"normalize_output"`` option should be set to ``False``, as Schur decomposition is only supported in the CPU currently.


.. _mcsolve:

Using Jax in ``mcsolve``
========================

Similar to ``mesolve``, the JAX backend can be used with ``mcsolve`` to simulate Monte Carlo quantum trajectories, by defining the operators and states as ``jax`` or ``jaxdia`` dtypes and using a JAX-based ODE integrator (currently, ``qutip-jax`` supports a ``diffrax``-based integrator, ``DiffraxIntegrator``).

The following code demonstrates the evolution of :math:`10` trajectories with ``mcsolve`` for the two-level system described in `QuTiP's Monte Carlo Solver tutorial <https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-monte.html>`_ with a Hilbert space dimension of :math:`N = 10^4` for the cavity mode:

.. code-block::

    import jax.numpy as jnp
    import qutip
    import qutip_jax

    N = 10000
    tlist = jnp.linspace(0.0, 10.0, 200)
    # ``jaxdia`` operators support higher dimensional Hilbert spaces in the GPU
    with qutip.CoreOptions(default_dtype="jaxdia"):
        a = qutip.tensor(qutip.qeye(2), qutip.destroy(N))
        sm = qutip.tensor(qutip.destroy(2), qutip.qeye(N))
    H = 2.0 * jnp.pi * a.dag() * a + 2.0 * jnp.pi * sm.dag() * sm + 2.0 * jnp.pi * 0.25 * (sm * a.dag() + sm.dag() * a)
    # using ``jax`` dtype since ``DiffraxIntegrator`` anyway converts the final state to ``jax``
    state = qutip.tensor(qutip.fock(2, 0, dtype="jax"), qutip.fock(N, 8, dtype="jax"))
    c_ops = [jnp.sqrt(0.1) * a]
    e_ops = [a.dag() * a, sm.dag() * sm]
    result = qutip.mcsolve(H, state, tlist, c_ops, e_ops, ntraj=10, options={
        "method": "diffrax"
    })

The default solver for ``DiffraxIntegrator`` is ``diffrax.Tsit5()`` with an adaptive step-size controller (``diffrax.PIDController()``) using QuTiP's default tolerances (``atol = 1e-8``, ``rtol = 1e-6``).
To use a different solver or step-size controller (see `Diffrax ODE Solvers <https://docs.kidger.site/diffrax/api/solvers/ode_solvers/>`_ and `Diffrax Step Size Controllers <https://docs.kidger.site/diffrax/api/stepsize_controller/>`_ for available options), the following options can be passed alongside ``"method": "diffrax"``:

.. code-block::

    from diffrax import Dopri5, ConstantStepSize
    options = dict(
        method = "diffrax",
        solver = Dopri5(),
        stepsize_controller = ConstantStepSize(),
        dt0 = 0.001
    )

Note that the coefficient function of a time-dependent Hamiltonian needs to be jit-wrapped before it is passed to the solver. An example snippet for a coefficient with additional arguments is given below:

.. code-block::

    from functools import partial
    import jax

    @partial(jax.jit, static_argnames=("omega", ))
    def H_1_coeff(t, omega):
        return 2.0 * jnp.pi * 0.25 * jnp.cos(2.0 * omega * t)

    H_0 = 2.0 * jnp.pi * a.dag() * a + 2.0 * jnp.pi * sm.dag() * sm
    H_1_op = sm * a.dag() + sm.dag() * a
    H = [H_0, [H_1_op, H_1_coeff]]
    result = qutip.mcsolve(H, state, tlist, c_ops, e_ops, ntraj=10, options={
        "method": "diffrax"
    }, args={
        "omega": 1.0 # arguments for the coefficient function are passed here
    })

Alternatively, the ``JaxJitCoeff`` class can be utilized as demonstrated by the following snippet:

.. code-block::

    from qutip_jax.qobjevo import JaxJitCoeff
    H = [H_0, [H_1_op, JaxJitCoeff(lambda t, omega: 2.0 * jnp.pi * 0.25 * jnp.cos(2.0 * omega * t), args={
        "omega": 1.0 # arguments for the coefficient function are passed here
    }, static_argnames=("omega", ))]]
    result = qutip.mcsolve(H, state, tlist, c_ops, e_ops, ntraj=10, options={
        "method": "diffrax"
    })