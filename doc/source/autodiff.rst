.. _qtjax_autodiff:

**************************************
Auto differentiation in Qutip's Solver
**************************************


.. _autodiff_example:

Auto differentiation in ``sesolve``
===================================

As long as the jax backend is used by each ``Qobj`` in a function, jax auto differentiation
should work:

.. code-block::

    import qutip_jax
    import qutip as qt
    from qutip.solver.sesolve import sesolve, SeSolver
    import jax
    @jax.jit
    def fp(t, w):
        return jax.numpy.exp(1j * t * w)

    @jax.jit
    def fm(t, w):
        return jax.numpy.exp(-1j * t * w)

    H = (
        qt.num(2) 
        + qt.destroy(2) * qt.coefficient(fp, args={"w": 3.1415}) 
        + qt.create(2) * qt.coefficient(fm, args={"w": 3.1415})
    ).to("jax")

    ket = qt.basis(2, 1)

    solver = SeSolver(H, options={"method": "diffrax"})

    def f(w, solver):
        result = solver.run(ket, [0, 1], e_ops=qt.num(2).to("jax"), args={"w":w})
        return result.e_data[0][1].real

    jax.grad(f)(0.5, solver)


Auto differentiation in ``mcsolve``
===================================

.. note::

   The functionality demonstrated in this example is currently available only in 
   the development (`dev.major`) branch of QuTiP. Ensure you are using the appropriate 
   version if you wish to replicate these results.

   The automatic differentiation (`jax.grad`) in `mcsolve` does not support parallel map operations. 
   To ensure accurate gradient computations, please use the default serial execution instead of 
   parallel mapping within `mcsolve`.


.. code-block:: python
    import qutip_jax as qjax
    import qutip as qt
    import jax
    import jax.numpy as jnp
    from functools import partial
    from qutip import mcsolve
    # Use JAX backend for QuTiP
    qjax.use_jax_backend()
    # Define time-dependent functions
    @partial(jax.jit, static_argnames=("omega",))
    def H_1_coeff(t, omega):
        return 2.0 * jnp.pi * 0.25 * jnp.cos(2.0 * omega * t)
    # Define operators and states
    size = 10
    a = qt.tensor(qt.destroy(size), qt.qeye(2)).to('jaxdia')    # Annihilation operator
    sm = qt.qeye(size).to('jaxdia') & qt.sigmax().to('jaxdia')  # Example spin operator
    # Define the Hamiltonian
    H_0 = 2.0 * jnp.pi * a.dag() * a + 2.0 * jnp.pi * sm.dag() * sm
    H_1_op = sm * a.dag() + sm.dag() * a
    # Initialize the Hamiltonian with time-dependent coefficients
    H = [H_0, [H_1_op, qt.coefficient(H_1_coeff, args={"omega": 1.0})]]
    # Define initial states, mixed states are not supported
    state = qt.basis(size, size - 1).to('jax') & qt.basis(2, 1).to('jax')
    
    # Define collapse operators and observables
    c_ops = [jnp.sqrt(0.1) * a]
    e_ops = [a.dag() * a, sm.dag() * sm]
    # Time list
    tlist = jnp.linspace(0.0, 10.0, 101)
    # Define the function for which we want to compute the gradient
    def f(omega):
        # Update the Hamiltonian with the new coefficient
        H[1][1] = qt.coefficient(H_1_coeff, args={"omega": omega})
        
        # Run the Monte Carlo solver
        result = mcsolve(H, state, tlist, c_ops, e_ops, ntraj=10, options={"method": "diffrax"})
        
        # Return the expectation value of the number operator at the final time
        return result.expect[0][-1].real
    # Compute the gradient
    gradient = jax.grad(f)(1.0)
    print("Gradient:", gradient)
