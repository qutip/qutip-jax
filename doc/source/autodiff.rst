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