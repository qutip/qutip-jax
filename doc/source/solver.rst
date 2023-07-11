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

