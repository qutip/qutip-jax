.. _qtjax_solver:

*****************************
Jax BackEnd in Qutip's Solver
*****************************


.. _mesolve:

Using Jax in ``mesolve``
========================


To use the jax arrays in Qutip solver such as ``mesolve``, two conditions must be fullfill:

1. All opertors and states must use jax arrays.
2. An ODE integrator supporting Jax must be used.
   ODE solver from the diffrax project are made available when importing qutip-jax

.. code-block::

    import qutip
    import qutip_jax

    with qutip.CoreOptions(default_dtype="jax"):
        H = qutip.rand_herm(5)
        c_ops = [qutip.destroy(5)]
        rho0 = qutip.basis(5, 4)

    result = qutip.mesolve(H, rho0, [0, 1], c_ops=c_ops, options={"method": "diffrax"})

