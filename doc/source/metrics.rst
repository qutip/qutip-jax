Using JAX with QuTiP Core Metrics and Entropy
=============================================

This guide demonstrates how to utilize JAX's `grad` and `jit` functionalities with QuTiP's `core.metrics` and `entropy` modules by changing the backend to JAX.

Setting Up the JAX Backend
--------------------------

To enable JAX as the backend for QuTiP, you need to set the backend to `jax` using the `use_jax_backend` function. This allows you to use JAX's `grad` and `jit` with QuTiP functions.

.. note::

    This feature is not available in a released version of QuTiP. It is only available on an experimental development branch called `dev-major` in QuTiP.

.. code-block:: python

    import qutip
    import qutip_jax

    # Use JAX as the backend
    qutip_jax.use_jax_backend()

Using `jax.jit` with QuTiP
--------------------------

`jax.jit` compiles your function to make it run faster. Here's how to use `jax.jit` with functions from `qutip.core.metrics` and `qutip.entropy`:

### Example with `fidelity` from `qutip.core.metrics`

.. code-block:: python

    import jax
    from qutip import basis
    from qutip.core.metrics import fidelity
    import qutip_jax

    # Use JAX as the backend
    qutip_jax.use_jax_backend()

    # Define states
    psi = basis(2, 0).to("jax")
    phi = basis(2, 1).to("jax")

    # JIT compile the fidelity function
    jit_fidelity = jax.jit(fidelity)

    # Compute fidelity using JIT compiled function
    result = jit_fidelity(psi, phi)
    print("Fidelity:", result)

### Example with `entropy_vn` from `qutip.entropy`

.. code-block:: python

    from qutip import ket2dm
    from qutip.entropy import entropy_vn
    import qutip_jax

    # Use JAX as the backend
    qutip_jax.use_jax_backend()

    # Define a density matrix
    rho = ket2dm(psi).to("jax")

    # JIT compile the entropy_vn function
    jit_entropy_vn = jax.jit(entropy_vn)

    # Compute von Neumann entropy using JIT compiled function
    result = jit_entropy_vn(rho)
    print("Von Neumann Entropy:", result)

Using `jax.grad` with QuTiP
---------------------------

`jax.grad` computes the gradient of a function. Here's how to use `jax.grad` with functions from `qutip.core.metrics` and `qutip.entropy`:

### Example with `fidelity` from `qutip.core.metrics`

To compute the gradient, you need a function that returns a scalar. Note that `jax.grad` for `fidelity` does not support `oper` states.

#### Gradient of `fidelity` for Ket/Bra States

.. code-block:: python

    import jax
    from qutip import basis, fidelity
    import qutip_jax

    # Use JAX as the backend
    qutip_jax.use_jax_backend()

    # Define bra and ket states
    bra_state = basis(2, 0).dag()
    ket_state = basis(2, 0)

    # Convert to JAX objects
    bra_state_jax = bra_state.to("jax")
    ket_state_jax = ket_state.to("jax")

    # Define a fidelity function
    def fidelity_jax(state1, state2):
        return fidelity(state1, state2)

    # Compute the gradient of the fidelity function with respect to the first argument
    grad_fidelity = jax.grad(fidelity_jax, argnums=0)

    # Calculate the gradient
    grad_result = grad_fidelity(bra_state_jax, ket_state_jax)
    print("Gradient of Fidelity:", grad_result)
    
### Example with `trace_dist` from `qutip.core.metrics`

The `trace_dist` function supports `oper` states for gradient computation.

.. code-block:: python

    from qutip import rand_dm
    from qutip.core.metrics import trace_dist
    import qutip_jax

    # Use JAX as the backend
    qutip_jax.use_jax_backend()

    # Define an operator state
    oper_state = rand_dm(2)
    ket_state = basis(2, 0)

    # Convert to JAX object
    oper_state_jax = oper_state.to("jax")
    ket_state_jax = ket_state.to("jax")
   
    # Define a trace distance function
    def trace_dist_jax(state1, state2):
        return trace_dist(state1, state2)

    # Compute the gradient of the trace distance function with respect to the first argument
    grad_trace_dist = jax.grad(trace_dist_jax, argnums=0)

    # Calculate the gradient
    grad_result = grad_trace_dist(oper_state_jax, ket_state_jax)
    print("Gradient of Trace Distance:", grad_result)

Changing Back to Default Backend
--------------------------------

If you want to switch back to the default backend (NumPy), use the following:

.. code-block:: python

    qutip.settings.core["numpy_backend"] = np

