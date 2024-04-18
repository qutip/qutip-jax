.. _qtjax_intro:

************
Jax in QuTiP
************


.. _basic_usage:

Basic usage
===========

In orther to enable qutip-jax, it is just necessary to import the module. Once imported, ``qutip.Qobj``'s data can be represented as a JAX array. Furthermore, diffrax ODE will be available as an option for qutip's solvers (``sesolve``, ``mcsolve``, etc.).
None of the functions in the module are expected to be used directly. Instead, they will be used by qutip, allowing the user to interact only with the already familiar QuTiP interface.

There are many ways to create a QuTiP ``Qobj`` backed by JAX's array class.

- Passing JAX array to the ``Qobj`` constructor.

.. code-block::

    import qutip
    import qutip_jax
    import jax.numpy as jnp

    jax_eye_qobj = qutip.Qobj(jnp.eye(3))
    assert isinstance(jax_eye_qobj.data, qutip_jax.JaxArray)

- Converting a ``Qobj`` or ``QobjEvo`` using the ``to`` method.

.. code-block::

    import qutip
    import qutip_jax

    jax_eye_qobj = qutip.qeye(3).to("jax")
    assert isinstance(jax_eye_qobj.data, qutip_jax.JaxArray)

- Using QuTiP's native constructors' ``dtype`` parameter.

.. code-block::

    import qutip
    import qutip_jax

    jax_eye_qobj = qutip.qeye(3, dtype="jax")
    assert isinstance(jax_eye_qobj.data, qutip_jax.JaxArray)

- Changing QuTiP's default settings to set JAX as the default backend.

.. code-block::

    import qutip
    import qutip_jax
    qutip.settings.core["default_dtype"] = "jax"

    jax_eye_qobj = qutip.qeye(3)
    assert isinstance(jax_eye_qobj.data, qutip_jax.JaxArray)

- Changing QuTiP's default settings within a context.

.. code-block::

    import qutip
    import qutip_jax

    with qutip.CoreOptions(default_dtype="jax"):
        jax_eye_qobj = qutip.qeye(3)

    assert isinstance(jax_eye_qobj.data, qutip_jax.JaxArray)
