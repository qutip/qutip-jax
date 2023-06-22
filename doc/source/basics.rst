.. _qtjax_intro:

************
Jax in Qutip
************


.. _basic_usage:

Basic Usages
============

Once imported, there re 3many ways to tell Qutip to use the jax backend.

- Pass Jax array to the Qobj constructor.

.. code-block::

    import qutip
    import qutip_jax
    import jax.numpy as jnp

    jax_eye_qobj = qutip.Qobj(jnp.eye(3))

- Manually convert a ``Qobj`` or ``QobjEvo``.

.. code-block::

    import qutip
    import qutip_jax

    jax_eye_qobj = qutip.qeye(3).to("jax")

- Use qutip state and operator function with ``dtype="jax"``.

.. code-block::

    import qutip
    import qutip_jax

    jax_eye_qobj = qutip.qeye(3, dtype="jax")

- Use qutip settings to set jax as the default backend.

.. code-block::

    import qutip
    import qutip_jax
    qutip.settings.core["default_dtype"] = "jax"

    jax_eye_qobj = qutip.qeye(3)

- Use qutip settings in a context.

.. code-block::

    import qutip
    import qutip_jax

    with qutip.CoreOptions("default_dtype"="jax"):
        jax_eye_qobj = qutip.qeye(3)