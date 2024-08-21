import jax.numpy as jnp
import numpy as np
from qutip import settings
from qutip import SESolver, MCSolver, MESolver

__all__ = ["set_as_default"]

def set_as_default(*, revert=False):
    if revert:
        settings.core['numpy_backend'] = np
        settings.core['default_dtype'] = None
        SESolver.solver_options['method'] = 'adams'
        MESolver.solver_options['method'] = 'adams'
        MCSolver.solver_options['method'] = 'adams'
    else:
        settings.core['numpy_backend'] = jnp
        settings.core['default_dtype'] = 'jax'
        SESolver.solver_options['method'] = 'diffrax'
        MESolver.solver_options['method'] = 'diffrax'
        MCSolver.solver_options['method'] = 'diffrax'
