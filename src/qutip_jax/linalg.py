from jax import jit
import jax.numpy as jnp
import numpy as np
from functools import partial

import qutip
from .jaxarray import JaxArray
from .properties import isherm_jaxarray


__all__ = [
    "eigs_jaxarray",
]


@partial(jit, static_argnums=[1, 2, 3, 4])
def _eigs_jaxarray(data, isherm, vecs, eigvals, low_first):
    """
    Internal functions for computing eigenvalues and eigenstates for a dense
    matrix.
    """
    if isherm and vecs:
        evals, evecs = jnp.linalg.eigh(data)
    elif vecs:
        evals, evecs = jnp.linalg.eig(data)
    elif isherm:
        evals = jnp.linalg.eigvalsh(data)
        evecs = None
    else:
        evals = jnp.linalg.eigvals(data)
        evecs = None

    perm = jnp.argsort(evals.real)
    evals = evals[perm]
    if not low_first:
        evals = evals[::-1]
    evals = evals[:eigvals]

    if vecs:
        evecs = evecs[:, perm]
        if not low_first:
            evecs = evecs[:, ::-1]
        evecs = evecs[:, :eigvals]

    return evals, evecs

# Can't jit it if we accept isherm=None
def eigs_jaxarray(data, isherm=None, vecs=True, sort='low', eigvals=0):
    """
    Return eigenvalues and eigenvectors for a Dense matrix.  Takes no special
    keyword arguments; see the primary documentation in :func:`.eigs`.
    """
    N = data.shape[0]
    if data.shape[0] != data.shape[1]:
        raise TypeError("Can only diagonalize square matrices")
    if sort not in ('low', 'high'):
        raise ValueError("'sort' must be 'low' or 'high'")
    if eigvals > N:
        raise ValueError("Number of requested eigen vals/vecs must be <= N.")
    eigvals = eigvals or N
    # Let dict raise keyerror of
    low_first = {"low": True, "high": False}[sort]
    isherm = isherm if isherm is not None else bool(isherm_jaxarray(data))

    evals, evecs = _eigs_jaxarray(data._jxa, isherm, vecs, eigvals, low_first)

    return (evals, JaxArray(evecs, copy=False)) if vecs else evals


qutip.data.eigs.add_specialisations(
    [(JaxArray, eigs_jaxarray),]
)
