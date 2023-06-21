from jax import jit
import jax.numpy as jnp
import numpy as np
from functools import partial

import qutip
from .jaxarray import JaxArray
from .properties import isherm_jaxarray


__all__ = [
    "eigs_jaxarray", "svd_jaxarray", "solve_jaxarray",
]


@partial(jit, static_argnums=[1, 2, 3, 4])
def _eigs_jaxarray(data, isherm, vecs, eigvals, low_first):
    """
    Internal function to dispatch the eigenvalue solver to `eigh`, `eig`,
    `eigvalsh` or `eigvals` based on the parameters.
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
    Return eigenvalues and eigenvectors for a `Data` of type `"jax"`.  Takes no
    special keyword arguments; see the primary documentation in :func:`.eigs`.
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


@partial(jit, static_argnums=[1, 2, 3])
def svd_jaxarray(data, vecs=True, full_matrices=True, hermitian=False):
    """
    Singular Value Decomposition:

    ``data = U @ S @ Vh``

    Where ``S`` is diagonal.

    Parameters
    ----------
    data : JaxArray
        Input matrix
    vecs : bool, optional (True)
        Whether the singular vectors (``U``, ``Vh``) should be returned.
    full_matrices : bool, optional (True)
        If ``True``, ``U`` and ``Vh`` will be square.
    hermitian : bool, optional (False)
        Whether to use a faster algorithms for hermitian matrix,
        (do not check the hermicity of ``data``.)

    Returns
    -------
    U : JaxArray
        Left singular vectors as columns. Only returned if ``vecs == True``.
    S : jax.Array
        Singular values.
    Vh : JaxArray
        Right singular vectors as rows. Only returned if ``vecs == True``.
    """
    out = jnp.linalg.svd(
        data._jxa,
        compute_uv=vecs, full_matrices=full_matrices, hermitian=hermitian
    )
    if vecs:
        u, s, vh = out
        return JaxArray(u, copy=False), s, JaxArray(vh, copy=False)
    return out


qutip.data.svd.add_specialisations(
    [(JaxArray, svd_jaxarray),]
)


def solve_jaxarray(matrix: JaxArray, target: JaxArray, method=None,
            options: dict={}) -> JaxArray:
    """
    Solve ``Ax=b`` for ``x``.

    Parameters:
    -----------

    matrix : JaxArray
        The matrix ``A``.

    target : JaxArray
        The matrix or vector ``b``.

    method : str {"solve", "lstsq"}, default="solve"
        The function from numpy.linalg to use to solve the system.

    options : dict
        Options to pass to the solver. "lstsq" use "rcond" while, "solve" do
        not use any.

    Returns:
    --------
    x : JaxArray
        Solution to the system Ax = b.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("can only solve using square matrix")
    if matrix.shape[1] != target.shape[0]:
        raise ValueError("target does not match the system")

    if method in ["solve", None]:
        out = jnp.linalg.solve(matrix._jxa, target._jxa)
    elif method == "lstsq":
        out, *_ = jnp.linalg.lstsq(
            matrix._jxa,
            target._jxa,
            rcond=options.get("rcond", None)
        )
    else:
        raise ValueError(f"Unknown solver {method},"
                         " 'solve' and 'lstsq' are supported.")
    return JaxArray(out, copy=False)


qutip.data.solve.add_specialisations(
    [(JaxArray, JaxArray, JaxArray, solve_jaxarray),]
)
