import jax.numpy as jnp

from .jaxarray import JaxArray
from .convert import jax_from_dense

import numpy as np

import qutip


__all__ = [
    "zeros_jaxarray",
    "identity_jaxarray",
    "diag_jaxarray",
    "one_element_jaxarray",
]


def zeros_jaxarray(rows, cols):
    """
    Creates a matrix representation of zeros with the given dimensions.

    Parameters
    ----------
        rows, cols : int
            The number of rows and columns in the output matrix.
    """
    return JaxArray(jnp.zeros((rows, cols), dtype=jnp.complex128))


def identity_jaxarray(dimensions, scale=None):
    """
    Creates a square identity matrix of the given dimension.

    Optionally, the `scale` can be given, where all the diagonal elements will
    be that instead of 1.

    Parameters
    ----------
    dimension : int
        The dimension of the square output identity matrix.
    scale : complex, optional
        The element which should be placed on the diagonal.
    """
    if scale is None:
        return JaxArray(jnp.eye(dimensions, dtype=jnp.complex128))
    return JaxArray(jnp.eye(dimensions, dtype=jnp.complex128) * scale)


def diag_jaxarray(diagonals, offsets=None, shape=None):
    """
    Constructs a matrix from diagonals and their offsets.

    Using this function in single-argument form produces a square matrix with
    the given values on the main diagonal. With lists of diagonals and offsets,
    the matrix will be the smallest possible square matrix if shape is not
    given, but in all cases the diagonals must fit exactly with no extra or
    missing elements. Duplicated diagonals will be summed together in the
    output.

    Parameters
    ----------
    diagonals : sequence of array_like of complex or array_like of complex
        The entries (including zeros) that should be placed on the diagonals in
        the output matrix.  Each entry must have enough entries in it to fill
        the relevant diagonal and no more.
    offsets : sequence of integer or integer, optional
        The indices of the diagonals.  `offsets[i]` is the location of the
        values `diagonals[i]`.  An offset of 0 is the main diagonal, positive
        values are above the main diagonal and negative ones are below the main
        diagonal.
    shape : tuple, optional
        The shape of the output as (``rows``, ``columns``).  The result does
        not need to be square, but the diagonals must be of the correct length
        to fit in exactly.
    """
    try:
        diagonals = list(diagonals)
        # Can this be replaced with pure jnp and lax conditionals?
        if diagonals and np.isscalar(diagonals[0]):
            # Catch the case where we're being called as (for example)
            #   diags([1, 2, 3], 0)
            # with a single diagonal and offset.
            diagonals = [diagonals]
    except TypeError:
        raise TypeError("diagonals must be a list of arrays of complex")

    if offsets is None:
        if len(diagonals) == 0:
            offsets = []
        elif len(diagonals) == 1:
            offsets = [0]
        else:
            raise TypeError(
                "offsets must be supplied if passing more than one diagonal"
            )

    offsets = np.atleast_1d(offsets)
    if offsets.ndim > 1:
        raise ValueError("offsets must be a 1D array of integers")
    if len(diagonals) != len(offsets):
        raise ValueError("number of diagonals does not match number of offsets")

    if shape:
        n_rows, n_cols = shape
    else:
        n_rows = n_cols = abs(offsets[0]) + len(diagonals[0])

    if n_rows == n_cols:
        # jax diag only create square matrix
        out = jnp.zeros((n_rows, n_cols), dtype=jnp.complex128)
        for offset, diag in zip(offsets, diagonals):
            out += jnp.diag(jnp.array(diag), offset)
        out = JaxArray(out)
    else:
        out = jax_from_dense(qutip.core.data.dense.diags(diagonals, offsets, shape))

    return out


def one_element_jaxarray(shape, position, value=None):
    """
    Creates a matrix with only one nonzero element.

    Parameters
    ----------
    shape : tuple
        The shape of the output as (``rows``, ``columns``).

    position : tuple
        The position of the non zero in the matrix as (``rows``, ``columns``).

    value : complex, optional
        The value of the non-null element.
    """
    if not (0 <= position[0] < shape[0] and 0 <= position[1] < shape[1]):
        raise ValueError(
            "Position of the elements out of bound: "
            + str(position)
            + " in "
            + str(shape)
        )
    value = value or 1
    out = jnp.zeros(shape, dtype=jnp.complex128)
    return JaxArray(out.at[position].set(value))


qutip.data.zeros.add_specialisations(
    [
        (JaxArray, zeros_jaxarray),
    ]
)

qutip.data.identity.add_specialisations([(JaxArray, identity_jaxarray)])

qutip.data.diag.add_specialisations(
    [
        (JaxArray, diag_jaxarray),
    ]
)

qutip.data.one_element.add_specialisations(
    [
        (JaxArray, one_element_jaxarray),
    ]
)
