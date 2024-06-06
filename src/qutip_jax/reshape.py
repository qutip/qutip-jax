import jax.numpy as jnp
import jax.lax as lax
from jax import jit
from . import JaxArray
import qutip
from functools import partial
import numpy as np

__all__ = [
    "reshape_jaxarray",
    "column_stack_jaxarray",
    "column_unstack_jaxarray",
    "split_columns_jaxarray",
    "ptrace_jaxarray",
]


# jit slower
def reshape_jaxarray(matrix, n_rows_out, n_cols_out):
    if n_rows_out * n_cols_out != matrix.shape[0] * matrix.shape[1]:
        message = (
            f"cannot reshape {matrix.shape} to ({n_rows_out}, {n_cols_out})"
        )
        raise ValueError(message)
    if n_rows_out <= 0 or n_cols_out <= 0:
        raise ValueError("must have > 0 rows and columns")
    return JaxArray._fast_constructor(
        jnp.reshape(matrix._jxa, (n_rows_out, n_cols_out)),
        (n_rows_out, n_cols_out)
    )


# jit slower
def column_stack_jaxarray(matrix):
    shape = (matrix._jxa.shape[0] * matrix._jxa.shape[1], 1)
    return JaxArray._fast_constructor(
        lax.reshape(matrix._jxa, shape, (1, 0)),
        shape
    )


@partial(jit, static_argnums=[1])
def column_unstack_jaxarray(matrix, rows):
    if matrix.shape[1] != 1:
        raise ValueError("input is not a single column")
    if rows < 1:
        raise ValueError("rows must be a positive integer")
    if matrix.shape[0] % rows:
        raise ValueError("number of rows does not divide into the shape")
    shape = (matrix._jxa.shape[0] * matrix._jxa.shape[1] // rows, rows)
    return JaxArray._fast_constructor(
        lax.reshape(matrix._jxa, shape, (1, 0)).transpose(),
        shape[::-1]
    )


@jit
def split_columns_jaxarray(matrix, copy=None):
    # `copy` is passed by some `Qobj` methods
    # but JaxArray always creates a new array.
    return [
        JaxArray(matrix._jxa[:, k]) for k in range(matrix.shape[1])
    ]


def _parse_ptrace_inputs(dims, sel, shape):
    dims = np.atleast_1d(dims).ravel()
    sel = np.atleast_1d(sel)
    sel.sort()

    if shape[0] != shape[1]:
        raise ValueError("ptrace is only defined for square density matrices")

    if shape[0] != np.prod(dims, dtype=int):
        raise ValueError(
            f"the input matrix shape, {shape} and the"
            f" dimension argument, {dims}, are not compatible."
        )
    if sel.ndim != 1:
        raise ValueError("Selection must be one-dimensional")

    if any(d < 1 for d in dims):
        raise ValueError(
            f"dimensions must be greated than zero but where dims={dims}."
        )

    for i in range(sel.shape[0]):
        if sel[i] < 0 or sel[i] >= dims.size:
            raise IndexError("Invalid selection index in ptrace.")
        if i > 0 and sel[i] == sel[i - 1]:
            raise ValueError("Duplicate selection index in ptrace.")

    return dims, sel


@partial(jit, static_argnums=[1, 2, 3, 4])
def _ptrace_core(matrix, dims, transpose_idx, dtrace, dkeep):
    tensor = matrix.reshape(dims)
    transposed = tensor.transpose(transpose_idx)
    reshaped = transposed.reshape([dtrace, dtrace, dkeep, dkeep])
    return jnp.trace(reshaped)


def ptrace_jaxarray(matrix, dims, sel):
    dims, sel = _parse_ptrace_inputs(dims, sel, matrix.shape)

    if len(sel) == len(dims):
        return matrix.copy()

    nd = dims.shape[0]
    dims2 = tuple(list(dims) * 2)
    sel = list(sel)
    qtrace = list(set(np.arange(nd)) - set(sel))


    dkeep = np.prod([dims[x] for x in sel], dtype=int)
    dtrace = np.prod([dims[x] for x in qtrace], dtype=int)

    transpose_idx = tuple(
        qtrace + [nd + q for q in qtrace]
        + sel + [nd + q for q in sel]
    )

    return JaxArray(
        _ptrace_core(matrix._jxa, dims2, transpose_idx, dtrace, dkeep)
    )


qutip.data.reshape.add_specialisations(
    [(JaxArray, JaxArray, reshape_jaxarray),]
)


qutip.data.column_stack.add_specialisations(
    [(JaxArray, JaxArray, column_stack_jaxarray),]
)


qutip.data.column_unstack.add_specialisations(
    [(JaxArray, JaxArray, column_unstack_jaxarray),]
)


qutip.data.split_columns.add_specialisations(
    [(JaxArray, split_columns_jaxarray),]
)


qutip.data.ptrace.add_specialisations(
    [(JaxArray, JaxArray, ptrace_jaxarray),]
)
