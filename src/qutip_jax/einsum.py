import jax
import jax.numpy as jnp
import numpy as np
from qutip.core.data import einsum
from qutip.core.data.convert import to as _to
from .jaxarray import JaxArray 


from functools import partial


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def _einsum_jax_core(
    op0_arr, 
    rest_arrs, 
    subscripts, 
    tensor_shapes, 
    tensor_perms, 
    out_perm,
    out_shape
):
    operands = (op0_arr,) + rest_arrs
    tensors = []
    
    for arr, shape, perm in zip(operands, tensor_shapes, tensor_perms):
        tensors.append(jnp.transpose(jnp.reshape(arr, shape), perm))

    result = jnp.einsum(subscripts, *tensors, optimize=True)

    if result.shape == ():
        return jnp.reshape(result, (1, 1))

    inv_out_perm = tuple(np.argsort(out_perm))
    result_physical = jnp.transpose(result, inv_out_perm)

    if out_shape is None:
        half = result_physical.ndim // 2
        rows = int(np.prod(result_physical.shape[:half]))
        cols = int(np.prod(result_physical.shape[half:]))
        out_shape = (rows, cols)

    return jnp.reshape(result_physical, out_shape)


def einsum_jax(
        op0, /,
        *rest_operands,
        subscripts,
        tensor_shapes,
        tensor_perms,
        out_perm,
        out_shape=None
):
    """
    JAX / XLA specialization for einsum.
    """
    # Unwrap QuTiP JaxArrays into raw JAX arrays
    jax_op0 = _to(JaxArray, op0)._jxa
    rest_arrs = tuple(_to(JaxArray, op)._jxa for op in rest_operands)
    
    tensor_shapes = tuple(tuple(s) for s in tensor_shapes)
    tensor_perms = tuple(tuple(p) for p in tensor_perms)
    out_perm = tuple(out_perm)
    if out_shape is not None:
        out_shape = tuple(out_shape)

    result_arr = _einsum_jax_core(
        jax_op0, rest_arrs, subscripts, tensor_shapes, tensor_perms, out_perm, out_shape
    )
    
    return JaxArray(result_arr)

einsum.add_specialisations([
    (JaxArray, JaxArray, einsum_jax),
])