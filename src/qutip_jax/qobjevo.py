import equinox as eqx
import jaxlib
import jax
import jax.numpy as jnp
import numpy as np
from .jaxarray import JaxArray
from .binops import matmul_jaxdia_jaxarray_jaxarray
from .create import zeros_jaxdia
from qutip.core.coefficient import coefficient_builders
from qutip.core.cy.coefficient import Coefficient
from qutip import Qobj


__all__ = []


class JaxJitCoeff(Coefficient):
    func: callable = eqx.static_field()
    args: dict

    def __init__(self, func, args={}, **_):
        self.func = func
        Coefficient.__init__(self, args)

    @eqx.filter_jit
    def __call__(self, t, _args=None, **kwargs):
        if _args:
            kwargs.update(_args)
        args = self.args.copy()
        for key in kwargs:
            if key in args:
                args[key] = kwargs[key]
        return self.func(t, **args)

    def replace_arguments(self, _args=None, **kwargs):
        if _args:
            kwargs.update(_args)
        return JaxJitCoeff(self.func, {**self.args, **kwargs})

    def __add__(self, other):
        if isinstance(other, JaxJitCoeff):

            def f(t, **kwargs):
                return self(t, **kwargs) + other(t, **kwargs)
            return JaxJitCoeff(eqx.filter_jit(f), {})

        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, JaxJitCoeff):

            def f(t, **kwargs):
                return self(t, **kwargs) * other(t, **kwargs)
            return JaxJitCoeff(eqx.filter_jit(f), {})

        return NotImplemented

    def conj(self):
        def f(t, **kwargs):
            return jnp.conj(self(t, **kwargs))

        return JaxJitCoeff(eqx.filter_jit(f), {})

    def _cdc(self):
        def f(t, **kwargs):
            val = self(t, **kwargs)
            return jnp.conj(val) * val

        return JaxJitCoeff(eqx.filter_jit(f), {})

    def copy(self):
        return self

    def __reduce__(self):
        # Jitted function cannot be pickled.
        # Extract the original function and re-jit it.
        # This can fail depending on the wrapped object.
        return (self.restore, (self.func.__wrapped__, self.args))

    @classmethod
    def restore(cls, func, args):
        return cls(eqx.filter_jit(func), args)

    def flatten(self):
        return (self.args,), (self.func,)

    @classmethod
    def unflatten(cls, aux_data, children):
        return JaxJitCoeff(*aux_data, *children)


coefficient_builders[eqx._jit._JitWrapper] = JaxJitCoeff
coefficient_builders[jaxlib.xla_extension.PjitFunction] = JaxJitCoeff
jax.tree_util.register_pytree_node(
    JaxJitCoeff, JaxJitCoeff.flatten, JaxJitCoeff.unflatten
)


class JaxQobjEvo(eqx.Module):
    """
    Pytree friendly QobjEvo for the Diffrax integrator.

    It only support list based `QobjEvo`.
    """

    batched_data: jnp.ndarray
    coeffs: list
    sparse_part: list
    dims: object = eqx.static_field()
    shape: tuple = eqx.static_field()

    def __init__(self, qobjevo):
        as_list = qobjevo.to_list()
        coeffs = []
        qobjs = []
        self.dims = qobjevo.dims
        self.shape = qobjevo.shape
        self.coeffs = []
        self.sparse_part = []
        self.batched_data = None

        constant = JaxJitCoeff(eqx.filter_jit(lambda t, **_: 1.0))

        for part in as_list:
            if isinstance(part, Qobj):
                qobjs.append(part)
                coeffs.append(constant)
            elif (
                isinstance(part, list) and isinstance(part[0], Qobj)
            ):
                qobjs.append(part[0])
                coeffs.append(part[1])
            else:
                raise NotImplementedError(
                    "Function based QobjEvo are not supported"
                )

        dense_part = []
        for qobj, coeff in zip(qobjs, coeffs):
            if type(qobj.data) in ["JaxDia"]:
                # TODO: CSR also?
                self.sparse_part.append(qobj.data, coeff)
            else:
                dense_part.append((qobj, coeff))

        if dense_part:
            self.batched_data = jnp.zeros(
                self.shape + (len(dense_part),), dtype=np.complex128
            )
            for i, (qobj, coeff) in enumerate(dense_part):
                self.batched_data = self.batched_data.at[:, :, i].set(
                    qobj.to("jax").data._jxa
                )
                self.coeffs.append(coeff)

    @eqx.filter_jit
    def _coeff(self, t, **args):
        list_coeffs = [coeff(t, **args) for coeff in self.coeffs]
        return jnp.array(list_coeffs, dtype=jnp.complex128)

    def __call__(self, t, **kwargs):
        return Qobj(self.data(t, **kwargs), dims=self.dims)

    @eqx.filter_jit
    def data(self, t, **kwargs):
        if self.batched_data is not None:
            coeff = self._coeff(t, **kwargs)
            data = jnp.dot(self.batched_data, coeff)
            out = JaxArray(data)
        else:
            out = zeros_jaxdia(*self.shape)
        for data, coeff in self.sparse_part:
            out = out + data
        return out

    @eqx.filter_jit
    def matmul_data(self, t, y, **kwargs):
        if self.batched_data is not None:
            coeffs = self._coeff(t, **kwargs)
            out = JaxArray(jnp.dot(jnp.dot(self.batched_data, coeffs), y._jxa))
        else:
            out = zeros_jaxdia(self.shape[1], 1)
        for data, coeff in self.sparse_part:
            out = matmul_jaxdia_jaxarray_jaxarray(data, y, coeff(t), out)
        return out

    def arguments(self, args):
        out = JaxQobjEvo.__new__(JaxQobjEvo)
        coeffs = [coeff.replace_arguments(args) for coeff in self.coeffs]
        sparse_part = [
            (data, coeff.replace_arguments(args))
            for data, coeff in self.sparse_part
        ]
        object.__setattr__(out, "coeffs", coeffs)
        object.__setattr__(out, "sparse_part", sparse_part)
        object.__setattr__(out, "batched_data", self.batched_data)
        object.__setattr__(out, "dims", self.dims)
        return out
