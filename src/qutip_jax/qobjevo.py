import equinox as eqx
import jaxlib
import jax
import jax.numpy as jnp
import numpy as np
from .jaxarray import JaxArray
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
    dims: object = eqx.static_field()

    def __init__(self, qobjevo):
        as_list = qobjevo.to_list()
        self.coeffs = []
        qobjs = []
        self.dims = qobjevo.dims

        constant = JaxJitCoeff(eqx.filter_jit(lambda t, **_: 1.))

        for part in as_list:
            if isinstance(part, Qobj):
                qobjs.append(part)
                self.coeffs.append(constant)
            elif (
                isinstance(part, list)
                and isinstance(part[0], Qobj)
            ):
                qobjs.append(part[0])
                self.coeffs.append(part[1])
            else:
                # TODO:
                raise NotImplementedError(
                    "Function based QobjEvo are not supported"
                )

        if qobjs:
            shape = qobjs[0].shape
            self.batched_data = jnp.zeros(
                shape + (len(qobjs),), dtype=np.complex128
            )
            for i, qobj in enumerate(qobjs):
                self.batched_data = self.batched_data.at[:, :, i].set(
                    qobj.to("jax").data._jxa
                )

    @eqx.filter_jit
    def _coeff(self, t, **args):
        list_coeffs = [coeff(t, **args) for coeff in self.coeffs]
        return jnp.array(list_coeffs, dtype=np.complex128)

    def __call__(self, t, **kwargs):
        return Qobj(self.data(t, **kwargs), dims=self.dims)

    @eqx.filter_jit
    def data(self, t, **kwargs):
        coeff = self._coeff(t, **kwargs)
        data = jnp.dot(self.H, coeff)
        return JaxArray(data)

    @eqx.filter_jit
    def matmul_data(self, t, y, **kwargs):
        coeffs = self._coeff(t, **kwargs)
        out = JaxArray(jnp.dot(jnp.dot(self.batched_data, coeffs), y._jxa))
        return out

    def arguments(self, args):
        out = JaxQobjEvo.__new__(JaxQobjEvo)
        coeffs = [coeff.replace_arguments(args) for coeff in self.coeffs]
        object.__setattr__(out, "coeffs", coeffs)
        object.__setattr__(out, "H", self.batched_data)
        object.__setattr__(out, "dims", self.dims)
        return out
