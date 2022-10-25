import equinox as eqx
import jaxlib
import jax
import jax.numpy as jnp
import numpy as np
from .jaxarray import JaxArray
from qutip.core.coefficient import coefficient_builders
from qutip.core.cy.coefficient import Coefficient
from qutip import Qobj, qzero
from functools import partial


class JaxJitCoeff(eqx.Module, Coefficient):
    func: callable
    args: dict

    def __init__(self, func, args={}, **_):
        self.func = func
        self.args = args

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


coefficient_builders[eqx.jit._JitWrapper] = JaxJitCoeff
coefficient_builders[jaxlib.xla_extension.CompiledFunction] = JaxJitCoeff


class JaxQobjEvo(eqx.Module):
    """

    """
    H: jnp.ndarray
    coeffs: list
    funcs: list
    dims: object

    def __init__(self, qobjevo):
        as_list = qobjevo.to_list()
        self.funcs = []
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
                # TODO: More effort will be needed for jit to work with general
                # function based QobjEvo
                raise ValueError("Function based QobjEvo not supported")
                # self.funcs.append(part)

        if qobjs:
            shape = qobjs[0].shape
            self.H = jnp.zeros(shape + (len(qobjs),), dtype=np.complex128)
            for i, qobj in enumerate(qobjs):
                self.H = self.H.at[:, :, i].set(qobj.to("jax").data._jxa)
                if self.coeffs[i] == 1:
                    self.coeffs[i] = qt.coefficient(lambda t: 1.)

    @eqx.filter_jit
    def _coeff(self, t, **args):
        list_coeffs = [coeff(t, **args) for coeff in self.coeffs]
        return jnp.array(list_coeffs, dtype=np.complex128)

    def __call__(self, t, **kwargs):
        return Qobj(self.data(t, **kwargs), dims=self.dims)

    @eqx.filter_jit
    def data(self, t, **kwargs):
        coeff = self._coeff(t, **kwargs)
        print(coeff)
        data = jnp.dot(self.H, coeff)
        print(data)
        array = JaxArray(data)
        print("array", array)
        return array

    @eqx.filter_jit
    def matmul_data(self, t, y, **kwargs):
        # out = jnp.zeros(y.shape, dtype=np.complex128)
        # for f, args in self.funcs:
        #     out = out + matmul_jaxarray(
        #         f(t, **args, **kwargs).to(JaxArray).data, y
        #     )._jxa
        coeffs = self._coeff(t, **kwargs)
        out = JaxArray(jnp.dot(jnp.dot(self.H, coeffs), y._jxa))
        return out
