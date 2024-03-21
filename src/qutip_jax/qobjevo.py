import jaxlib
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
from .jaxarray import JaxArray
from .jaxdia import JaxDia
from .binops import matmul_jaxdia_jaxarray_jaxarray
from .create import zeros_jaxdia, zeros_jaxarray
from qutip.core.coefficient import coefficient_builders
from qutip.core.cy.coefficient import Coefficient
from qutip import Qobj
from qutip.core.data.matmul import matmul
from functools import partial


__all__ = []


class JaxJitCoeff(Coefficient):
    func: callable
    static_argnames: tuple
    args: dict

    def __init__(self, func, args={}, static_argnames=(), **_):
        self.func = func
        self.static_argnames = static_argnames
        Coefficient.__init__(self, args)
        self.jit_call = jit(self._caller, static_argnames=self.static_argnames)

    def __call__(self, t, _args=None, **kwargs):
        if _args:
            kwargs.update(_args)
        return self.jit_call(t, **kwargs)

    def _caller(self, t, **kwargs):
        args = self.args.copy()
        for key in kwargs:
            if key in args:
                args[key] = kwargs[key]
        return self.func(t, **args)

    def replace_arguments(self, _args=None, **kwargs):
        if _args:
            kwargs.update(_args)
        args = self.args.copy()
        for key in kwargs:
            if key in args:
                args[key] = kwargs[key]
        return JaxJitCoeff(self.func, args=args)

    def __add__(self, other):
        if isinstance(other, JaxJitCoeff):
            merge_static = tuple(
                set(self.static_argnames) | set(other.static_argnames)
            )

            def f(t, **kwargs):
                return self._caller(t, **kwargs) + other._caller(t, **kwargs)

            return JaxJitCoeff(
                jit(f, static_argnames=merge_static),
                args={**self.args, **other.args},
                static_argnames=merge_static,
            )

        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, JaxJitCoeff):
            merge_static = tuple(
                set(self.static_argnames) | set(other.static_argnames)
            )

            def f(t, **kwargs):
                return self._caller(t, **kwargs) * self._caller(t, **kwargs)

            return JaxJitCoeff(
                jit(f, static_argnames=merge_static),
                args={**self.args, **other.args},
                static_argnames=merge_static,
            )

        return NotImplemented

    def conj(self):
        def f(t, **kwargs):
            return jnp.conj(self._caller(t, **kwargs))

        return JaxJitCoeff(
            jit(f, static_argnames=self.static_argnames),
            args=self.args,
            static_argnames=self.static_argnames,
        )

    def _cdc(self):
        def f(t, **kwargs):
            val = self(t, **kwargs)
            return jnp.conj(val) * val

        return JaxJitCoeff(
            jit(f, static_argnames=self.static_argnames),
            args=self.args,
            static_argnames=self.static_argnames,
        )

    def copy(self):
        return self

    def __reduce__(self):
        # Jitted function cannot be pickled.
        # Extract the original function and re-jit it.
        # This can fail depending on the wrapped object.
        return (
            self.restore,
            (self.func.__wrapped__, self.args, self.static_argnames)
        )

    @classmethod
    def restore(cls, func, args, static_argnames):
        return cls(
            jit(func, static_argnames=static_argnames),
            args,
            static_argnames
        )

    def flatten(self):
        static_args = {
            key: val for key, val in self.args.items()
            if key in self.static_argnames
        }
        jax_args = {
            key: val for key, val in self.args.items()
            if key not in self.static_argnames
        }
        return (jax_args,), (self.func, static_args, self.static_argnames)

    @classmethod
    def unflatten(cls, aux_data, children):
        func, static_args, static_argnames = aux_data

        return JaxJitCoeff(
            func,
            args={**children[0], **static_args},
            static_argnames=static_argnames
        )


coefficient_builders[jaxlib.xla_extension.PjitFunction] = JaxJitCoeff
jax.tree_util.register_pytree_node(
    JaxJitCoeff, JaxJitCoeff.flatten, JaxJitCoeff.unflatten
)


class JaxQobjEvo:
    """
    Pytree friendly QobjEvo for the Diffrax integrator.

    It only support list based `QobjEvo`.
    """

    batched_data: jnp.ndarray
    sparse_part: list
    coeffs: list
    sparse_part: list
    shape: tuple
    dims: object

    def __init__(self, qobjevo):
        as_list = qobjevo.to_list()
        coeffs = []
        qobjs = []
        self.dims = qobjevo.dims
        self.shape = qobjevo.shape
        self.coeffs = []
        self.sparse_part = []
        self.batched_data = None

        constant = JaxJitCoeff(jit(lambda t, **_: 1.0))

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
            if type(qobj.data) in [JaxDia]:
                # TODO: CSR also?
                self.sparse_part.append((qobj.data, coeff))
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

    @jit
    def _coeff(self, t):
        list_coeffs = [coeff(t) for coeff in self.coeffs]
        return jnp.array(list_coeffs, dtype=np.complex128)

    def __call__(self, t, _args=None, **kwargs):
        if args is not None:
            kwargs.update(_args)
        if kwargs:
            caller = self.arguments(kwargs)
        else:
            caller = self
        return Qobj(caller.data(t), dims=self.dims)

    @jit
    def data(self, t):
        if self.batched_data is not None:
            coeff = self._coeff(t)
            data = jnp.dot(self.batched_data, coeff)
            out = JaxArray(data)
        else:
            out = zeros_jaxdia(*self.shape)
        for data, coeff in self.sparse_part:
            out = out + data * coeff(t)
        return out

    @partial(jax.jit, donate_argnums=(3,))
    def matmul_data(self, t, y, out=None):
        if out is None and self.batched_data is not None:
            coeffs = self._coeff(t)
            out = JaxArray._fast_constructor(
                jnp.dot(jnp.dot(self.batched_data, coeffs), y._jxa),
                y.shape
            )
        elif type(out) is JaxArray and self.batched_data is not None:
            coeffs = self._coeff(t)
            out = JaxArray._fast_constructor(
                jnp.dot(jnp.dot(self.batched_data, coeffs), y._jxa) + out._jxa,
                y.shape
            )
        elif self.batched_data is not None:
            out = JaxArray._fast_constructor(
                jnp.dot(jnp.dot(self.batched_data, coeffs), y._jxa),
                y.shape
            ) + out
        elif out is None:
            out = zeros_jaxarray(*y.shape)

        for data, coeff in self.sparse_part:
            if isinstance(y, JaxArray):
                out = matmul_jaxdia_jaxarray_jaxarray(data, y, coeff(t), out)
            else:
                out = out + matmul(data, y, coeff(t))
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
        object.__setattr__(out, "shape", self.shape)
        return out

    def flatten(self):
        return (
            (self.batched_data, self.coeffs, self.sparse_part),
            {"dims": self.dims, "shape": self.shape}
        )

    @classmethod
    def unflatten(cls, aux_data, children):
        out = cls.__new__(cls)
        out.batched_data = children[0]
        out.coeffs = children[1]
        out.sparse_part = children[2]
        out.dims = aux_data["dims"]
        out.shape = aux_data["shape"]
        return out


jax.tree_util.register_pytree_node(
    JaxQobjEvo, JaxQobjEvo.flatten, JaxQobjEvo.unflatten
)
