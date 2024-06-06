import diffrax
from qutip.solver.integrator import Integrator
import jax
import jax.numpy as jnp
from qutip.solver.mcsolve import MCSolver
from qutip.solver.mesolve import MESolver
from qutip.solver.sesolve import SESolver
from qutip.core import data as _data
from qutip_jax import JaxArray
from qutip_jax.qobjevo import JaxQobjEvo

__all__ = ["DiffraxIntegrator"]


@jax.jit
def _cplx2float(arr):
    return jnp.stack([arr.real, arr.imag])


@jax.jit
def _float2cplx(arr):
    return arr[0] + 1j * arr[1]


@jax.jit
def dstate(t, y, args):
    state = _float2cplx(y)
    H, = args
    d_state = H.matmul_data(t, JaxArray(state))
    return _cplx2float(d_state._jxa)


class DiffraxIntegrator(Integrator):
    method: str = "diffrax"
    supports_blackbox: bool = False  # No feedback support
    support_time_dependant: bool = True
    integrator_options: dict = {
        "dt0": 0.0001,
        "solver": diffrax.Tsit5(),
        "stepsize_controller": diffrax.PIDController(atol=1e-8, rtol=1e-6),
        "max_steps": 100000,
    }

    def __init__(self, system, options):
        self.system = JaxQobjEvo(system)
        self._is_set = False  # get_state can be used and return a valid state.
        self._options = self.integrator_options.copy()
        self.options = options
        self.ODEsystem = diffrax.ODETerm(dstate)
        self.solver_state = None
        self.name = f"{self.method}: {self.options['solver']}"

    def _prepare(self):
        pass

    def set_state(self, t, state0):
        self.solver_state = None
        self.t = t
        if not isinstance(state0, JaxArray):
            state0 = _data.to(JaxArray, state0)
        self.state = _cplx2float(state0._jxa)
        self._is_set = True

    def get_state(self, copy=False):
        return self.t, JaxArray(_float2cplx(self.state))

    def integrate(self, t, copy=False, **kwargs):
        if kwargs:
            self.arguments(kwargs)
        sol = diffrax.diffeqsolve(
            self.ODEsystem,
            t0=self.t,
            t1=t,
            y0=self.state,
            saveat=diffrax.SaveAt(t1=True, solver_state=True),
            solver_state=self.solver_state,
            args=(self.system,),
            **self._options,
        )
        self.t = t
        self.state = sol.ys[0, :]
        self.solver_state = sol.solver_state
        return self.get_state()

    def arguments(self, args):
        self.system = self.system.arguments(args)
        self.solver_state = None

    def _flatten(self):
        children = (
            self.system,
            self._options,
            self.solver_state,
        )
        if self._is_set:
            children += (self.t, self.state)
        aux_data = {
            "_is_set": self._is_set,
        }
        return (children, aux_data)

    @classmethod
    def _unflatten(cls, aux_data, children):
        out = cls.__new__(cls)
        out.system = children[0]
        out._options = children[1]
        out.solver_state = children[2]
        out._is_set = aux_data["_is_set"]
        if out._is_set:
            out.t = children[3]
            out.state = children[4]
        out.ODEsystem = diffrax.ODETerm(out.dstate)
        return out

    @property
    def options(self):
        """
        Supported options by diffrax method:

        dt0 : float, default=0.0001
            Initial step size.

        solver: AbstractSolver, default=Tsit5(),
            ODE solver instance from diffrax.

        stepsize_controller: AbstractStepSizeController, default=ConstantStepSize()
            Step size controller from diffrax.

        max_steps: int, default=100000
            Maximum number of steps for the integration.
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)

MCSolver.add_integrator(DiffraxIntegrator, "diffrax")
MESolver.add_integrator(DiffraxIntegrator, "diffrax")
SESolver.add_integrator(DiffraxIntegrator, "diffrax")
jax.tree_util.register_pytree_node(
    DiffraxIntegrator, DiffraxIntegrator._flatten, DiffraxIntegrator._unflatten
)
