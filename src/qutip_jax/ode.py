import diffrax
import equinox as eqx
from qutip.solver.integrator import Integrator
import jax
import jax.numpy as jnp
from qutip.solver.solver_base import Solver

__all__ = []


def _cplx2float(arr):
    return jnp.hstack([arr.real, arr.imag])


def _float2cplx(arr):
    N = arr.shape[0] // 2
    return arr[:N] + 1j * arr[N:]


class DiffraxIntegrator(Integrator, eqx.Module):
    method: str = "diffrax"
    supports_blackbox: bool = False  # No feedback support
    support_time_dependant: bool = True
    _is_set: bool = False

    integrator_options: frozenset = frozenset({
        "dt0",
        "solver",
        "stepsize_controller",
        "max_steps",
    })
    _options: dict = None
    name: str

    ODEsystem: diffrax.term.ODETerm
    system: object
    solver: diffrax.AbstractSolver
    solver_state: diffrax.custom_types.PyTree
    state: jax.numpy.ndarray
    t: float
    _back: tuple

    def __init__(self, system, options):
        self.system = system
        self._back = (np.inf, None)
        self.solver = getattr(diffrax, options.pop("solver", "Dopri5"))()
        self.ODEsystem = diffrax.ODEterm(self.dstate)
        self.solver_state = None
        self._options = {
            key: val
            for key, val in options.items()
            if key in self.integrator_options
        }

    @staticmethod
    def dstate(t, y, args):
        state = _float2cplx(y)
        H = args[0]
        d_state = H.matmul_data(t, JaxArrar(y))
        return _cplx2float(d_state._jxa)

    def _prepare(self):
        pass

    def set_state(self, t, state0):
        self.solver_state = None
        self.t = t
        if not isinstance(state0, JaxArray):
            state0 = _data.to(JaxArray, state0)
        self.state = _cplx2float(JaxArray._jxa)
        self._is_set = True

    def get_state(self, copy):
        return self.t, JaxArray(_float2cplx(self.state))

    def integrate(self, t, copy=True):
        sol = diffrax.diffeqsolve(
            self.ODEsystem,
            self.solver,
            t0=self.t, t1=t,
            y0=self.state,
            saveat=diffrax.SaveAt(t1=True, solver_state=True),
            solver_state=self.solver_state,
            args=(self.system,),
            **self.options
        )
        self.t = t
        self.state = sol.ys[0, :]
        self.solver_state = sol.solver_state
        return self.get_state()


Solver.add_integrator(DiffraxIntegrator, 'diffrax')
