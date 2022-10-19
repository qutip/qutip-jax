class JaxCoeff(eqx.Module):
    func: callable
    args: dict

    def __init__(self, func, args):
        self.func = func
        self.args = args


class JaxQobjEvo(eqx.Module):
    H: jax.numpy.ndarray
    coeffs: list
    funcs: list

    def __init__(self, qobjevo):
        as_list = to_list(qobjevo)
        self.funcs = []
        qobjs = []
        self.coeffs = []

        for part in as_list:
            if not isinstance(part[0], qt.Qobj):
                self.funcs.append(part[0])
            elif part[0].norm() > 0:
                qobjs.append(part[0])
                self.coeffs.append(part[1])

        if qobjs and self.funcs:
            raise ValueError()
        elif qobjs:
            shape = qobjs[0].shape
            self.H = jnp.zeros(shape + (len(qobjs),), dtype=np.complex128)
            for i, qobj in enumerate(qobjs):
                self.H = self.H.at[:, :, i].set(qobj.to("jax").data._jxa)
                if self.coeffs[i] == 1:
                    self.coeffs[i] = qt.coefficient(lambda t: 1.)


    def coeff(self, t):
        list_coeffs = [coeff(t) for coeff in self.coeffs]
        return jnp.array(list_coeffs, dtype=np.complex128)


    def matmul_data(self, t, y, args=None):
        print("In")
        if self.funcs:
            out = jnp.zeros(y.shape, dtype=np.complex128)
            for element in self.funcs:
                out = out + element.matmul_data_t(t, y)
        else:
            coeffs = self.coeff(t)
            out = JaxArray(jnp.dot(jnp.dot(self.H, coeffs), y._jxa))
        return out
