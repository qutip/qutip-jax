from qutip import Qobj, QobjEvo
from jax import tree_util
from qutip.core.cy._element import _EvoElement, _ConstantElement
from qutip_jax.qobjevo import JaxJitCoeff


def qobj_tree_flatten(qobj):
    children = (qobj.to("jax").data,)
    aux_data = {
        "dims": qobj.dims,
        "type": qobj.type,
        "superrep": qobj.superrep,
        # Attribute that depend on the data are not safe to be set.
        "_isherm": None,
        "_isunitary": None,
    }
    return (children, aux_data)


def qobj_tree_unflatten(aux_data, children):
    out = Qobj.__new__(Qobj)
    out._data = children[0]
    for attr, val in aux_data.items():
        setattr(out, attr, val)
    return out


tree_util.register_pytree_node(
    Qobj, qobj_tree_flatten, qobj_tree_unflatten
)
