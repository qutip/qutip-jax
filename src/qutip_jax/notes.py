%%cython --cplus -I /home/eric/miniconda3/lib/python3.9/site-packages/numpy/core/include

from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.cy._element cimport *
import qutip as qt

def to_list(QobjEvo self):
    out = []
    for element in self.elements:
        if isinstance(element, _ConstantElement):
            out.append([element.qobj(0), 1])
        elif isinstance(element, _EvoElement):
            coeff = (<_EvoElement> element)._coefficient
            out.append([element.qobj(0), coeff])
        else:
            out.append([element])
    return out
