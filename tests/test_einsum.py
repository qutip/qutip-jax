import numpy as np
import pytest
import qutip as qt
from qutip.core.dimensions import einsum
from qutip_jax import JaxArray, einsum_jax
from qutip.tests.core.data.test_einsum import TestEinsum as _TestEinsum

_cases_mark = [
    m for m in _TestEinsum.test_einsum.pytestmark
    if m.name == 'parametrize' and 'subscripts' in m.args[0]
][0]


class TestEinsum:
    specialisations = [
        pytest.param(einsum_jax, JaxArray, JaxArray, id="JaxArray"),
    ]

    @pytest.mark.parametrize("einsum_func, data_type, out_type", specialisations)
    @pytest.mark.parametrize(*_cases_mark.args)
    def test_einsum(
        self,
        einsum_func,
        data_type,
        out_type,
        subscripts,
        shapes,
        perms,
        out_perm,
        out_shape,
        operands_data,
        expected_data,
    ):
        _TestEinsum().test_einsum(
            einsum_func,
            data_type,
            out_type,
            subscripts,
            shapes,
            perms,
            out_perm,
            out_shape,
            operands_data,
            expected_data,
        )


_cx = qt.Qobj(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dims=[[2, 2], [2, 2]],
).to("jax")
_cx_dag = _cx.dag()
_rho_01 = qt.ket2dm(
    qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
).to("jax")
_thermal_dm_2q = qt.tensor(
    qt.thermal_dm(2, 1), qt.thermal_dm(2, 1)
).to("jax")
_L1 = qt.spre(qt.tensor(qt.sigmax(), qt.sigmay())).to("jax")
_L2 = qt.spost(qt.tensor(qt.sigmaz(), qt.sigmax())).to("jax")
_oper_ket = qt.operator_to_vector(
    qt.tensor(qt.sigmax(), qt.sigmay())
).to("jax")
_oper_bra = _oper_ket.dag()


@pytest.mark.parametrize(["subscripts", "operands", "expected", "out_dims"], [
    pytest.param("ii", [qt.sigmaz().to("jax")], 0, None),
    pytest.param("ij,ji", [qt.sigmaz().to("jax"), qt.sigmaz().to("jax")], 2, None),
    pytest.param(
        "ijij", [_thermal_dm_2q], 1, None,
    ),
    pytest.param(
        "ikjl,jm->ikml",
        [qt.tensor(qt.sigmaz(), qt.sigmaz()).to("jax"), qt.sigmaz().to("jax")],
        qt.tensor(qt.qeye(2), qt.sigmaz()).to("jax"),
        None,
    ),
    pytest.param(
        "abcd,cde->abe",
        [_cx, qt.tensor(qt.basis(2, 0), qt.basis(2, 1)).to("jax")],
        qt.tensor(qt.basis(2, 0), qt.basis(2, 1)).to("jax"),
        None,
        id="ket_multiplication",
    ),
    pytest.param(
        "abcd,cdef->abef",
        [_cx, _rho_01],
        _cx @ _rho_01,
        None,
        id="density_matrix_left_multiplication",
    ),
    pytest.param(
        "cdef,efgh->cdgh",
        [_rho_01, _cx_dag],
        _rho_01 @ _cx_dag,
        None,
        id="density_matrix_right_multiplication",
    ),
    pytest.param(
        "abcd,cdef,efgh->abgh",
        [_cx, _rho_01, _cx_dag],
        _cx @ _rho_01 @ _cx_dag,
        None,
        id="density_matrix_conjugation",
    ),
    pytest.param(
        "ijklabcd,abcdmnop->ijklmnop",
        [_L1, _L2],
        _L1 @ _L2,
        _L1.dims,
        id="superoperator_multiplication",
    ),
    pytest.param(
        "yijkl,ijklz->",
        [_oper_bra, _oper_ket],
        complex(_oper_bra.to("dense") @ _oper_ket.to("dense")),
        None,
        id="operator_ket_bra_inner_product",
    ),
])
def test_qobj_einsum(subscripts, operands, expected, out_dims):
    res = einsum(subscripts, *operands, out_dims=out_dims)
    if isinstance(expected, qt.Qobj):
        assert isinstance(res.data, JaxArray)
        assert res.dims == expected.dims
        np.testing.assert_allclose(res.full(), expected.full(), atol=1e-12)
    else:
        assert np.isclose(res, expected, atol=1e-12)


@pytest.mark.parametrize(["subscripts", "operands"], [
    pytest.param(
        "ij", [qt.sigmax().to("jax")],
        id="single_operand_no_contraction",
    ),
    pytest.param(
        "ij->ji", [qt.sigmay().to("jax")],
        id="single_operand_transpose",
    ),
    pytest.param(
        "ijkl->kjil",
        [qt.tensor(qt.sigmam(), qt.sigmaz()).to("jax")],
        id="single_operand_permutation",
    ),
    pytest.param(
        "cdef,ghef->cdgh",
        [_rho_01, _cx],
        id="col_col_contraction",
    ),
    pytest.param(
        "ij,jk->ki",
        [qt.sigmax().to("jax"), qt.sigmaz().to("jax")],
        id="output_col_before_row",
    ),
    pytest.param(
        "ikjl,jm->mlik",
        [qt.tensor(qt.sigmaz(), qt.sigmaz()).to("jax"), qt.sigmaz().to("jax")],
        id="output_col_before_row_composite",
    ),
])
def test_einsum_rejects_implicit_transpose(subscripts, operands):
    with pytest.raises(ValueError):
        einsum(subscripts, *operands)


def test_qobj_einsum_jit_scalar():
    import jax

    z = qt.sigmaz().to("jax")

    @jax.jit
    def calc(z_data):
        op = qt.Qobj(z_data, dims=[[2], [2]])
        return einsum("ij,ji", op, op)

    val = calc(z.data)
    np.testing.assert_allclose(val, 2.0, atol=1e-12)



