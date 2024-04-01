from qutip_jax import JaxArray, JaxDia
from qutip_jax.create import *
import pytest
import numpy as np


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1), id="minimal"),
        pytest.param((10, 10), id="oper"),
        pytest.param((10, 1), id="column"),
        pytest.param((1, 10), id="row"),
        pytest.param((10, 2), id="tall"),
        pytest.param((2, 10), id="wide"),
    ],
)
@pytest.mark.parametrize("func", [zeros_jaxarray, zeros_jaxdia])
def test_zeros(func, shape):
    """Tests the function that creates zero JAX arrays."""
    base = func(shape[0], shape[1])
    nd = base.to_array()
    # assert isinstance(base, JaxArray)
    assert base.shape == shape
    assert nd.shape == shape
    assert np.count_nonzero(nd) == 0


@pytest.mark.parametrize("dimension", [1, 5, 100])
@pytest.mark.parametrize(
    "scale",
    [None, 2, -0.1, 1.5, 1.5 + 1j],
    ids=["none", "int", "negative", "float", "complex"],
)
@pytest.mark.parametrize("func", [identity_jaxarray, identity_jaxdia])
def test_identity(func, dimension, scale):
    """Tests the function that creates identity JAX arrays."""
    # scale=None is testing that the default value returns the identity.
    base = func(dimension) if scale is None else func(dimension, scale)
    nd = base.to_array()
    numpy_test = np.eye(dimension, dtype=np.complex128)
    if scale is not None:
        numpy_test *= scale
    # assert isinstance(base, JaxArray)
    assert base.shape == (dimension, dimension)
    assert np.count_nonzero(nd - numpy_test) == 0


@pytest.mark.parametrize(
    ["diagonals", "offsets", "shape"],
    [
        pytest.param([2j, 3, 5, 9], None, None, id="main diagonal"),
        pytest.param([1], None, None, id="1x1"),
        pytest.param([[0.2j, 0.3]], None, None, id="main diagonal list"),
        pytest.param([0.2j, 0.3], 2, None, id="superdiagonal"),
        pytest.param([0.2j, 0.3], -2, None, id="subdiagonal"),
        pytest.param(
            [[0.2, 0.3, 0.4], [0.1, 0.9]], [-2, 3], None, id="two diagonals"
        ),
        pytest.param([1, 2, 3], 0, (3, 5), id="main wide"),
        pytest.param([1, 2, 3], 0, (5, 3), id="main tall"),
        pytest.param([[1, 2, 3], [4, 5]], [-1, -2], (4, 8), id="two wide sub"),
        pytest.param(
            [[1, 2, 3, 4], [4, 5, 4j, 1j]], [1, 2], (4, 8), id="two wide super"
        ),
        pytest.param([[1, 2, 3], [4, 5]], [1, 2], (8, 4), id="two tall super"),
        pytest.param(
            [[1, 2, 3, 4], [4, 5, 4j, 1j]], [-1, -2], (8, 4), id="two tall sub"
        ),
        pytest.param(
            [[1, 2, 3], [4, 5, 6], [1, 2]],
            [1, -1, -2],
            (4, 4),
            id="out of order",
        ),
        pytest.param(
            [[1, 2, 3], [4, 5, 6], [1, 2]],
            [1, 1, -2],
            (4, 4),
            id="sum duplicates",
        ),
    ],
)
@pytest.mark.parametrize(
    ["func", "dtype"], [(diag_jaxarray, JaxArray), (diag_jaxdia, JaxDia)]
)
def test_diags(func, dtype, diagonals, offsets, shape):
    """Tests the function that creates diagonal JAX arrays."""
    base = func(diagonals, offsets, shape)
    # Build numpy version test.
    if not isinstance(diagonals[0], list):
        diagonals = [diagonals]
    offsets = np.atleast_1d(offsets if offsets is not None else [0])
    if shape is None:
        size = len(diagonals[0]) + abs(offsets[0])
        shape = (size, size)
    test = np.zeros(shape, dtype=np.complex128)
    for diagonal, offset in zip(diagonals, offsets):
        test[np.where(np.eye(*shape, k=offset) == 1)] += diagonal
    # assert isinstance(base, JaxArray)
    assert base.shape == shape
    assert isinstance(base, dtype)
    np.testing.assert_allclose(base.to_array(), test, rtol=1e-10)


@pytest.mark.parametrize(
    ["shape", "position", "value"],
    [
        pytest.param((1, 1), (0, 0), None, id="minimal"),
        pytest.param((10, 10), (5, 5), 1.0j, id="on diagonal"),
        pytest.param((10, 10), (1, 5), 1.0, id="upper"),
        pytest.param((10, 10), (5, 1), 2.0, id="lower"),
        pytest.param((10, 1), (5, 0), None, id="column"),
        pytest.param((1, 10), (0, 5), -5.0j, id="row"),
        pytest.param((10, 2), (5, 1), 1 + 2j, id="tall"),
        pytest.param((2, 10), (1, 5), 10, id="wide"),
    ],
)
@pytest.mark.parametrize("func", [one_element_jaxarray, one_element_jaxdia])
def test_one_element(func, shape, position, value):
    """Tests the function that creates single element JAX arrays."""
    test = np.zeros(shape, dtype=np.complex128)
    if value is None:
        base = func(shape, position)
        test[position] = 1.0 + 0.0j
    else:
        base = func(shape, position, value)
        test[position] = value
    # assert isinstance(base, JaxArray)
    assert base.shape == shape
    assert np.allclose(base.to_array(), test, atol=1e-10)


@pytest.mark.parametrize(
    ["shape", "position", "value"],
    [
        pytest.param((0, 0), (0, 0), None, id="zero shape"),
        pytest.param((10, -2), (5, 0), 1.0j, id="neg shape"),
        pytest.param((10, 10), (10, 5), 1.0, id="outside"),
        pytest.param((10, 10), (5, -1), 2.0, id="outside neg"),
    ],
)
@pytest.mark.parametrize("func", [one_element_jaxarray, one_element_jaxdia])
def test_one_element_error(func, shape, position, value):
    """Tests for wrong inputs to the one_element_jaxarray function."""
    with pytest.raises(ValueError) as exc:
        base = func(shape, position, value)
    assert str(exc.value).startswith(
        "Position of the elements" " out of bound: "
    )
