import qutip.settings as settings
import numpy as np
import scipy
import pytest
import qutip
from qutip_jax import JaxArray, solve_jaxarray, svd_jaxarray
import jax

from qutip.core import data as _data
from qutip.core.data import Data, Dense, CSR


class TestSolve:
    def op_numpy(self, A, b):
        return np.linalg.solve(A, b)

    def _gen_op(self, N, dtype):
        return qutip.rand_unitary(N, dtype=dtype).data

    def _gen_ket(self, N, dtype):
        return qutip.rand_ket(N, dtype=dtype).data

    @pytest.mark.parametrize("method", ["solve", "lstsq"])
    def test_mathematically_correct_JaxArray(self, method):
        A = self._gen_op(10, JaxArray)
        b = self._gen_ket(10, JaxArray)
        expected = self.op_numpy(A.to_array(), b.to_array())
        test = solve_jaxarray(A, b, method)
        test1 = _data.solve(A, b, method)

        assert test.shape == expected.shape
        np.testing.assert_allclose(
            test.to_array(), expected, atol=1e-7, rtol=1e-7
        )
        np.testing.assert_allclose(
            test1.to_array(), expected, atol=1e-7, rtol=1e-7
        )

    def test_incorrect_shape_non_square(self):
        key = jax.random.PRNGKey(1)
        A = JaxArray(jax.random.uniform(shape=(2, 3), key=key))
        b = JaxArray(jax.random.uniform(shape=(3, 1), key=key))
        with pytest.raises(ValueError):
            test1 = solve_jaxarray(A, b)

    def test_incorrect_shape_mismatch(self):
        key = jax.random.PRNGKey(1)
        A = JaxArray(jax.random.uniform(shape=(3, 3), key=key))
        b = JaxArray(jax.random.uniform(shape=(2, 1), key=key))
        with pytest.raises(ValueError):
            test1 = solve_jaxarray(A, b)


class TestSVD:
    def op_numpy(self, A):
        return jax.numpy.linalg.svd(A)

    def _gen_dm(self, N, rank, dtype):
        return qutip.rand_dm(N, rank=rank, dtype=dtype).data

    def _gen_non_square(self, N):
        mat = np.random.randn(N, N // 2)
        for i in range(N // 2):
            # Ensure no zeros singular values
            mat[i, i] += 5
        return _data.Dense(mat)

    @pytest.mark.parametrize("shape", ["square", "non-square"])
    def test_mathematically_correct_svd_jaxarray(self, shape):
        if shape == "square":
            matrix = self._gen_dm(10, 6, JaxArray)
        else:
            matrix = _data.to(JaxArray, self._gen_non_square(12))
        u, s, v = self.op_numpy(matrix.to_array())
        test_U, test_S, test_V = svd_jaxarray(matrix, True)
        only_S = _data.svd(matrix, False)

        assert sum(test_S > 1e-10) == 6
        # columns are definied up to a sign
        np.testing.assert_allclose(
            np.abs(test_U.to_array()), np.abs(u), atol=1e-7, rtol=1e-7
        )
        # rows are definied up to a sign
        np.testing.assert_allclose(
            np.abs(test_V.to_array()), np.abs(v), atol=1e-7, rtol=1e-7
        )
        np.testing.assert_allclose(test_S, s, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(only_S, s, atol=1e-7, rtol=1e-7)

        s_as_matrix = _data.diag[JaxArray](
            [test_S], 0, (test_U.shape[1], test_V.shape[0])
        )

        np.testing.assert_allclose(
            matrix.to_array(),
            (test_U @ s_as_matrix @ test_V).to_array(),
            atol=1e-7,
            rtol=1e-7,
        )
