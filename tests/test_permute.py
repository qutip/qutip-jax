import qutip
from qutip_jax import JaxArray


class TestPermute:
    def test_psi(self):
        A = qutip.basis(3, 0, dtype="jax")
        B = qutip.basis(5, 4, dtype="jax")
        C = qutip.basis(4, 2, dtype="jax")
        psi = qutip.tensor(A, B, C)
        psi2 = psi.permute([2, 0, 1])
        assert psi2 == qutip.tensor(C, A, B)
        assert isinstance(psi2.data, JaxArray)

        psi_bra = psi.dag()
        psi2_bra = psi_bra.permute([2, 0, 1])
        assert psi2_bra == qutip.tensor(C, A, B).dag()
        assert isinstance(psi2_bra.data, JaxArray)

        for _ in range(3):
            A = qutip.rand_ket(3, dtype="jax")
            B = qutip.rand_ket(4, dtype="jax")
            C = qutip.rand_ket(5, dtype="jax")
            psi = qutip.tensor(A, B, C)
            psi2 = psi.permute([1, 0, 2])
            assert psi2 == qutip.tensor(B, A, C)
            assert isinstance(psi2.data, JaxArray)

            psi_bra = psi.dag()
            psi2_bra = psi_bra.permute([1, 0, 2])
            assert psi2_bra == qutip.tensor(B, A, C).dag()
            assert isinstance(psi2_bra.data, JaxArray)

    def test_oper(self):
        A = qutip.fock_dm(3, 0, dtype="jax")
        B = qutip.fock_dm(5, 4, dtype="jax")
        C = qutip.fock_dm(4, 2, dtype="jax")
        rho = qutip.tensor(A, B, C)
        rho2 = rho.permute([2, 0, 1])
        assert rho2 == qutip.tensor(C, A, B)
        assert isinstance(rho2.data, JaxArray)

        for _ in range(3):
            A = qutip.rand_dm(3, dtype="jax")
            B = qutip.rand_dm(4, dtype="jax")
            C = qutip.rand_dm(5, dtype="jax")
            rho = qutip.tensor(A, B, C)
            rho2 = rho.permute([1, 0, 2])
            assert rho2 == qutip.tensor(B, A, C)
            assert isinstance(rho2.data, JaxArray)

            rho_vec = qutip.operator_to_vector(rho)
            rho2_vec = rho_vec.permute([[1, 0, 2], [4, 3, 5]])
            assert rho2_vec == qutip.operator_to_vector(qutip.tensor(B, A, C))
            assert isinstance(rho2_vec.data, JaxArray)

            rho_vec_bra = qutip.operator_to_vector(rho).dag()
            rho2_vec_bra = rho_vec_bra.permute([[1, 0, 2], [4, 3, 5]])
            assert (
                rho2_vec_bra
                == qutip.operator_to_vector(qutip.tensor(B, A, C)).dag()
            )
            assert isinstance(rho2_vec_bra.data, JaxArray)

    def test_super(self):
        for _ in range(3):
            super_dims = [3, 5, 4]
            U = qutip.rand_unitary(super_dims, dtype="jax")
            Unew = U.permute([2, 1, 0])
            S_tens = qutip.to_super(U)
            S_tens_new = qutip.to_super(Unew)
            assert S_tens_new == S_tens.permute([[2, 1, 0], [5, 4, 3]])
            assert isinstance(S_tens_new.data, JaxArray)
