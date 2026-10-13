import numpy as np
from qiskit.quantum_info import Operator

from qru_registerization.protocol_circuits import (
    alternating_protocol,
    score_block_encoding,
)
from qru_registerization.qru import qru_unitary, random_qru_params, signed_coordinate


def test_score_block_encoding_is_hermitian_unitary_and_has_correct_score_block():
    params = random_qru_params(2, seed=611, scale=0.6)
    axis = [0.3, -0.4, np.sqrt(0.75)]
    b = Operator(score_block_encoding(params, 2, -0.7, 0.4, axis, 1.1)).data
    np.testing.assert_allclose(b, b.conj().T, atol=1e-12)
    np.testing.assert_allclose(b @ b, np.eye(8), atol=1e-12)
    expected = [
        signed_coordinate(qru_unitary(params, -0.7 + 0.4 * x)[:, 0], axis)
        for x in range(4)
    ]
    np.testing.assert_allclose(b[:4, :4], np.diag(expected), atol=1e-12)


def test_fused_noncommuting_protocol_has_same_entire_unitary():
    params = random_qru_params(2, seed=611, scale=0.6)
    axis = [0.3, -0.4, np.sqrt(0.75)]
    full, _ = alternating_protocol(params, -0.7, 1.6, axis, k=4)
    fused, _ = alternating_protocol(params, -0.7, 1.6, axis, k=4, fused=True)
    np.testing.assert_allclose(Operator(full).data, Operator(fused).data, atol=2e-11)
