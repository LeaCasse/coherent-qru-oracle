import numpy as np
import pytest
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

from qru_registerization.qru import rx, ry, rz
from qru_registerization.quaternion_geometry import su2_to_quaternion
from qru_registerization.quaternion_lift import (
    quaternion_product,
    quaternion_to_su2,
    su2_lift_quaternion,
)


def test_lift_preserves_antipodes_that_projective_conversion_identifies():
    u = rz(0.7) @ ry(-0.8)
    np.testing.assert_allclose(su2_to_quaternion(u), su2_to_quaternion(-u))
    np.testing.assert_allclose(su2_lift_quaternion(u), -su2_lift_quaternion(-u))
    for matrix in [u, -u, -np.eye(2)]:
        np.testing.assert_allclose(
            quaternion_to_su2(su2_lift_quaternion(matrix)), matrix, atol=1e-12
        )
    controlled_positive = Operator(UnitaryGate(np.eye(2)).control()).data
    controlled_negative = Operator(UnitaryGate(-np.eye(2)).control()).data
    assert np.linalg.norm(controlled_positive - controlled_negative) > 1


def test_hamilton_product_and_right_fibre_reconstruct_true_operator():
    for a in [-4.0, -0.5, 1.0, 5.0]:
        u = rz(a) @ ry(0.3) @ rx(-0.2)
        fibre = rz(1.7)
        product = quaternion_product(su2_lift_quaternion(u), su2_lift_quaternion(fibre))
        np.testing.assert_allclose(quaternion_to_su2(product), u @ fibre, atol=1e-12)


def test_u2_phase_cannot_be_silently_discarded_by_lift_api():
    with pytest.raises(ValueError, match="SU"):
        su2_lift_quaternion(np.exp(0.3j) * rx(0.7))
    with pytest.raises(ValueError, match="unitary"):
        su2_lift_quaternion(np.diag([1, 2]))
