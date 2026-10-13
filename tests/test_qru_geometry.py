from __future__ import annotations

import numpy as np

from qru_registerization.qru import (
    bloch_vector,
    evaluate_trajectory,
    random_qru_params,
    ry,
    rz,
)
from qru_registerization.quaternion_geometry import (
    bloch_from_quaternion,
    bloch_geodesic_distance,
    fubini_study_distance,
    quaternion_path_distances,
    quotient_distance_from_quaternions,
    quotient_quaternion_distance,
    relative_rotation_angle,
    su2_to_quaternion,
    unitary_path_distances,
)


def test_trajectory_is_consistent_across_representations() -> None:
    xs = np.linspace(-1.0, 1.0, 25)
    trajectory = evaluate_trajectory(random_qru_params(depth=4, seed=3, scale=0.7), xs)
    assert np.allclose(np.linalg.norm(trajectory.quaternions, axis=1), 1.0)
    reconstructed = np.asarray(
        [bloch_from_quaternion(q) for q in trajectory.quaternions]
    )
    assert np.allclose(trajectory.bloch, reconstructed, atol=1e-10)
    assert np.allclose(
        np.einsum("nji,njk->nik", np.conj(trajectory.unitaries), trajectory.unitaries),
        np.eye(2),
        atol=1e-10,
    )


def test_raw_rotation_detects_pure_gauge_motion() -> None:
    base = ry(0.7)
    first = base @ rz(-1.1)
    second = base @ rz(1.4)
    raw = relative_rotation_angle(su2_to_quaternion(first), su2_to_quaternion(second))
    quotient = quotient_quaternion_distance(first, second)
    assert raw > 1.0
    assert quotient < 1e-10


def test_quotient_matches_state_and_bloch_distances() -> None:
    first = ry(0.31) @ rz(-0.6)
    second = rz(0.23) @ ry(-0.42)
    ket0 = np.array([1.0, 0.0], dtype=complex)
    psi_first, psi_second = first @ ket0, second @ ket0
    quotient = quotient_quaternion_distance(first, second)
    assert np.isclose(
        quotient, fubini_study_distance(psi_first, psi_second), atol=1e-10
    )
    assert np.isclose(
        quotient,
        bloch_geodesic_distance(bloch_vector(psi_first), bloch_vector(psi_second)),
        atol=1e-10,
    )


def test_quaternion_sign_is_projective() -> None:
    first = su2_to_quaternion(ry(0.3) @ rz(-0.4))
    second = su2_to_quaternion(ry(-0.7) @ rz(0.2))
    assert np.isclose(
        relative_rotation_angle(first, second),
        relative_rotation_angle(first, -second),
    )


def test_vectorized_unitary_path_distances_match_quaternions() -> None:
    xs = np.linspace(-1.0, 1.0, 61)
    trajectory = evaluate_trajectory(random_qru_params(depth=4, seed=17, scale=0.6), xs)
    assert np.allclose(
        unitary_path_distances(trajectory.unitaries),
        quaternion_path_distances(trajectory.quaternions),
        atol=2e-12,
        rtol=2e-12,
    )


def test_explicit_quaternion_fibre_quotient_matches_state_quotient() -> None:
    rng = np.random.default_rng(144)
    for _ in range(20):
        angles = rng.uniform(-np.pi, np.pi, size=6)
        first = rz(angles[0]) @ ry(angles[1]) @ rz(angles[2])
        second = rz(angles[3]) @ ry(angles[4]) @ rz(angles[5])
        explicit = quotient_distance_from_quaternions(
            su2_to_quaternion(first), su2_to_quaternion(second)
        )
        state = quotient_quaternion_distance(first, second)
        assert np.isclose(explicit, state, atol=2e-10, rtol=2e-10)

        gauged = second @ rz(float(rng.uniform(-4.0, 4.0)))
        gauged_explicit = quotient_distance_from_quaternions(
            su2_to_quaternion(first), su2_to_quaternion(gauged)
        )
        assert np.isclose(gauged_explicit, explicit, atol=2e-10, rtol=2e-10)
