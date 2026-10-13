from __future__ import annotations

import numpy as np


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("zero quaternion")
    # Antipodes are distinct in SU(2), but equivalent for SO(3) and projective action.
    q = q / norm
    return q if q[0] >= 0 else -q


def su2_to_quaternion(U: np.ndarray) -> np.ndarray:
    """Convert a 2x2 SU(2) rotation matrix to q=(w,x,y,z).

    Convention:
        U = [[w - i z, -y - i x],
             [y - i x,  w + i z]].

    This matches RX/RY/RZ(theta)=exp(-i theta sigma/2).
    A global phase is removed before conversion.
    """
    U = np.asarray(U, dtype=complex).reshape(2, 2)
    det = np.linalg.det(U)
    if abs(det) == 0:
        raise ValueError("matrix determinant is zero")
    U = U / np.sqrt(det)  # remove global phase; rotations should now be in SU(2)
    w = U[0, 0].real
    z = -U[0, 0].imag
    y = U[1, 0].real
    x = -U[1, 0].imag
    return normalize_quaternion(np.array([w, x, y, z], dtype=float))


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Return the SO(3) rotation matrix associated with q=(w,x,y,z)."""
    w, x, y, z = normalize_quaternion(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def bloch_from_quaternion(q: np.ndarray) -> np.ndarray:
    """Return r = R(q) e_z, the Bloch vector of U(q)|0>."""
    ez = np.array([0.0, 0.0, 1.0])
    return quaternion_to_rotation_matrix(q) @ ez


def relative_rotation_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Relative SO(3) rotation angle in [0, pi].

    Unit quaternions q and -q are identified. The factor two converts
    projective S^3 separation into the corresponding SO(3) rotation angle.
    """
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    dot = abs(float(np.dot(q1, q2)))
    dot = np.clip(dot, -1.0, 1.0)
    return float(2.0 * np.arccos(dot))


def quaternion_path_distances(qs: np.ndarray) -> np.ndarray:
    qs = np.asarray(qs, dtype=float)
    return np.array(
        [relative_rotation_angle(qs[i], qs[i + 1]) for i in range(len(qs) - 1)]
    )


def unitary_path_distances(unitaries: np.ndarray) -> np.ndarray:
    """Vectorized consecutive SO(3) rotation angles for an SU(2) path."""
    unitaries = np.asarray(unitaries, dtype=complex)
    if unitaries.ndim != 3 or unitaries.shape[1:] != (2, 2) or len(unitaries) < 2:
        raise ValueError("unitaries must have shape (n>=2, 2, 2)")
    determinants = np.linalg.det(unitaries)
    if np.any(np.abs(determinants) == 0.0):
        raise ValueError("unitaries must be nonsingular")
    su2 = unitaries / np.sqrt(determinants)[:, None, None]
    relatives = np.einsum(
        "nji,njk->nik",
        np.conj(su2[:-1]),
        su2[1:],
    )
    half_traces = 0.5 * np.abs(np.trace(relatives, axis1=1, axis2=2))
    return 2.0 * np.arccos(np.clip(half_traces, 0.0, 1.0))


def fubini_study_distance(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    """Bloch-angle distance ``2 d_FS`` for pure states, in ``[0, pi]``."""
    a = np.asarray(psi_a, dtype=complex).reshape(-1)
    b = np.asarray(psi_b, dtype=complex).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("state vectors must be non-zero")
    overlap = float(abs(np.vdot(a / na, b / nb)))
    overlap = float(np.clip(overlap, 0.0, 1.0))
    if 1.0 - overlap < 1e-14:
        return 0.0
    return float(2.0 * np.arccos(overlap))


def bloch_geodesic_distance(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Geodesic angle between unit Bloch vectors in [0, pi]."""
    a = np.asarray(r_a, dtype=float).reshape(3)
    b = np.asarray(r_b, dtype=float).reshape(3)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("Bloch vectors must be non-zero")
    dot = float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))
    if 1.0 - dot < 1e-14:
        return 0.0
    return float(np.arccos(dot))


def quotient_quaternion_distance(U_a: np.ndarray, U_b: np.ndarray) -> float:
    """Distance on SU(2)/U(1) for states prepared from |0>.

    This equals min_phi d_rot(U_a, U_b Rz(phi)) and is evaluated through
    the equivalent Fubini--Study distance between U_a|0> and U_b|0>.
    """
    Ua = np.asarray(U_a, dtype=complex).reshape(2, 2)
    Ub = np.asarray(U_b, dtype=complex).reshape(2, 2)
    ket0 = np.array([1.0, 0.0], dtype=complex)
    return fubini_study_distance(Ua @ ket0, Ub @ ket0)


def quotient_distance_from_quaternions(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Return the explicit fibre-minimized distance on ``SU(2)/U(1)``.

    For right multiplication of ``q_b`` by the z-axis fibre quaternion, the
    Euclidean overlap with ``q_a`` is ``A cos(phi/2)+B sin(phi/2)``.  Its
    maximum absolute value is ``sqrt(A**2+B**2)``, which gives the exact
    projective Bloch angle without constructing state vectors.
    """
    first = normalize_quaternion(q_a)
    second = normalize_quaternion(q_b)
    w_a, x_a, y_a, z_a = first
    w_b, x_b, y_b, z_b = second
    cosine_coefficient = float(first @ second)
    sine_coefficient = float(-w_a * z_b + x_a * y_b - y_a * x_b + z_a * w_b)
    overlap = float(np.sqrt(cosine_coefficient**2 + sine_coefficient**2))
    return float(2.0 * np.arccos(np.clip(overlap, 0.0, 1.0)))


def quotient_quaternion_path_distances(qs: np.ndarray) -> np.ndarray:
    """Consecutive explicit quotient distances from an ``S^3`` lift."""
    values = np.asarray(qs, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or len(values) < 2:
        raise ValueError("qs must have shape (n>=2, 4)")
    return np.asarray(
        [
            quotient_distance_from_quaternions(values[index], values[index + 1])
            for index in range(len(values) - 1)
        ],
        dtype=float,
    )


def body_angular_velocity(unitaries: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Return body-frame angular velocity for an SU(2) path.

    The returned vector satisfies
    ``U(x)^dagger U'(x) = -i (omega(x).sigma) / 2``.
    Second-order finite differences are projected back onto ``su(2)`` to
    suppress numerical differentiation drift.
    """
    U = np.asarray(unitaries, dtype=complex)
    x = np.asarray(xs, dtype=float).reshape(-1)
    if U.ndim != 3 or U.shape[1:] != (2, 2) or len(U) < 3:
        raise ValueError("unitaries must have shape (n>=3, 2, 2)")
    if len(x) != len(U) or np.any(np.diff(x) <= 0):
        raise ValueError("xs must be strictly increasing and match unitaries")

    dU = np.gradient(U, x, axis=0, edge_order=2)
    paulis = np.asarray(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, -1.0j], [1.0j, 0.0]],
            [[1.0, 0.0], [0.0, -1.0]],
        ],
        dtype=complex,
    )
    omega = np.empty((len(U), 3), dtype=float)
    for index, (Ui, dUi) in enumerate(zip(U, dU, strict=True)):
        lie = Ui.conj().T @ dUi
        lie = 0.5 * (lie - lie.conj().T)
        hermitian_generator = 1.0j * lie
        omega[index] = [
            float(np.real(np.trace(hermitian_generator @ sigma))) for sigma in paulis
        ]
    return omega
