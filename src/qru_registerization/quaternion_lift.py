"""Faithful SU(2) quaternion lifts, including the sign relevant to controls.

The historical quaternion_geometry.su2_to_quaternion canonicalizes antipodes
and removes determinant phase: it is suitable for projective/SO(3) geometry.
These APIs instead require SU(2), never identify q with -q, and never remove
an unknown U(2) phase. U(q)=q0 I-i(q1 X+q2 Y+q3 Z).
"""

import numpy as np


def su2_lift_quaternion(unitary):
    u = np.asarray(unitary, complex)
    if u.shape != (2, 2) or not np.all(np.isfinite(u)):
        raise ValueError("finite 2-by-2 SU(2) matrix required")
    if not np.allclose(u.conj().T @ u, np.eye(2), atol=1e-10, rtol=0):
        raise ValueError("matrix must be unitary")
    if not np.isclose(np.linalg.det(u), 1, atol=1e-10, rtol=0):
        raise ValueError("matrix must be SU(2); specify U(2) phase separately")
    q = np.array([u[0, 0].real, -u[1, 0].imag, u[1, 0].real, -u[0, 0].imag])
    return q / np.linalg.norm(q)


def quaternion_to_su2(quaternion):
    q = np.asarray(quaternion, float)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) == 0:
        raise ValueError("finite nonzero four-component quaternion required")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[w - 1j * z, -y - 1j * x], [y - 1j * x, w + 1j * z]])


def quaternion_product(first, second):
    """Hamilton product: U(q*r)=U(q)U(r), with both antipodes retained."""
    a, b = (np.asarray(q, float) for q in (first, second))
    if a.shape != (4,) or b.shape != (4,) or not np.all(np.isfinite([a, b])):
        raise ValueError("two finite four-component quaternions required")
    return np.r_[
        a[0] * b[0] - a[1:] @ b[1:],
        a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:]),
    ]
