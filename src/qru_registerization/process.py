"""Exact retained channels for multi-use dilations with arbitrary interventions.

Global tensor order is (input branch, environment=2*y+system, target).
Retained basis is (input branch, target). Choi vectors use column stacking.
All workspace is retained coherently or renewed explicitly, never postselected.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import block_diag

from .channel import estimator_matrix, trace_norm
from .interface import directional_amplitude_unitary, phase_taper_amplitudes
from .qru import qru_unitary, rz, signed_coordinate

PAULI = np.array(
    [
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ],
    complex,
)


def rotation(axis, angle):
    axis = np.asarray(axis, float)
    if axis.shape != (3,) or not np.isclose(np.linalg.norm(axis), 1):
        raise ValueError("axis must be a unit three-vector")
    h = np.einsum("j,jab->ab", axis, PAULI)
    return np.cos(angle) * np.eye(2) - 1j * np.sin(angle) * h


def choi_from_superoperator(superoperator):
    d = round(np.sqrt(len(superoperator)))
    return (
        superoperator.reshape(d, d, d, d, order="F")
        .transpose(
            0,
            2,
            1,
            3,
        )
        .reshape(d * d, d * d, order="F")
    )


def superoperator_from_choi(choi):
    d = round(np.sqrt(len(choi)))
    return (
        choi.reshape(d, d, d, d, order="F")
        .transpose(
            0,
            2,
            1,
            3,
        )
        .reshape(d * d, d * d, order="F")
    )


def kraus_choi(kraus):
    vectors = np.array([k.ravel(order="F") for k in kraus])
    return vectors.T @ vectors.conj()


@dataclass
class ProcessResult:
    choi: np.ndarray
    ideal: np.ndarray
    zero_kraus: np.ndarray | None
    kraus: np.ndarray | None
    scores: np.ndarray
    estimates: np.ndarray
    distributions: np.ndarray


def process_channel(
    params,
    xs,
    axis,
    pulses,
    interventions=None,
    phase_bits=3,
    calibrated=True,
    mode="shared",
    fibre=None,
):
    """pulses are (unit Pauli-axis, angle); gates act between adjacent uses.

    shared: keep the same environment coherently, trace it only at the end.
    fresh: trace and initialize a new environment after every use.
    Arbitrary interventions can mix both input and target.
    """
    xs = np.asarray(xs, float)
    n = len(xs)
    d = 2 * n
    if n < 1 or mode not in ("shared", "fresh") or not pulses or phase_bits < 2:
        raise ValueError("invalid process contract")
    gates = (
        [np.eye(d)] * (len(pulses) - 1)
        if interventions is None
        else list(interventions)
    )
    if len(gates) != len(pulses) - 1:
        raise ValueError("one intervention between adjacent uses required")
    for gate in gates:
        if np.shape(gate) != (d, d) or not np.allclose(
            gate.conj().T @ gate,
            np.eye(d),
            atol=1e-10,
        ):
            raise ValueError("interventions must be retained-register unitaries")
    phases = np.zeros(n) if fibre is None else np.asarray(fibre, float)
    if phases.shape != (n,) or not np.all(np.isfinite(phases)):
        raise ValueError("one finite fibre phase per branch required")
    size = 2**phase_bits
    env_size = 2 * size
    scores = np.array(
        [signed_coordinate(qru_unitary(params, float(x))[:, 0], axis) for x in xs]
    )
    estimates = -np.cos(2 * np.pi * np.arange(size) / size)
    if calibrated:
        estimates /= np.cos(np.pi / size)
    estimators = [
        estimator_matrix(
            directional_amplitude_unitary(
                qru_unitary(params, float(x)) @ rz(float(phi)),
                axis,
            ),
            phase_taper_amplitudes(phase_bits, "sine"),
        )
        for x, phi in zip(xs, phases, strict=True)
    ]
    distributions = np.array(
        [np.sum(abs(e[:, 0].reshape(size, 2)) ** 2, axis=1) for e in estimators]
    )
    iso = np.zeros((n, env_size, 2, d), complex)
    for x in range(n):
        iso[x, 0, :, 2 * x : 2 * x + 2] = np.eye(2)
    superoperator = np.eye(d * d, dtype=complex)
    ideal = np.eye(d, dtype=complex)
    for j, (target_axis, angle) in enumerate(pulses):
        controlled = block_diag(
            *[rotation(target_axis, angle * g) for g in estimates for _ in range(2)]
        )
        dilation = [
            np.kron(e.conj().T, np.eye(2)) @ controlled @ np.kron(e, np.eye(2))
            for e in estimators
        ]
        if mode == "shared":
            for x in range(n):
                iso[x] = (dilation[x] @ iso[x].reshape(2 * env_size, d)).reshape(
                    env_size,
                    2,
                    d,
                )
        else:
            local_kraus = np.array(
                [
                    block_diag(
                        *[
                            w.reshape(env_size, 2, env_size, 2)[a, :, 0, :]
                            for w in dilation
                        ]
                    )
                    for a in range(env_size)
                ]
            )
            superoperator = (
                sum(np.kron(k.conj(), k) for k in local_kraus) @ superoperator
            )
        ideal = block_diag(*[rotation(target_axis, angle * s) for s in scores]) @ ideal
        if j < len(gates):
            gate = gates[j]
            ideal = gate @ ideal
            if mode == "shared":
                retained = iso.transpose(0, 2, 1, 3).reshape(d, env_size * d)
                iso = (gate @ retained).reshape(n, 2, env_size, d).transpose(0, 2, 1, 3)
            else:
                superoperator = np.kron(gate.conj(), gate) @ superoperator
    if mode == "shared":
        kraus = iso.transpose(1, 0, 2, 3).reshape(env_size, d, d)
        return ProcessResult(
            kraus_choi(kraus),
            ideal,
            kraus[0],
            kraus,
            scores,
            estimates,
            distributions,
        )
    return ProcessResult(
        choi_from_superoperator(superoperator),
        ideal,
        None,
        None,
        scores,
        estimates,
        distributions,
    )


@lru_cache(maxsize=4)
def _diamond_problem(d):
    import cvxpy as cp

    j = cp.Parameter((d * d, d * d), hermitian=True)
    rho = cp.Variable((d, d), hermitian=True)
    x = cp.Variable((d * d, d * d), hermitian=True)
    weight = cp.kron(rho, np.eye(d))
    primal = cp.Problem(
        cp.Maximize(cp.real(cp.trace(j @ x))),
        [rho >> 0, cp.trace(rho) == 1, weight - x >> 0, weight + x >> 0],
    )
    y = cp.Variable((d * d, d * d), hermitian=True)
    t = cp.Variable()
    partial = sum(y[a::d, a::d] for a in range(d))
    dual = cp.Problem(
        cp.Minimize(t),
        [
            y - j >> 0,
            y + j >> 0,
            t * np.eye(d) - partial >> 0,
        ],
    )
    return j, rho, primal, y, dual


def diamond_choi(delta):
    """Hermiticity-preserving diamond SDP; full norm, floating-point witnesses."""
    delta = np.asarray(delta, complex)
    d = round(np.sqrt(len(delta)))
    if delta.shape != (d * d, d * d) or not np.allclose(delta, delta.conj().T):
        raise ValueError("Hermitian square Choi difference required")
    if np.max(abs(delta)) < 1e-12:
        return {"diamond_lower": 0.0, "diamond_upper": 0.0, "sdp_gap": 0.0}
    parameter, rho, primal, y, dual = _diamond_problem(d)
    parameter.value = (delta + delta.conj().T) / 2
    for problem in [primal, dual]:
        problem.solve(
            solver="CLARABEL",
            tol_gap_abs=1e-8,
            tol_feas=1e-8,
            tol_gap_rel=1e-8,
            max_iter=150,
        )
        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Diamond SDP failed: {problem.status}")
    eigen, vectors = np.linalg.eigh((rho.value + rho.value.conj().T) / 2)
    eigen = np.maximum(eigen, 0)
    eigen /= sum(eigen)
    root = (vectors * np.sqrt(eigen)) @ vectors.conj().T
    weight = np.kron(root, np.eye(d))
    lower = trace_norm(weight @ delta @ weight)
    yy = (y.value + y.value.conj().T) / 2
    repair = (
        max(0, -np.linalg.eigvalsh(yy - delta)[0], -np.linalg.eigvalsh(yy + delta)[0])
        + 1e-9
    )
    partial = sum(yy[a::d, a::d] for a in range(d)) + d * repair * np.eye(d)
    upper = float(np.linalg.eigvalsh(partial)[-1])
    return {"diamond_lower": lower, "diamond_upper": upper, "sdp_gap": upper - lower}


def process_metrics(result, diamond=True):
    d = len(result.ideal)
    vector = result.ideal.ravel(order="F")
    ideal = np.outer(vector, vector.conj())
    partial = sum(result.choi[a::d, a::d] for a in range(d))
    values = {
        "choi_fidelity": float(np.vdot(vector, result.choi @ vector).real / d**2),
        "choi_trace_norm": trace_norm(result.choi - ideal) / d,
        "tp_max_error": float(np.max(abs(partial - np.eye(d)))),
        "cp_min_eigenvalue": float(np.linalg.eigvalsh(result.choi)[0]),
    }
    if result.zero_kraus is not None:
        leakage = np.eye(d) - result.zero_kraus.conj().T @ result.zero_kraus
        values["work_leakage_worst"] = max(0, float(np.linalg.eigvalsh(leakage)[-1]))
    if diamond:
        values.update(diamond_choi(result.choi - ideal))
    return values
