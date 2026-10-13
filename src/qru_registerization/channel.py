"""Exact finite coherent channels, without state sampling or postselection.

Basis: i=2*x+z (input branch, Z-target eigenstate), environment=2*y+system.
Matrix powers here accelerate classical verification; they never define gate cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .interface import (
    amplitude_grover_operator,
    directional_amplitude_unitary,
    phase_index_to_signed_code,
    phase_taper_amplitudes,
)
from .qru import qru_unitary, rz, signed_coordinate


@dataclass
class ChannelResult:
    gram: np.ndarray
    ideal: np.ndarray
    environments: np.ndarray
    scores: np.ndarray
    estimates: np.ndarray
    distributions: np.ndarray
    zero_probability: np.ndarray
    system_one_probability: np.ndarray


def _preparation(amplitudes):
    a = np.asarray(amplitudes, dtype=float)
    if a.ndim != 1 or not np.all(np.isfinite(a)):
        raise ValueError("real finite amplitude vector required")
    if not np.isclose(np.linalg.norm(a), 1):
        raise ValueError("amplitudes must have unit norm")
    v = -a.copy()
    v[0] += 1
    norm = np.vdot(v, v).real
    return (
        np.eye(len(a)) if norm < 1e-24 else np.eye(len(a)) - 2 * np.outer(v, v) / norm
    )


def estimator_matrix(amplitude, amplitudes):
    """Unitary E=(F_N ⊗ I) sum_n |n><n|⊗Q^n (P_a ⊗ A)."""
    size = len(amplitudes)
    powers = np.empty((size, 2, 2), complex)
    powers[0] = np.eye(2)
    q = amplitude_grover_operator(amplitude)
    for n in range(1, size):
        powers[n] = q @ powers[n - 1]
    before = np.kron(_preparation(amplitudes), amplitude).reshape(size, 2, 2 * size)
    controlled = np.einsum("nij,njk->nik", powers, before)
    return (np.fft.fft(controlled, axis=0) / np.sqrt(size)).reshape(2 * size, 2 * size)


def coherent_channel(
    params,
    xs,
    axis,
    phase_bits=3,
    magnitude_bits=2,
    gamma=0.8,
    taper="tophat",
    fibre=None,
    cleanup=True,
    dephase=0.0,
    amplitudes=None,
    control_scores=None,
):
    """Return the whole retained input-target CPTP Schur channel.

    dephase is the probability of complete phase-word dephasing immediately
    after E. Its forgotten measurement label remains an environment register.
    All branches are retained; zero_probability is a diagnostic, not conditioning.
    control_scores overrides the signed decoder with a finite phase-word table.
    These are rotation scores, which need not lie in [-1,1]. Passing this table
    does not infer a synthesis cost; use the matching structural circuit.
    """
    xs = np.asarray(xs, float)
    if xs.ndim != 1 or len(xs) < 1 or not np.all(np.isfinite(xs)):
        raise ValueError("xs must be a nonempty finite vector")
    if phase_bits < 1 or magnitude_bits < 1 or not np.isfinite(gamma):
        raise ValueError("invalid precision or gamma")
    if not 0 <= dephase <= 1:
        raise ValueError("dephase must lie in [0,1]")
    phases = np.zeros(len(xs)) if fibre is None else np.asarray(fibre, float)
    if phases.shape != xs.shape or not np.all(np.isfinite(phases)):
        raise ValueError("one finite fibre phase per branch required")
    a = (
        phase_taper_amplitudes(phase_bits, taper)
        if amplitudes is None
        else np.asarray(amplitudes, float)
    )
    size = 2**phase_bits
    if a.shape != (size,):
        raise ValueError("amplitude length does not match phase_bits")
    estimates = np.array(
        [
            phase_index_to_signed_code(y, phase_bits, magnitude_bits)[2]
            for y in range(size)
        ]
    )
    if control_scores is not None:
        estimates = np.asarray(control_scores, float)
        if estimates.shape != (size,) or not np.all(np.isfinite(estimates)):
            raise ValueError("one finite control score per phase word required")
    d = 2 * len(xs)
    width = 2 * size * (1 + size) if dephase else 2 * size
    env = np.zeros((d, width), complex)
    zero = np.zeros(d)
    system_one = np.zeros(d)
    distributions = []
    scores = []
    for xindex, x in enumerate(xs):
        u = qru_unitary(params, float(x))
        scores.append(signed_coordinate(u[:, 0], axis))
        amplitude = directional_amplitude_unitary(u @ rz(float(phases[xindex])), axis)
        e = estimator_matrix(amplitude, a)
        initial = e[:, 0]
        distributions.append(np.sum(abs(initial.reshape(size, 2)) ** 2, axis=1))
        for z in range(2):
            index = 2 * xindex + z
            controlled = (
                np.repeat(np.exp(-1j * gamma * estimates * (1 - 2 * z)), 2) * initial
            )
            coherent = e.conj().T @ controlled if cleanup else controlled
            env[index, : 2 * size] = np.sqrt(1 - dephase) * coherent
            zero[index] = (1 - dephase) * abs(coherent[0]) ** 2
            system_one[index] = (1 - dephase) * np.sum(abs(coherent[1::2]) ** 2)
            if dephase:
                marked = np.zeros((2 * size, size), complex)
                for y in range(size):
                    marked[2 * y : 2 * y + 2, y] = controlled[2 * y : 2 * y + 2]
                if cleanup:
                    marked = e.conj().T @ marked
                env[index, 2 * size :] = np.sqrt(dephase) * marked.T.ravel()
                zero[index] += dephase * np.sum(abs(marked[0]) ** 2)
                system_one[index] += dephase * np.sum(abs(marked[1::2]) ** 2)
    scores = np.asarray(scores)
    ideal = np.exp(-1j * gamma * np.repeat(scores, 2) * np.tile([1, -1], len(xs)))
    return ChannelResult(
        env @ env.conj().T,
        ideal,
        env,
        scores,
        estimates,
        np.asarray(distributions),
        zero,
        system_one,
    )


def trace_norm(matrix):
    return float(np.sum(abs(np.linalg.eigvalsh((matrix + matrix.conj().T) / 2))))


@lru_cache(maxsize=12)
def _diamond_problem(d):
    import cvxpy as cp

    delta = cp.Parameter((d, d), hermitian=True)
    p = cp.Variable(d, nonneg=True)
    x = cp.Variable((d, d), hermitian=True)
    primal = cp.Problem(
        cp.Maximize(cp.real(cp.trace(delta @ x))),
        [cp.sum(p) == 1, cp.diag(p) - x >> 0, cp.diag(p) + x >> 0],
    )
    y = cp.Variable((d, d), hermitian=True)
    t = cp.Variable()
    dual = cp.Problem(
        cp.Minimize(t), [y - delta >> 0, y + delta >> 0, cp.real(cp.diag(y)) <= t]
    )
    return delta, p, primal, y, dual


def schur_diamond(delta):
    """Watrous SDP reduced by diagonal-unitary covariance to d-by-d blocks.

    Return a physical witness lower bound and repaired dual upper bound.
    These are floating-point certificates, not interval-arithmetic proofs.
    Convention: full diamond norm, no factor 1/2.
    """
    delta = np.asarray(delta, complex)
    if delta.ndim != 2 or delta.shape[0] != delta.shape[1]:
        raise ValueError("square Hermitian Schur difference required")
    if not np.allclose(delta, delta.conj().T, atol=1e-10):
        raise ValueError("Hermitian Schur difference required")
    if np.max(abs(delta)) < 1e-13:
        return {"diamond_lower": 0.0, "diamond_upper": 0.0, "sdp_gap": 0.0}
    parameter, p, primal, y, dual = _diamond_problem(len(delta))
    parameter.value = (delta + delta.conj().T) / 2
    for problem in [primal, dual]:
        problem.solve(
            solver="CLARABEL",
            tol_gap_abs=1e-9,
            tol_feas=1e-9,
            tol_gap_rel=1e-9,
            max_iter=200,
        )
        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"SDP did not converge: {problem.status}")
    weights = np.maximum(p.value, 0)
    weights /= sum(weights)
    lower = trace_norm(np.sqrt(weights)[:, None] * delta * np.sqrt(weights)[None, :])
    yy = (y.value + y.value.conj().T) / 2
    repair = (
        max(0.0, -np.linalg.eigvalsh(yy - delta)[0], -np.linalg.eigvalsh(yy + delta)[0])
        + 1e-10
    )
    upper = float(max(np.diag(yy).real) + repair)
    return {"diamond_lower": lower, "diamond_upper": upper, "sdp_gap": upper - lower}


def channel_metrics(result, diamond=True):
    ideal_gram = np.outer(result.ideal, result.ideal.conj())
    delta = result.gram - ideal_gram
    d = len(delta)
    mu = float(
        np.max(
            result.distributions @ result.estimates**2
            - 2 * result.scores * (result.distributions @ result.estimates)
            + result.scores**2
        )
    )
    fidelity = float(np.vdot(result.ideal, result.gram @ result.ideal).real / d**2)
    values = {
        "choi_fidelity": fidelity,
        "choi_trace_norm": trace_norm(delta) / d,
        "worst_work_leakage": float(1 - min(result.zero_probability)),
        "mean_work_leakage": float(1 - np.mean(result.zero_probability)),
        "max_system_one": float(max(result.system_one_probability)),
        "score_mse_max": max(mu, 0.0),
        "cp_min_eigenvalue": float(np.linalg.eigvalsh(result.gram)[0]),
        "tp_max_error": float(max(abs(np.diag(result.gram) - 1))),
    }
    if diamond:
        values.update(schur_diamond(delta))
    return values
