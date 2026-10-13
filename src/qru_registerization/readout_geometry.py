from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qru import QRUTrajectory, evaluate_trajectory, random_qru_params
from .quaternion_geometry import (
    body_angular_velocity,
    fubini_study_distance,
    quaternion_path_distances,
    quotient_quaternion_path_distances,
    unitary_path_distances,
)


@dataclass(frozen=True)
class LiftVelocity:
    body: np.ndarray
    raw_density: np.ndarray
    horizontal_density: np.ndarray
    vertical_density: np.ndarray


@dataclass(frozen=True)
class AxisSelection:
    axis: np.ndarray | None
    eigenvalues: np.ndarray
    eigengap: float
    identifiable: bool
    energy: float
    method: str


def canonical_axis(axis: np.ndarray) -> np.ndarray:
    """Normalize an eigenvector and fix its otherwise arbitrary sign."""
    axis = np.asarray(axis, dtype=float).reshape(3)
    if not np.all(np.isfinite(axis)):
        raise ValueError("axis must be finite")
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise ValueError("axis must be non-zero")
    axis = axis / norm
    index = int(np.argmax(np.abs(axis)))
    return -axis if axis[index] < 0.0 else axis


def axis_angle(first: np.ndarray, second: np.ndarray) -> float:
    """Unsigned angle in radians between two unoriented axes."""
    first = canonical_axis(first)
    second = canonical_axis(second)
    return float(np.arccos(np.clip(abs(float(first @ second)), 0.0, 1.0)))


def right_fibre_lift(unitaries: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Apply ``U_i -> U_i Rz(phases_i)`` without changing the state ray."""
    unitaries = np.asarray(unitaries, dtype=complex)
    phases = np.asarray(phases, dtype=float).reshape(-1)
    if unitaries.ndim != 3 or unitaries.shape[1:] != (2, 2):
        raise ValueError("unitaries must have shape (n, 2, 2)")
    if len(phases) != len(unitaries) or not np.all(np.isfinite(phases)):
        raise ValueError("phases must be finite and match unitaries")
    rotations = np.zeros_like(unitaries)
    rotations[:, 0, 0] = np.exp(-0.5j * phases)
    rotations[:, 1, 1] = np.exp(0.5j * phases)
    return np.einsum("nij,njk->nik", unitaries, rotations)


def lift_velocity(unitaries: np.ndarray, xs: np.ndarray) -> LiftVelocity:
    """Decompose an SU(2) lift into quotient and vertical kinetic densities."""
    body = body_angular_velocity(unitaries, xs)
    horizontal = np.sum(body[:, :2] ** 2, axis=1)
    vertical = body[:, 2] ** 2
    return LiftVelocity(
        body=body,
        raw_density=np.sum(body**2, axis=1),
        horizontal_density=horizontal,
        vertical_density=vertical,
    )


def gauge_steering_profile(
    xs: np.ndarray,
    bloch_path: np.ndarray,
    reference_axis: np.ndarray,
    *,
    width: float = 0.32,
    interior_fraction: float = 0.15,
    speed_quantile: float = 0.60,
) -> tuple[np.ndarray, float, float]:
    """Build a smooth train-only fibre phase that stresses a raw-lift axis.

    The phase transition is centred on a resolved interior tangent whose
    direction is least aligned with ``reference_axis``. Right multiplication
    by ``Rz(alpha * profile)`` changes only the lift, never the state ray.
    """
    xs = np.asarray(xs, dtype=float).reshape(-1)
    bloch_path = np.asarray(bloch_path, dtype=float)
    if bloch_path.shape != (len(xs), 3) or len(xs) < 3:
        raise ValueError("bloch_path must have shape (len(xs)>=3, 3)")
    if np.any(np.diff(xs) <= 0.0):
        raise ValueError("xs must be strictly increasing")
    if width <= 0.0:
        raise ValueError("width must be positive")
    if not 0.0 <= interior_fraction < 0.5:
        raise ValueError("interior_fraction must lie in [0, 0.5)")
    if not 0.0 <= speed_quantile <= 1.0:
        raise ValueError("speed_quantile must lie in [0, 1]")

    axis = canonical_axis(reference_axis)
    dx = np.diff(xs)
    tangents = np.diff(bloch_path, axis=0) / dx[:, None]
    speeds = np.linalg.norm(tangents, axis=1)
    directions = np.divide(
        tangents,
        speeds[:, None],
        out=np.zeros_like(tangents),
        where=speeds[:, None] > 1e-12,
    )
    alignment = np.abs(directions @ axis)
    midpoints = 0.5 * (xs[:-1] + xs[1:])
    span = float(np.ptp(xs))
    interior = (midpoints > xs[0] + interior_fraction * span) & (
        midpoints < xs[-1] - interior_fraction * span
    )
    resolved = speeds >= np.quantile(speeds, speed_quantile)
    candidates = np.flatnonzero(interior & resolved)
    if candidates.size == 0:
        raise RuntimeError("trajectory has no resolved interior tangent")
    target = int(candidates[np.argmin(alignment[candidates])])
    center = float(midpoints[target])
    return np.tanh((xs - center) / width), center, float(alignment[target])


def projective_speed_density(states: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Return ``4 g_FS = ||r'||^2`` from a sampled pure-state path."""
    states = np.asarray(states, dtype=complex)
    xs = np.asarray(xs, dtype=float).reshape(-1)
    if states.ndim != 2 or len(states) < 3:
        raise ValueError("states must have shape (n>=3, dimension)")
    if len(xs) != len(states) or np.any(np.diff(xs) <= 0):
        raise ValueError("xs must be strictly increasing and match states")
    norms = np.linalg.norm(states, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("states must be non-zero")
    states = states / norms[:, None]
    projectors = np.einsum("ni,nj->nij", states, np.conj(states))
    derivative = np.gradient(projectors, xs, axis=0, edge_order=2)
    density = 2.0 * np.sum(np.abs(derivative) ** 2, axis=(1, 2))
    return np.maximum(np.real(density), 0.0)


def _grid(path_length: int, xs: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    grid = (
        np.arange(path_length, dtype=float)
        if xs is None
        else np.asarray(xs, dtype=float).reshape(-1)
    )
    if len(grid) != path_length or np.any(np.diff(grid) <= 0):
        raise ValueError("xs must be strictly increasing and match the path")
    return grid, np.diff(grid)


def tangent_covariance(
    bloch_path: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    xs: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``sum w_i D_i D_i^T dx_i / sum w_i dx_i``."""
    bloch_path = np.asarray(bloch_path, dtype=float)
    if bloch_path.ndim != 2 or bloch_path.shape[1] != 3 or len(bloch_path) < 3:
        raise ValueError("bloch_path must have shape (n>=3, 3)")
    _, dx = _grid(len(bloch_path), xs)
    tangents = np.diff(bloch_path, axis=0) / dx[:, None]
    if weights is None:
        weights = np.ones(len(tangents), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != len(tangents):
            raise ValueError("weights must have one value per path increment")
        if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite and non-negative")
    quadrature = weights * dx
    denominator = float(np.sum(quadrature))
    if denominator <= 0.0:
        return np.zeros((3, 3), dtype=float)
    return np.einsum("i,ij,ik->jk", quadrature, tangents, tangents) / denominator


def tangent_energy(
    bloch_path: np.ndarray,
    axis: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    xs: np.ndarray | None = None,
) -> float:
    covariance = tangent_covariance(bloch_path, weights=weights, xs=xs)
    axis = canonical_axis(axis)
    return float(axis @ covariance @ axis)


def _principal_axis(
    covariance: np.ndarray,
    *,
    method: str,
    energy_tol: float,
    eigengap_tol: float,
) -> AxisSelection:
    covariance = np.asarray(covariance, dtype=float).reshape(3, 3)
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    energy = float(max(values[0], 0.0))
    eigengap = float((values[0] - values[1]) / (abs(values[0]) + energy_tol))
    identifiable = bool(energy > energy_tol and eigengap > eigengap_tol)
    axis = canonical_axis(vectors[:, 0]) if identifiable else None
    return AxisSelection(axis, values, eigengap, identifiable, energy, method)


def select_axis(
    bloch_path: np.ndarray,
    *,
    weights: np.ndarray,
    xs: np.ndarray,
    method: str,
    energy_tol: float = 1e-12,
    eigengap_tol: float = 1e-6,
) -> AxisSelection:
    return _principal_axis(
        tangent_covariance(bloch_path, weights=weights, xs=xs),
        method=method,
        energy_tol=energy_tol,
        eigengap_tol=eigengap_tol,
    )


def increment_speed(increments: np.ndarray, xs: np.ndarray) -> np.ndarray:
    increments = np.asarray(increments, dtype=float).reshape(-1)
    xs = np.asarray(xs, dtype=float).reshape(-1)
    if len(xs) != len(increments) + 1 or np.any(np.diff(xs) <= 0):
        raise ValueError("xs must contain one more point than increments")
    if np.any(increments < 0.0) or not np.all(np.isfinite(increments)):
        raise ValueError("increments must be finite and non-negative")
    return increments / np.diff(xs)


def state_increments(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=complex)
    if states.ndim != 2 or len(states) < 2:
        raise ValueError("states must have shape (n>=2, dimension)")
    return np.asarray(
        [
            fubini_study_distance(states[index], states[index + 1])
            for index in range(len(states) - 1)
        ]
    )


def quotient_weights(
    states: np.ndarray, xs: np.ndarray, power: float = 2.0
) -> np.ndarray:
    return increment_speed(state_increments(states), xs) ** power


def quaternion_quotient_weights(
    quaternions: np.ndarray, xs: np.ndarray, power: float = 2.0
) -> np.ndarray:
    """Lift-based quotient-speed weights from explicit fibre minimization."""
    return increment_speed(quotient_quaternion_path_distances(quaternions), xs) ** power


def raw_lift_weights(
    quaternions: np.ndarray, xs: np.ndarray, power: float = 2.0
) -> np.ndarray:
    return increment_speed(quaternion_path_distances(quaternions), xs) ** power


def raw_unitary_weights(
    unitaries: np.ndarray, xs: np.ndarray, power: float = 2.0
) -> np.ndarray:
    """Raw-lift speed weights computed directly from an SU(2) path."""
    return increment_speed(unitary_path_distances(unitaries), xs) ** power


def select_quotient_axis(
    bloch_path: np.ndarray,
    states: np.ndarray,
    xs: np.ndarray,
    *,
    power: float = 2.0,
    energy_tol: float = 1e-12,
    eigengap_tol: float = 1e-6,
) -> AxisSelection:
    return select_axis(
        bloch_path,
        weights=quotient_weights(states, xs, power),
        xs=xs,
        method="quotient",
        energy_tol=energy_tol,
        eigengap_tol=eigengap_tol,
    )


def select_quaternion_quotient_axis(
    bloch_path: np.ndarray,
    quaternions: np.ndarray,
    xs: np.ndarray,
    *,
    power: float = 2.0,
    energy_tol: float = 1e-12,
    eigengap_tol: float = 1e-6,
) -> AxisSelection:
    """Select the quotient axis using only the quaternion lift and Bloch path."""
    return select_axis(
        bloch_path,
        weights=quaternion_quotient_weights(quaternions, xs, power),
        xs=xs,
        method="quaternion_quotient",
        energy_tol=energy_tol,
        eigengap_tol=eigengap_tol,
    )


def select_raw_lift_axis(
    bloch_path: np.ndarray,
    quaternions: np.ndarray,
    xs: np.ndarray,
    *,
    power: float = 2.0,
    energy_tol: float = 1e-12,
    eigengap_tol: float = 1e-6,
) -> AxisSelection:
    return select_axis(
        bloch_path,
        weights=raw_lift_weights(quaternions, xs, power),
        xs=xs,
        method="raw_lift",
        energy_tol=energy_tol,
        eigengap_tol=eigengap_tol,
    )


def select_raw_lift_axis_from_unitaries(
    bloch_path: np.ndarray,
    unitaries: np.ndarray,
    xs: np.ndarray,
    *,
    power: float = 2.0,
    energy_tol: float = 1e-12,
    eigengap_tol: float = 1e-6,
    method: str = "raw_lift",
) -> AxisSelection:
    return select_axis(
        bloch_path,
        weights=raw_unitary_weights(unitaries, xs, power),
        xs=xs,
        method=method,
        energy_tol=energy_tol,
        eigengap_tol=eigengap_tol,
    )


def select_best_pauli_axis(
    bloch_path: np.ndarray,
    *,
    weights: np.ndarray,
    xs: np.ndarray,
) -> AxisSelection:
    """Select the most energetic Cartesian Pauli axis."""
    axes = np.eye(3)
    energies = np.asarray(
        [tangent_energy(bloch_path, axis, weights=weights, xs=xs) for axis in axes]
    )
    order = np.argsort(energies)[::-1]
    best, runner_up = int(order[0]), int(order[1])
    margin = float((energies[best] - energies[runner_up]) / (energies[best] + 1e-12))
    return AxisSelection(
        axis=axes[best],
        eigenvalues=energies[order],
        eigengap=margin,
        identifiable=bool(energies[best] > 1e-12),
        energy=float(energies[best]),
        method="best_pauli",
    )


def readout_values(bloch_path: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.asarray(bloch_path, dtype=float) @ canonical_axis(axis)


def orient_axis_to_target(
    axis: np.ndarray,
    bloch_path: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Orient an intrinsic axis line using one reference split only."""
    path = np.asarray(bloch_path, dtype=float)
    values = np.asarray(target, dtype=float).reshape(-1)
    if path.shape != (len(values), 3):
        raise ValueError("bloch_path and target must have matching lengths")
    direction = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("axis must be finite and non-zero")
    direction = direction / norm
    positive = path @ direction
    positive_loss = float(np.mean((positive - values) ** 2))
    negative_loss = float(np.mean((-positive - values) ** 2))
    return (direction, 1) if positive_loss <= negative_loss else (-direction, -1)


@dataclass(frozen=True)
class ReadoutCase:
    """A QRU trajectory paired with its gauge-invariant tangent readout."""

    params: np.ndarray
    trajectory: QRUTrajectory
    weights: np.ndarray
    selection: AxisSelection

    @property
    def axis(self) -> np.ndarray:
        if self.selection.axis is None:
            raise RuntimeError("readout axis is not identifiable")
        return self.selection.axis

    @property
    def signed(self) -> np.ndarray:
        return readout_values(self.trajectory.bloch, self.axis)


def analyze_readout_case(
    params: np.ndarray,
    xs: np.ndarray,
    *,
    energy_tol: float = 1e-12,
    eigengap_tol: float = 1e-5,
) -> ReadoutCase:
    trajectory = evaluate_trajectory(params, xs)
    selection = select_quotient_axis(
        trajectory.bloch,
        trajectory.states,
        trajectory.xs,
        energy_tol=energy_tol,
        eigengap_tol=eigengap_tol,
    )
    if selection.axis is None:
        raise RuntimeError("QRU readout is not identifiable")
    return ReadoutCase(
        params=np.asarray(params, dtype=float),
        trajectory=trajectory,
        weights=quotient_weights(trajectory.states, trajectory.xs),
        selection=selection,
    )


def random_readout_case(
    *,
    depth: int,
    seed: int,
    n_x: int,
    scale: float = 0.9,
    energy_tol: float = 1e-12,
    eigengap_tol: float = 1e-5,
) -> ReadoutCase:
    params = random_qru_params(depth=depth, seed=seed, scale=scale)
    try:
        return analyze_readout_case(
            params,
            np.linspace(-np.pi, np.pi, n_x),
            energy_tol=energy_tol,
            eigengap_tol=eigengap_tol,
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"QRU readout is not identifiable (depth={depth}, seed={seed})"
        ) from error
