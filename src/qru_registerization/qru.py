from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quaternion_geometry import su2_to_quaternion

PAULIS = np.asarray(
    [
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, -1.0j], [1.0j, 0.0]],
        [[1.0, 0.0], [0.0, -1.0]],
    ],
    dtype=complex,
)


@dataclass(frozen=True)
class QRUDifferentials:
    """Exact state derivatives needed for state and lift-aware training.

    ``state_dx`` is the input derivative of the prepared state. The two
    parameter derivative arrays use flattened parameter order
    ``[alpha_0, beta_0, gamma_0, delta_0, ...]``.
    """

    states: np.ndarray
    state_dx: np.ndarray
    state_dparams: np.ndarray
    state_dx_dparams: np.ndarray
    bloch: np.ndarray
    bloch_dx: np.ndarray
    bloch_dparams: np.ndarray
    vertical_velocity: np.ndarray
    vertical_velocity_dparams: np.ndarray


def rx(theta: float) -> np.ndarray:
    cosine = np.cos(theta / 2.0)
    sine = np.sin(theta / 2.0)
    return np.asarray([[cosine, -1.0j * sine], [-1.0j * sine, cosine]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    cosine = np.cos(theta / 2.0)
    sine = np.sin(theta / 2.0)
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.asarray(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=complex,
    )


def random_qru_params(
    depth: int,
    seed: int = 7,
    scale: float = 1.0,
) -> np.ndarray:
    """Sample ``[alpha, beta, gamma, delta]`` for each QRU layer."""
    if depth <= 0:
        raise ValueError("depth must be positive")
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("scale must be finite and non-negative")
    return np.random.default_rng(seed).normal(0.0, scale, size=(depth, 4))


def qru_unitary(params: np.ndarray, x: float) -> np.ndarray:
    """Return the QRU unitary at input ``x``.

    Each layer acts as ``Rz(gamma) Ry(beta*x + delta) Rx(alpha)``.
    """
    parameters = np.asarray(params, dtype=float)
    if parameters.ndim != 2 or parameters.shape[1] != 4:
        raise ValueError("params must have shape (depth, 4)")
    if not np.all(np.isfinite(parameters)) or not np.isfinite(x):
        raise ValueError("params and x must be finite")

    unitary = np.eye(2, dtype=complex)
    for alpha, beta, gamma, delta in parameters:
        layer = rz(gamma) @ ry(beta * float(x) + delta) @ rx(alpha)
        unitary = layer @ unitary
    return unitary


def normalize_state(state: np.ndarray) -> np.ndarray:
    vector = np.asarray(state, dtype=complex).reshape(2)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("zero vector is not a quantum state")
    return vector / norm


def bloch_vector(state: np.ndarray) -> np.ndarray:
    """Return ``(<X>, <Y>, <Z>)`` for a pure single-qubit state."""
    vector = normalize_state(state)
    return np.asarray(
        [np.vdot(vector, pauli @ vector).real for pauli in PAULIS], dtype=float
    )


def signed_coordinate(state_or_bloch: np.ndarray, axis: np.ndarray) -> float:
    """Return the signed coordinate ``axis.T @ r``."""
    value = np.asarray(state_or_bloch)
    bloch = (
        bloch_vector(value)
        if value.shape == (2,)
        else np.asarray(value, dtype=float).reshape(3)
    )
    direction = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("axis must be non-zero")
    return float((direction / norm) @ bloch)


def _rotation_derivative(rotation: np.ndarray, pauli: np.ndarray) -> np.ndarray:
    return (-0.5j * pauli) @ rotation


def _rotation_second_derivative(rotation: np.ndarray) -> np.ndarray:
    return -0.25 * rotation


def qru_state_differentials(params: np.ndarray, xs: np.ndarray) -> QRUDifferentials:
    """Return exact input/parameter derivatives of a QRU state path.

    The vertical lift velocity is

    ``omega_z = -2 Im <psi | d psi / dx>``.

    It is the fibre component in the decomposition
    ``||omega||^2 = ||r'||^2 + omega_z^2``. Its parameter Jacobian is
    evaluated analytically from the mixed state derivatives.
    """
    parameters = np.asarray(params, dtype=float)
    grid = np.asarray(xs, dtype=float).reshape(-1)
    if parameters.ndim != 2 or parameters.shape[1] != 4:
        raise ValueError("params must have shape (depth, 4)")
    if grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("xs must contain finite values")
    if not np.all(np.isfinite(parameters)):
        raise ValueError("params must be finite")

    depth = len(parameters)
    n_params = 4 * depth
    n_x = len(grid)
    states = np.empty((n_x, 2), dtype=complex)
    state_dx = np.empty((n_x, 2), dtype=complex)
    state_dparams = np.empty((n_x, n_params, 2), dtype=complex)
    state_dx_dparams = np.empty((n_x, n_params, 2), dtype=complex)
    bloch = np.empty((n_x, 3), dtype=float)
    bloch_dx = np.empty((n_x, 3), dtype=float)
    bloch_dparams = np.empty((n_x, 3, n_params), dtype=float)
    vertical_velocity = np.empty(n_x, dtype=float)
    vertical_velocity_dparams = np.empty((n_x, n_params), dtype=float)
    ket0 = np.array([1.0, 0.0], dtype=complex)

    for sample_index, x in enumerate(grid):
        state = ket0.copy()
        derivative_x = np.zeros(2, dtype=complex)
        derivatives = np.zeros((n_params, 2), dtype=complex)
        mixed_derivatives = np.zeros((n_params, 2), dtype=complex)

        for layer_index, (alpha, beta, gamma, delta) in enumerate(parameters):
            rx_value = rx(alpha)
            ry_value = ry(beta * float(x) + delta)
            rz_value = rz(gamma)
            drx = _rotation_derivative(rx_value, PAULIS[0])
            dry = _rotation_derivative(ry_value, PAULIS[1])
            drz = _rotation_derivative(rz_value, PAULIS[2])
            d2ry = _rotation_second_derivative(ry_value)

            layer = rz_value @ ry_value @ rx_value
            layer_x = beta * rz_value @ dry @ rx_value
            layer_params = (
                rz_value @ ry_value @ drx,
                float(x) * rz_value @ dry @ rx_value,
                drz @ ry_value @ rx_value,
                rz_value @ dry @ rx_value,
            )
            layer_x_params = (
                beta * rz_value @ dry @ drx,
                rz_value @ dry @ rx_value
                + float(x) * beta * rz_value @ d2ry @ rx_value,
                beta * drz @ dry @ rx_value,
                beta * rz_value @ d2ry @ rx_value,
            )

            previous_state = state
            previous_x = derivative_x
            previous_derivatives = derivatives
            previous_mixed = mixed_derivatives

            state = layer @ previous_state
            derivative_x = layer_x @ previous_state + layer @ previous_x
            derivatives = np.einsum("ab,pb->pa", layer, previous_derivatives)
            mixed_derivatives = np.einsum(
                "ab,pb->pa", layer_x, previous_derivatives
            ) + np.einsum("ab,pb->pa", layer, previous_mixed)

            offset = 4 * layer_index
            for local_index, (layer_param, layer_x_param) in enumerate(
                zip(layer_params, layer_x_params, strict=True)
            ):
                parameter_index = offset + local_index
                derivatives[parameter_index] += layer_param @ previous_state
                mixed_derivatives[parameter_index] += (
                    layer_x_param @ previous_state + layer_param @ previous_x
                )

        states[sample_index] = state
        state_dx[sample_index] = derivative_x
        state_dparams[sample_index] = derivatives
        state_dx_dparams[sample_index] = mixed_derivatives
        bloch[sample_index] = np.asarray(
            [np.vdot(state, pauli @ state).real for pauli in PAULIS], dtype=float
        )
        bloch_dx[sample_index] = np.asarray(
            [
                2.0 * float(np.real(np.vdot(derivative_x, pauli @ state)))
                for pauli in PAULIS
            ],
            dtype=float,
        )
        for parameter_index, derivative in enumerate(derivatives):
            bloch_dparams[sample_index, :, parameter_index] = [
                2.0 * float(np.real(np.vdot(derivative, pauli @ state)))
                for pauli in PAULIS
            ]

        overlap = np.vdot(state, derivative_x)
        vertical_velocity[sample_index] = -2.0 * float(np.imag(overlap))
        vertical_velocity_dparams[sample_index] = -2.0 * np.imag(
            np.einsum("pi,i->p", np.conj(derivatives), derivative_x)
            + np.einsum("i,pi->p", np.conj(state), mixed_derivatives)
        )

    return QRUDifferentials(
        states=states,
        state_dx=state_dx,
        state_dparams=state_dparams,
        state_dx_dparams=state_dx_dparams,
        bloch=bloch,
        bloch_dx=bloch_dx,
        bloch_dparams=bloch_dparams,
        vertical_velocity=vertical_velocity,
        vertical_velocity_dparams=vertical_velocity_dparams,
    )


def qru_bloch_and_jacobian(
    params: np.ndarray, xs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Bloch path and its exact parameter Jacobian.

    The Jacobian has shape ``(n_x, 3, depth, 4)`` and follows the parameter
    order ``[alpha, beta, gamma, delta]`` used by :func:`qru_unitary`.
    """
    parameters = np.asarray(params, dtype=float)
    grid = np.asarray(xs, dtype=float).reshape(-1)
    if parameters.ndim != 2 or parameters.shape[1] != 4:
        raise ValueError("params must have shape (depth, 4)")
    if grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("xs must contain finite values")
    if not np.all(np.isfinite(parameters)):
        raise ValueError("params must be finite")

    depth = len(parameters)
    bloch = np.empty((len(grid), 3), dtype=float)
    jacobian = np.empty((len(grid), 3, depth, 4), dtype=float)
    ket0 = np.array([1.0, 0.0], dtype=complex)

    for sample_index, x in enumerate(grid):
        layers: list[np.ndarray] = []
        layer_derivatives: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        for alpha, beta, gamma, delta in parameters:
            rx_value = rx(alpha)
            ry_value = ry(beta * float(x) + delta)
            rz_value = rz(gamma)
            layer = rz_value @ ry_value @ rx_value
            layers.append(layer)
            layer_derivatives.append(
                (
                    rz_value @ ry_value @ _rotation_derivative(rx_value, PAULIS[0]),
                    float(x)
                    * rz_value
                    @ _rotation_derivative(ry_value, PAULIS[1])
                    @ rx_value,
                    _rotation_derivative(rz_value, PAULIS[2]) @ ry_value @ rx_value,
                    rz_value @ _rotation_derivative(ry_value, PAULIS[1]) @ rx_value,
                )
            )

        prefixes = [np.eye(2, dtype=complex)]
        for layer in layers:
            prefixes.append(layer @ prefixes[-1])
        suffixes = [np.eye(2, dtype=complex) for _ in range(depth + 1)]
        for layer_index in range(depth - 1, -1, -1):
            suffixes[layer_index] = suffixes[layer_index + 1] @ layers[layer_index]

        state = prefixes[-1] @ ket0
        bloch[sample_index] = np.asarray(
            [np.vdot(state, pauli @ state).real for pauli in PAULIS], dtype=float
        )
        for layer_index, derivatives in enumerate(layer_derivatives):
            for parameter_index, derivative in enumerate(derivatives):
                d_unitary = (
                    suffixes[layer_index + 1] @ derivative @ prefixes[layer_index]
                )
                d_state = d_unitary @ ket0
                jacobian[sample_index, :, layer_index, parameter_index] = [
                    2.0 * float(np.real(np.vdot(d_state, pauli @ state)))
                    for pauli in PAULIS
                ]

    return bloch, jacobian


@dataclass(frozen=True)
class QRUTrajectory:
    """Exact QRU path on a strictly ordered input grid."""

    xs: np.ndarray
    unitaries: np.ndarray
    states: np.ndarray
    bloch: np.ndarray
    quaternions: np.ndarray


def evaluate_trajectory(params: np.ndarray, xs: np.ndarray) -> QRUTrajectory:
    grid = np.asarray(xs, dtype=float).reshape(-1)
    if grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("xs must contain at least two finite values")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("xs must be strictly increasing")

    unitaries = np.asarray([qru_unitary(params, float(x)) for x in grid])
    states = unitaries[:, :, 0]
    bloch = np.asarray([bloch_vector(state) for state in states])
    quaternions = np.asarray([su2_to_quaternion(unitary) for unitary in unitaries])
    return QRUTrajectory(grid, unitaries, states, bloch, quaternions)
