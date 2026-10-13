from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qru import PAULIS, ry, rz, signed_coordinate


def normalize_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float).reshape(3)
    if not np.all(np.isfinite(axis)):
        raise ValueError("axis must be finite")
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise ValueError("axis must be non-zero")
    return axis / norm


def directional_observable(axis: np.ndarray) -> np.ndarray:
    axis = normalize_axis(axis)
    return np.einsum("i,ijk->jk", axis, PAULIS)


def directional_projector(axis: np.ndarray) -> np.ndarray:
    return 0.5 * (np.eye(2, dtype=complex) + directional_observable(axis))


def basis_rotation(axis: np.ndarray) -> np.ndarray:
    r"""Return ``B`` such that ``B^dagger Z B = axis.sigma``."""
    vx, vy, vz = normalize_axis(axis)
    theta = float(np.arccos(np.clip(vz, -1.0, 1.0)))
    phi = float(np.arctan2(vy, vx))
    return ry(-theta) @ rz(-phi)


def projector_probability(state: np.ndarray, axis: np.ndarray) -> float:
    state = np.asarray(state, dtype=complex).reshape(2)
    norm = float(np.linalg.norm(state))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("state must be finite and non-zero")
    state = state / norm
    probability = np.vdot(state, directional_projector(axis) @ state).real
    return float(np.clip(probability, 0.0, 1.0))


def probability_to_signed(probability: float) -> float:
    if (
        not np.isfinite(probability)
        or probability < -1e-10
        or probability > 1.0 + 1e-10
    ):
        raise ValueError("probability must lie in [0, 1]")
    return float(2.0 * np.clip(probability, 0.0, 1.0) - 1.0)


def _validate_precision(bits: int) -> None:
    if not isinstance(bits, (int, np.integer)) or bits <= 0:
        raise ValueError("bits must be a positive integer")


def encode_signed(value: float, bits: int) -> tuple[int, np.ndarray]:
    """Truncated signed-magnitude fixed point with canonical zero."""
    _validate_precision(bits)
    if not np.isfinite(value):
        raise ValueError("value must be finite")
    clipped = float(np.clip(value, -1.0, 1.0))
    magnitude = min(abs(clipped), 1.0 - 2.0 ** (-bits))
    integer = int(np.floor(magnitude * (2**bits) + 1e-12))
    word = np.array(
        [(integer >> (bits - 1 - index)) & 1 for index in range(bits)],
        dtype=int,
    )
    sign = int(clipped < 0.0 and integer != 0)
    return sign, word


def decode_signed(sign: int, word: np.ndarray) -> float:
    if sign not in (0, 1):
        raise ValueError("sign must be binary")
    word = np.asarray(word)
    if word.ndim != 1 or word.size == 0 or not np.all(np.isin(word, (0, 1))):
        raise ValueError("word must be a non-empty binary vector")
    integer = 0
    for bit in word.astype(int, copy=False):
        integer = (integer << 1) | int(bit)
    if integer == 0:
        return 0.0
    magnitude = integer / (2**word.size)
    return float((-1.0 if sign else 1.0) * magnitude)


def quantize_signed(value: float, bits: int) -> float:
    return decode_signed(*encode_signed(value, bits))


def quantization_bound(bits: int) -> float:
    _validate_precision(bits)
    return 2.0 ** (-bits)


@dataclass(frozen=True)
class RegisterizedPath:
    axis: np.ndarray
    signed: np.ndarray
    probabilities: np.ndarray
    quantized: np.ndarray


def registerize_path(
    states: np.ndarray,
    bloch_path: np.ndarray,
    axis: np.ndarray,
    bits: int,
) -> RegisterizedPath:
    """Map a signed Bloch coordinate to projector probabilities and fixed point."""
    from .readout_geometry import canonical_axis

    oriented = canonical_axis(axis)
    signed = np.asarray([signed_coordinate(point, oriented) for point in bloch_path])
    probabilities = np.asarray(
        [projector_probability(state, oriented) for state in states], dtype=float
    )
    quantized = np.asarray(
        [quantize_signed(probability_to_signed(p), bits) for p in probabilities],
        dtype=float,
    )
    return RegisterizedPath(oriented, signed, probabilities, quantized)
