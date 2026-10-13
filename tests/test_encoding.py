from __future__ import annotations

import numpy as np
import pytest

from qru_registerization.encoding import (
    basis_rotation,
    decode_signed,
    directional_observable,
    encode_signed,
    projector_probability,
    quantization_bound,
    quantize_signed,
    registerize_path,
)


def test_basis_rotation_implements_directional_observable() -> None:
    rng = np.random.default_rng(4)
    z = np.diag([1.0, -1.0]).astype(complex)
    for _ in range(20):
        axis = rng.normal(size=3)
        rotation = basis_rotation(axis)
        assert np.allclose(
            rotation.conj().T @ z @ rotation,
            directional_observable(axis),
            atol=1e-12,
        )


def test_projector_probability_matches_signed_coordinate() -> None:
    rng = np.random.default_rng(8)
    for _ in range(20):
        axis = rng.normal(size=3)
        state = rng.normal(size=2) + 1j * rng.normal(size=2)
        state /= np.linalg.norm(state)
        observable = directional_observable(axis)
        signed = float(np.vdot(state, observable @ state).real)
        assert np.isclose(projector_probability(state, axis), 0.5 * (1.0 + signed))


@pytest.mark.parametrize("bits", [2, 3, 5])
def test_signed_fixed_point_roundtrip_and_bound(bits: int) -> None:
    for value in np.linspace(-1.0, 1.0, 101):
        sign, word = encode_signed(float(value), bits)
        decoded = decode_signed(sign, word)
        assert decoded == quantize_signed(float(value), bits)
        assert abs(decoded - value) <= quantization_bound(bits) + 1e-12
    assert encode_signed(0.0, bits)[0] == 0


def test_registerize_path_uses_canonical_axis_orientation() -> None:
    states = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    bloch = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    first = registerize_path(states, bloch, np.array([0.0, 0.0, -1.0]), 3)
    second = registerize_path(states, bloch, np.array([0.0, 0.0, 1.0]), 3)
    assert np.allclose(first.axis, second.axis)
    assert np.allclose(first.signed, second.signed)
    assert np.allclose(first.quantized, second.quantized)
