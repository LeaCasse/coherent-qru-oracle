from __future__ import annotations

import itertools

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qru_registerization.interface import (
    analytic_qae_phase_distribution,
    build_decoder,
    build_phase_taper_preparation,
    build_qae_interface,
    decoder_profile,
    phase_index_to_signed_code,
    phase_taper_amplitudes,
    phase_taper_work_qubits,
    qae_phase_distribution,
    signed_pauli_control_fidelity,
    validate_anf_interface,
)
from qru_registerization.qru import ry


@pytest.mark.parametrize("target_count", [1, 2, 3])
def test_signed_pauli_control_matches_direct_rotation(target_count: int) -> None:
    for sign in (0, 1):
        for word in itertools.product((0, 1), repeat=3):
            assert (
                signed_pauli_control_fidelity(
                    sign, word, 0.73, target_count=target_count
                )
                > 1.0 - 1e-12
            )


@pytest.mark.parametrize("taper", ["tophat", "sine"])
def test_analytic_qae_distribution_matches_statevector(taper: str) -> None:
    rng = np.random.default_rng(77)
    for phase_bits in (2, 3, 4):
        for probability in rng.uniform(0.0, 1.0, size=5):
            amplitude = ry(2.0 * np.arcsin(np.sqrt(probability)))
            actual = qae_phase_distribution(amplitude, phase_bits, taper=taper)
            expected = analytic_qae_phase_distribution(
                probability, phase_bits, taper=taper
            )
            assert max(abs(actual[i] - expected[i]) for i in actual) < 1e-10


@pytest.mark.parametrize("taper", ["tophat", "sine"])
def test_phase_taper_preparation_matches_declared_amplitudes(taper: str) -> None:
    for phase_bits in (2, 3, 4):
        expected = phase_taper_amplitudes(phase_bits, taper)
        actual = Statevector.from_instruction(
            build_phase_taper_preparation(phase_bits, taper)
        ).data
        assert np.isclose(np.linalg.norm(expected), 1.0, atol=1e-14)
        if phase_taper_work_qubits(taper) == 0:
            assert np.allclose(actual, expected, atol=1e-12)
        else:
            size = 2**phase_bits
            assert np.sum(np.abs(actual[size:]) ** 2) < 1e-24
            assert abs(np.vdot(expected, actual[:size])) > 1.0 - 1e-12
    with pytest.raises(ValueError):
        phase_taper_amplitudes(3, "unknown")


@pytest.mark.parametrize("precision", [(3, 2), (4, 3), (5, 4)])
def test_anf_decoder_matches_exact_map(precision: tuple[int, int]) -> None:
    phase_bits, magnitude_bits = precision
    decoder = build_decoder(phase_bits, magnitude_bits, method="anf")
    assert "unitary" not in [instruction.operation.name for instruction in decoder.data]
    for index in range(2**phase_bits):
        circuit = QuantumCircuit(decoder.num_qubits)
        for bit in range(phase_bits):
            if index & (1 << bit):
                circuit.x(bit)
        circuit.compose(decoder, inplace=True)
        observed = int(np.argmax(Statevector.from_instruction(circuit).probabilities()))
        sign, word, _ = phase_index_to_signed_code(index, phase_bits, magnitude_bits)
        expected = index | (sign << phase_bits)
        for offset, value in enumerate(word, start=1):
            expected |= int(value) << (phase_bits + offset)
        assert observed == expected


@pytest.mark.parametrize("target_count", [1, 2])
def test_qae_anf_interface_is_coherent_and_reversible(target_count: int) -> None:
    amplitude = ry(2.0 * np.arcsin(np.sqrt(0.30587876120028185)))
    metrics = validate_anf_interface(amplitude, 4, 3, 0.73, target_count=target_count)
    assert metrics["state_fidelity"] > 1.0 - 1e-10
    assert metrics["signed_zero_probability"] > 1.0 - 1e-10
    circuit = build_qae_interface(
        amplitude,
        4,
        3,
        0.73,
        method="anf",
        target_count=target_count,
    )
    names = [instruction.operation.name for instruction in circuit.data]
    assert "measure" not in names and "reset" not in names


def test_central_anf_profile() -> None:
    profile = decoder_profile(4, 3)
    assert profile["anf_terms"] == 24
    assert profile["lookup_word_toggles"] == 41
    assert profile["anf_max_degree"] == 4
