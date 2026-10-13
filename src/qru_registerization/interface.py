from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity

from .encoding import basis_rotation, decode_signed, encode_signed


@dataclass(frozen=True)
class InterfaceLayout:
    """Qubit layout for QAE, signed workspace, and downstream targets."""

    phase_bits: int
    magnitude_bits: int
    target_count: int = 1

    def __post_init__(self) -> None:
        if min(self.phase_bits, self.magnitude_bits, self.target_count) <= 0:
            raise ValueError("all register sizes must be positive")

    @property
    def phase(self) -> tuple[int, ...]:
        return tuple(range(self.phase_bits))

    @property
    def system(self) -> int:
        return self.phase_bits

    @property
    def sign(self) -> int:
        return self.phase_bits + 1

    @property
    def magnitude(self) -> tuple[int, ...]:
        start = self.phase_bits + 2
        return tuple(range(start, start + self.magnitude_bits))

    @property
    def targets(self) -> tuple[int, ...]:
        start = self.phase_bits + self.magnitude_bits + 2
        return tuple(range(start, start + self.target_count))

    @property
    def size(self) -> int:
        return self.phase_bits + self.magnitude_bits + self.target_count + 2


def _validate_unitary(matrix: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=complex)
    if value.shape != (2, 2):
        raise ValueError(f"{name} must be 2x2")
    if not np.allclose(value.conj().T @ value, np.eye(2), atol=1e-10):
        raise ValueError(f"{name} must be unitary")
    return value


def _lsb_bits(index: int, width: int) -> tuple[int, ...]:
    return tuple((index >> bit) & 1 for bit in range(width))


def directional_amplitude_unitary(
    qru_matrix: np.ndarray, axis: np.ndarray
) -> np.ndarray:
    r"""Return ``A_v`` with ``|<1|A_v|0>|^2=(1+v.r)/2``."""
    qru_matrix = _validate_unitary(qru_matrix, name="qru_matrix")
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    return x_gate @ basis_rotation(axis) @ qru_matrix


def amplitude_grover_operator(amplitude_unitary: np.ndarray) -> np.ndarray:
    r"""Return ``Q=-A S_0 A^dagger S_good`` for one-qubit QAE."""
    amplitude_unitary = _validate_unitary(amplitude_unitary, name="amplitude_unitary")
    s_zero = np.diag([-1.0, 1.0]).astype(complex)
    s_good = np.diag([1.0, -1.0]).astype(complex)
    return -amplitude_unitary @ s_zero @ amplitude_unitary.conj().T @ s_good


def phase_taper_amplitudes(phase_bits: int, taper: str = "tophat") -> np.ndarray:
    """Return normalized ancilla amplitudes for coherent phase estimation.

    ``tophat`` is the standard uniform QPE state. ``sine`` uses
    ``sin(pi*n/N)`` and suppresses the long Fourier tails while retaining a
    fully coherent phase register.
    """
    if phase_bits <= 0:
        raise ValueError("phase_bits must be positive")
    size = 2**phase_bits
    if taper == "tophat":
        amplitudes = np.ones(size, dtype=float)
    elif taper == "sine":
        amplitudes = np.sin(np.pi * np.arange(size, dtype=float) / size)
    else:
        raise ValueError("taper must be 'tophat' or 'sine'")
    norm = float(np.linalg.norm(amplitudes))
    if norm == 0.0:
        raise RuntimeError("phase taper has zero norm")
    return amplitudes / norm


def phase_taper_work_qubits(taper: str = "tophat") -> int:
    """Return clean work qubits required by the structured taper preparation."""
    if taper == "tophat":
        return 0
    if taper == "sine":
        return 1
    raise ValueError("taper must be 'tophat' or 'sine'")


def build_phase_taper_preparation(
    phase_bits: int,
    taper: str = "tophat",
) -> QuantumCircuit:
    """Build an exact structured phase-taper preparation.

    The sine state is prepared on ``phase_bits + 1`` qubits. A QFT maps
    ``(|1> - |2N-1>)/sqrt(2)`` to ``i |-> |sine_N>``; the most-significant
    work qubit is then returned exactly to ``|0>``.
    """
    from qiskit.circuit.library import QFTGate

    phase_taper_amplitudes(phase_bits, taper)
    work_qubits = phase_taper_work_qubits(taper)
    circuit = QuantumCircuit(
        phase_bits + work_qubits, name=f"{taper}_taper_m{phase_bits}"
    )
    if taper == "tophat":
        circuit.h(range(phase_bits))
        return circuit

    work = phase_bits
    circuit.x(0)
    circuit.h(1)
    circuit.z(1)
    for qubit in range(2, phase_bits + 1):
        circuit.cx(1, qubit)
    circuit.append(QFTGate(phase_bits + 1), range(phase_bits + 1))
    circuit.h(work)
    circuit.x(work)
    return circuit


def build_coherent_qae(
    amplitude_unitary: np.ndarray,
    phase_bits: int,
    *,
    taper: str = "tophat",
) -> QuantumCircuit:
    """Build measurement-free QPE-based amplitude estimation."""
    from qiskit.circuit.library import QFTGate, UnitaryGate

    if phase_bits <= 0:
        raise ValueError("phase_bits must be positive")
    amplitude_unitary = _validate_unitary(amplitude_unitary, name="amplitude_unitary")
    grover = amplitude_grover_operator(amplitude_unitary)
    work_qubits = phase_taper_work_qubits(taper)
    circuit = QuantumCircuit(
        phase_bits + 1 + work_qubits, name=f"coherent_qae_{taper}_m{phase_bits}"
    )
    system = phase_bits
    circuit.append(UnitaryGate(amplitude_unitary, label="A_v"), [system])
    taper_qubits = list(range(phase_bits))
    if work_qubits:
        taper_qubits.append(phase_bits + 1)
    circuit.compose(
        build_phase_taper_preparation(phase_bits, taper),
        qubits=taper_qubits,
        inplace=True,
    )
    for bit in range(phase_bits):
        power = np.linalg.matrix_power(grover, 2**bit)
        circuit.append(
            UnitaryGate(power, label=f"Q^{2**bit}").control(1),
            [bit, system],
        )
    circuit.append(QFTGate(phase_bits).inverse(), list(range(phase_bits)))
    return circuit


def qae_phase_distribution(
    amplitude_unitary: np.ndarray,
    phase_bits: int,
    *,
    taper: str = "tophat",
) -> dict[int, float]:
    """Return the exact phase-register distribution of coherent QAE."""
    state = Statevector.from_instruction(
        build_coherent_qae(amplitude_unitary, phase_bits, taper=taper)
    )
    raw = state.probabilities_dict(qargs=list(range(phase_bits)))
    values = {
        int(str(bits), 2): float(probability) for bits, probability in raw.items()
    }
    return {index: values.get(index, 0.0) for index in range(2**phase_bits)}


def analytic_qae_phase_distribution(
    probability: float,
    phase_bits: int,
    *,
    taper: str = "tophat",
) -> dict[int, float]:
    """Return the tapered finite-QPE QAE distribution for ``p=sin^2(theta)``."""
    if phase_bits <= 0:
        raise ValueError("phase_bits must be positive")
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    size = 2**phase_bits
    phase = float(np.arcsin(np.sqrt(probability)) / np.pi)
    amplitudes = phase_taper_amplitudes(phase_bits, taper)
    times = np.arange(size, dtype=float)

    def kernel(eigenphase: float) -> np.ndarray:
        modulated = amplitudes * np.exp(2.0j * np.pi * eigenphase * times)
        fourier = np.fft.fft(modulated) / np.sqrt(size)
        return np.abs(fourier) ** 2

    probabilities = 0.5 * (kernel(phase) + kernel((1.0 - phase) % 1.0))
    probabilities /= np.sum(probabilities)
    return {index: float(probabilities[index]) for index in range(size)}


def phase_index_to_signed_code(
    index: int,
    phase_bits: int,
    magnitude_bits: int,
) -> tuple[int, np.ndarray, float]:
    r"""Map phase bin ``y`` to signed fixed point for ``-cos(2 pi y/2^m)``."""
    if min(phase_bits, magnitude_bits) <= 0:
        raise ValueError("register sizes must be positive")
    if not 0 <= index < 2**phase_bits:
        raise ValueError("phase index outside the register range")
    signed_value = float(-np.cos(2.0 * np.pi * index / (2**phase_bits)))
    sign, word = encode_signed(signed_value, magnitude_bits)
    return int(sign), np.asarray(word, dtype=int), decode_signed(sign, word)


def _validate_registers(
    phase_qubits: Sequence[int],
    sign_qubit: int,
    magnitude_qubits: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    phase = tuple(int(qubit) for qubit in phase_qubits)
    magnitude = tuple(int(qubit) for qubit in magnitude_qubits)
    qubits = (*phase, int(sign_qubit), *magnitude)
    if not phase or not magnitude:
        raise ValueError("phase and magnitude registers must be non-empty")
    if min(qubits) < 0 or len(set(qubits)) != len(qubits):
        raise ValueError("register qubits must be distinct and non-negative")
    return phase, magnitude


def _apply_basis_controlled_x(
    circuit: QuantumCircuit,
    controls: Sequence[int],
    word: Sequence[int],
    target: int,
) -> None:
    controls = tuple(int(qubit) for qubit in controls)
    word = tuple(int(bit) for bit in word)
    if len(controls) != len(word) or not controls:
        raise ValueError("control word must match a non-empty control register")
    open_controls = [
        qubit for qubit, bit in zip(controls, word, strict=True) if bit == 0
    ]
    for qubit in open_controls:
        circuit.x(qubit)
    circuit.mcx(list(controls), int(target))
    for qubit in reversed(open_controls):
        circuit.x(qubit)


def apply_lookup_decoder(
    circuit: QuantumCircuit,
    phase_qubits: Sequence[int],
    sign_qubit: int,
    magnitude_qubits: Sequence[int],
) -> None:
    """Apply the exact wordwise decoder baseline."""
    phase, magnitude = _validate_registers(phase_qubits, sign_qubit, magnitude_qubits)
    outputs = (int(sign_qubit), *magnitude)
    for index in range(2 ** len(phase)):
        sign, word, _ = phase_index_to_signed_code(index, len(phase), len(magnitude))
        for target, bit in zip(outputs, (sign, *word), strict=True):
            if bit:
                _apply_basis_controlled_x(
                    circuit, phase, _lsb_bits(index, len(phase)), target
                )


def algebraic_normal_form(values: Sequence[int]) -> np.ndarray:
    """Return positive-polarity Reed--Muller coefficients over GF(2)."""
    coefficients = np.asarray(tuple(values), dtype=np.uint8)
    if (
        coefficients.ndim != 1
        or coefficients.size == 0
        or coefficients.size & (coefficients.size - 1)
        or not np.all(np.isin(coefficients, (0, 1)))
    ):
        raise ValueError("values must be a non-empty binary power-of-two table")
    coefficients = coefficients.copy()
    for bit in range(int(np.log2(coefficients.size))):
        stride = 1 << bit
        for mask in range(coefficients.size):
            if mask & stride:
                coefficients[mask] ^= coefficients[mask ^ stride]
    return coefficients.astype(int)


def decoder_anf_terms(
    phase_bits: int,
    magnitude_bits: int,
) -> tuple[tuple[int, ...], ...]:
    """Return exact ANF monomial masks for sign and magnitude outputs."""
    if min(phase_bits, magnitude_bits) <= 0:
        raise ValueError("register sizes must be positive")
    tables = [[] for _ in range(1 + magnitude_bits)]
    for index in range(2**phase_bits):
        sign, word, _ = phase_index_to_signed_code(index, phase_bits, magnitude_bits)
        for table, bit in zip(tables, (sign, *word), strict=True):
            table.append(int(bit))
    return tuple(
        tuple(
            mask
            for mask, coefficient in enumerate(algebraic_normal_form(table))
            if coefficient
        )
        for table in tables
    )


def decoder_profile(phase_bits: int, magnitude_bits: int) -> dict[str, int | float]:
    """Return finite-decoder structural statistics."""
    terms = decoder_anf_terms(phase_bits, magnitude_bits)
    masks = [mask for output in terms for mask in output]
    lookup_toggles = 0
    for index in range(2**phase_bits):
        sign, word, _ = phase_index_to_signed_code(index, phase_bits, magnitude_bits)
        lookup_toggles += sign + int(np.sum(word))
    return {
        "phase_bits": phase_bits,
        "magnitude_bits": magnitude_bits,
        "output_bits": 1 + magnitude_bits,
        "anf_terms": len(masks),
        "anf_nonconstant_terms": sum(mask != 0 for mask in masks),
        "anf_max_degree": max((mask.bit_count() for mask in masks), default=0),
        "lookup_word_toggles": lookup_toggles,
        "term_reduction_factor": float(lookup_toggles / max(len(masks), 1)),
    }


def apply_anf_decoder(
    circuit: QuantumCircuit,
    phase_qubits: Sequence[int],
    sign_qubit: int,
    magnitude_qubits: Sequence[int],
) -> None:
    r"""Apply ``|y>|w> -> |y>|w xor f(y)>`` using X/CNOT/MCX gates."""
    phase, magnitude = _validate_registers(phase_qubits, sign_qubit, magnitude_qubits)
    outputs = (int(sign_qubit), *magnitude)
    for target, terms in zip(
        outputs, decoder_anf_terms(len(phase), len(magnitude)), strict=True
    ):
        for mask in terms:
            controls = [phase[bit] for bit in range(len(phase)) if mask & (1 << bit)]
            if not controls:
                circuit.x(target)
            elif len(controls) == 1:
                circuit.cx(controls[0], target)
            else:
                circuit.mcx(controls, target)


def build_decoder(
    phase_bits: int,
    magnitude_bits: int,
    *,
    method: str,
) -> QuantumCircuit:
    """Build either the ``lookup`` baseline or exact ``anf`` decoder."""
    if min(phase_bits, magnitude_bits) <= 0:
        raise ValueError("register sizes must be positive")
    circuit = QuantumCircuit(
        phase_bits + magnitude_bits + 1,
        name=f"phase_to_signed_{method}",
    )
    phase = tuple(range(phase_bits))
    sign = phase_bits
    magnitude = tuple(range(phase_bits + 1, phase_bits + magnitude_bits + 1))
    decoder = {"lookup": apply_lookup_decoder, "anf": apply_anf_decoder}.get(method)
    if decoder is None:
        raise ValueError("method must be 'lookup' or 'anf'")
    decoder(circuit, phase, sign, magnitude)
    return circuit


def apply_signed_pauli_z_control(
    circuit: QuantumCircuit,
    sign_qubit: int,
    magnitude_qubits: Sequence[int],
    target_qubits: Sequence[int],
    gamma: float,
) -> None:
    r"""Apply ``exp(-i gamma s Z_1...Z_k)`` from signed magnitude ``s``."""
    magnitude = tuple(int(qubit) for qubit in magnitude_qubits)
    targets = tuple(int(qubit) for qubit in target_qubits)
    qubits = (int(sign_qubit), *magnitude, *targets)
    if not magnitude or not targets:
        raise ValueError("magnitude and target registers must be non-empty")
    if min(qubits) < 0 or len(set(qubits)) != len(qubits):
        raise ValueError("register and target qubits must be distinct")
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite")

    pivot = targets[-1]
    for control in targets[:-1]:
        circuit.cx(control, pivot)
    for index, magnitude_qubit in enumerate(magnitude, start=1):
        angle = 2.0 * float(gamma) * 2.0 ** (-index)
        circuit.x(sign_qubit)
        circuit.mcrz(angle, [sign_qubit, magnitude_qubit], pivot)
        circuit.x(sign_qubit)
        circuit.mcrz(-angle, [sign_qubit, magnitude_qubit], pivot)
    for control in reversed(targets[:-1]):
        circuit.cx(control, pivot)


def build_qae_interface(
    amplitude_unitary: np.ndarray,
    phase_bits: int,
    magnitude_bits: int,
    gamma: float,
    *,
    method: str,
    target_count: int = 1,
    uncompute_signed: bool = True,
) -> QuantumCircuit:
    """Build QAE, finite decoding, signed Pauli control, and workspace cleanup."""
    layout = InterfaceLayout(phase_bits, magnitude_bits, target_count)
    circuit = QuantumCircuit(layout.size, name=f"qae_{method}_interface")
    circuit.compose(
        build_coherent_qae(amplitude_unitary, phase_bits),
        qubits=[*layout.phase, layout.system],
        inplace=True,
    )
    if method == "lookup":
        decoder: Callable[..., None] = apply_lookup_decoder
    elif method == "anf":
        decoder = apply_anf_decoder
    else:
        raise ValueError("method must be 'lookup' or 'anf'")
    decoder(circuit, layout.phase, layout.sign, layout.magnitude)
    for target in layout.targets:
        circuit.h(target)
    apply_signed_pauli_z_control(
        circuit,
        layout.sign,
        layout.magnitude,
        layout.targets,
        gamma,
    )
    if uncompute_signed:
        decoder(circuit, layout.phase, layout.sign, layout.magnitude)
    return circuit


def _apply_phase_word_reference(
    circuit: QuantumCircuit,
    layout: InterfaceLayout,
    gamma: float,
) -> None:
    """Apply the directly phase-word-controlled Pauli evolution used for validation."""
    from qiskit.circuit.library import RZGate

    pivot = layout.targets[-1]
    for control in layout.targets[:-1]:
        circuit.cx(control, pivot)
    for index in range(2**layout.phase_bits):
        _, _, signed_value = phase_index_to_signed_code(
            index, layout.phase_bits, layout.magnitude_bits
        )
        if abs(signed_value) <= 1e-15:
            continue
        word = _lsb_bits(index, layout.phase_bits)
        open_controls = [
            qubit for qubit, bit in zip(layout.phase, word, strict=True) if bit == 0
        ]
        for qubit in open_controls:
            circuit.x(qubit)
        circuit.append(
            RZGate(2.0 * float(gamma) * signed_value).control(layout.phase_bits),
            [*layout.phase, pivot],
        )
        for qubit in reversed(open_controls):
            circuit.x(qubit)
    for control in reversed(layout.targets[:-1]):
        circuit.cx(control, pivot)


def validate_anf_interface(
    amplitude_unitary: np.ndarray,
    phase_bits: int,
    magnitude_bits: int,
    gamma: float,
    *,
    target_count: int = 1,
) -> dict[str, float]:
    """Compare the ANF interface with direct phase-word-controlled evolution."""
    layout = InterfaceLayout(phase_bits, magnitude_bits, target_count)
    actual = Statevector.from_instruction(
        build_qae_interface(
            amplitude_unitary,
            phase_bits,
            magnitude_bits,
            gamma,
            method="anf",
            target_count=target_count,
        )
    )
    reference_circuit = QuantumCircuit(layout.size, name="phase_word_reference")
    reference_circuit.compose(
        build_coherent_qae(amplitude_unitary, phase_bits),
        qubits=[*layout.phase, layout.system],
        inplace=True,
    )
    for target in layout.targets:
        reference_circuit.h(target)
    _apply_phase_word_reference(reference_circuit, layout, gamma)
    reference = Statevector.from_instruction(reference_circuit)

    signed = [layout.sign, *layout.magnitude]
    reduced = partial_trace(
        actual,
        [qubit for qubit in range(actual.num_qubits) if qubit not in signed],
    )
    return {
        "state_fidelity": float(state_fidelity(reference, actual)),
        "signed_zero_probability": float(actual.probabilities(qargs=signed)[0]),
        "signed_register_purity": float(np.real(np.trace(reduced.data @ reduced.data))),
    }


def signed_pauli_control_fidelity(
    sign: int,
    word: Sequence[int],
    gamma: float,
    *,
    target_count: int,
) -> float:
    """Validate signed-register control against the corresponding direct rotation."""
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    word = np.asarray(tuple(word), dtype=int)
    if (
        sign not in (0, 1)
        or word.ndim != 1
        or word.size == 0
        or not np.all(np.isin(word, (0, 1)))
    ):
        raise ValueError("invalid signed code")
    magnitude = tuple(range(1, 1 + word.size))
    targets = tuple(range(1 + word.size, 1 + word.size + target_count))
    actual = QuantumCircuit(1 + word.size + target_count)
    reference = QuantumCircuit(1 + word.size + target_count)
    for circuit in (actual, reference):
        if sign:
            circuit.x(0)
        for qubit, bit in zip(magnitude, word, strict=True):
            if bit:
                circuit.x(qubit)
        for target in targets:
            circuit.h(target)
    apply_signed_pauli_z_control(actual, 0, magnitude, targets, gamma)
    pivot = targets[-1]
    for control in targets[:-1]:
        reference.cx(control, pivot)
    reference.rz(2.0 * gamma * decode_signed(sign, word), pivot)
    for control in reversed(targets[:-1]):
        reference.cx(control, pivot)
    return float(
        state_fidelity(
            Statevector.from_instruction(reference),
            Statevector.from_instruction(actual),
        )
    )
