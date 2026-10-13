"""Structural baselines; no lookup of the unknown QRU scores."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UCRZGate

from .structural import affine_direct_interface, affine_estimator, affine_input_oracle


def score_block_encoding(params, input_bits, x0, step, axis, fibre_step=0.0):
    """Hermitian unitary B=-A† Z A with <0_sys|B|0_sys>=diag(s_x).

    Uses two structural A/A† calls. This is a block encoding, not a completed
    Hamiltonian-simulation/QSP circuit.
    """
    a = affine_input_oracle(params, input_bits, x0, step, axis, fibre_step=fibre_step)
    b = QuantumCircuit(a.num_qubits, name="score_block")
    b.compose(a, inplace=True)
    b.z(input_bits)
    b.global_phase += np.pi
    b.compose(a.inverse(), inplace=True)
    return b


def alternating_protocol(params, x0, step, axis, phase_bits=2, k=2, fused=False):
    """Alternating Z,X controls with shared work, optionally cancelling E E†.

    Exact same input-target-work unitary. The optimization is elementary
    cancellation, not a new synthesis method. Only an input-preserving
    protocol with the same estimator can use this construction.
    """
    n = 2**phase_bits
    scores = -np.cos(2 * np.pi * np.arange(n) / n) / np.cos(np.pi / n)
    e = affine_estimator(params, 1, x0, step, axis, phase_bits=phase_bits, taper="sine")
    target = e.num_qubits
    circuit = QuantumCircuit(target + 1)
    phase = list(range(1, 1 + phase_bits))
    if fused:
        circuit.compose(e, qubits=range(target), inplace=True)
    for j in range(k):
        if j % 2:
            circuit.h(target)
        if fused:
            circuit.append(UCRZGate((1.6 * scores).tolist()), [target, *phase])
        else:
            step_circuit, _ = affine_direct_interface(
                params,
                1,
                x0,
                step,
                axis,
                phase_bits=phase_bits,
                gamma=0.8,
                taper="sine",
                control_scores=scores,
            )
            circuit.compose(step_circuit, inplace=True)
        if j % 2:
            circuit.h(target)
    if fused:
        circuit.compose(e.inverse(), qubits=range(target), inplace=True)
    return circuit, {
        "target": target,
        "input": [0],
        "work": list(range(1, target)),
    }
