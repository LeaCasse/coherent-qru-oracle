"""Multi-input QRU oracle with affine binary data loading and expanded queries."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, StatePreparation, UCRZGate

from .encoding import normalize_axis
from .interface import (
    apply_anf_decoder,
    apply_signed_pauli_z_control,
    build_phase_taper_preparation,
    phase_taper_work_qubits,
)


def affine_input_oracle(params, input_bits, x0, step, axis, fibre0=0.0, fibre_step=0.0):
    """Load x=x0+step*sum_j 2^j b_j using one controlled Ry per input bit."""
    if input_bits < 1 or not np.all(np.isfinite([x0, step, fibre0, fibre_step])):
        raise ValueError("invalid affine input register")
    params = np.asarray(params, float)
    if (
        params.ndim != 2
        or params.shape[1] != 4
        or not len(params)
        or not np.all(np.isfinite(params))
    ):
        raise ValueError("params must be finite and have shape (depth>0,4)")
    circuit = QuantumCircuit(input_bits + 1, name="A_X_affine")
    system = input_bits
    circuit.rz(fibre0, system)
    for j in range(input_bits):
        if fibre_step:
            circuit.crz(fibre_step * 2**j, j, system)
    for alpha, beta, gamma, delta in params:
        circuit.rx(alpha, system)
        circuit.ry(beta * x0 + delta, system)
        for j in range(input_bits):
            circuit.cry(beta * step * 2**j, j, system)
        circuit.rz(gamma, system)
    vx, vy, vz = normalize_axis(axis)
    circuit.rz(-np.arctan2(vy, vx), system)
    circuit.ry(-np.arccos(np.clip(vz, -1, 1)), system)
    circuit.x(system)
    return circuit


def affine_estimator(
    params,
    input_bits,
    x0,
    step,
    axis,
    phase_bits=3,
    taper="tophat",
    fibre0=0.0,
    fibre_step=0.0,
    amplitudes=None,
):
    if phase_bits < 1:
        raise ValueError("phase_bits must be positive")
    a = affine_input_oracle(params, input_bits, x0, step, axis, fibre0, fibre_step)
    grover = QuantumCircuit(input_bits + 1, name="Q_X")
    grover.z(input_bits)
    grover.compose(a.inverse(), inplace=True)
    grover.x(input_bits)
    grover.z(input_bits)
    grover.x(input_bits)
    grover.compose(a, inplace=True)
    grover.global_phase += np.pi
    system = input_bits + phase_bits
    phase = list(range(input_bits, system))
    extra = phase_taper_work_qubits(taper) if amplitudes is None else 0
    e = QuantumCircuit(system + 1 + extra, name="E_X")
    e.compose(a, qubits=[*range(input_bits), system], inplace=True)
    if amplitudes is None:
        e.compose(
            build_phase_taper_preparation(phase_bits, taper),
            qubits=[*phase, *range(system + 1, system + 1 + extra)],
            inplace=True,
        )
    else:
        if len(amplitudes) != 2**phase_bits:
            raise ValueError("amplitude length must match phase_bits")
        e.append(StatePreparation(amplitudes), phase)
    controlled = grover.to_gate().control(1)
    for j in range(phase_bits):
        for _ in range(2**j):
            e.append(controlled, [phase[j], *range(input_bits), system])
    e.append(QFTGate(phase_bits).inverse(), phase)
    return e


def affine_interface(
    params,
    input_bits,
    x0,
    step,
    axis,
    phase_bits=3,
    magnitude_bits=2,
    gamma=0.8,
    taper="tophat",
    cleanup=True,
    fibre0=0.0,
    fibre_step=0.0,
    amplitudes=None,
):
    e = affine_estimator(
        params,
        input_bits,
        x0,
        step,
        axis,
        phase_bits,
        taper,
        fibre0,
        fibre_step,
        amplitudes,
    )
    sign = e.num_qubits
    magnitude = list(range(sign + 1, sign + 1 + magnitude_bits))
    target = magnitude[-1] + 1
    circuit = QuantumCircuit(target + 1)
    circuit.compose(e, qubits=range(e.num_qubits), inplace=True)
    phase = list(range(input_bits, input_bits + phase_bits))
    apply_anf_decoder(circuit, phase, sign, magnitude)
    apply_signed_pauli_z_control(circuit, sign, magnitude, [target], gamma)
    apply_anf_decoder(circuit, phase, sign, magnitude)
    if cleanup:
        circuit.compose(e.inverse(), qubits=range(e.num_qubits), inplace=True)
    return circuit, {
        "input": list(range(input_bits)),
        "target": target,
        "phase": phase,
        "system": input_bits + phase_bits,
        "signed": [sign, *magnitude],
        "work": list(range(input_bits, target)),
    }


def affine_direct_interface(
    params,
    input_bits,
    x0,
    step,
    axis,
    phase_bits=3,
    gamma=0.8,
    taper="tophat",
    cleanup=True,
    fibre0=0.0,
    fibre_step=0.0,
    amplitudes=None,
    control_scores=None,
):
    """E V E† with a phase-word multiplexor, without a signed register.

    Default scores are -cos(2*pi*y/2**m). Custom scores allow an exact
    contract-matched comparison with the signed decoder. Rz angles are
    classical constants, not a free implementation of an unknown function.
    The table and its generic synthesis grow as 2**m; all gates count.
    """
    size = 2**phase_bits
    scores = (
        -np.cos(2 * np.pi * np.arange(size) / size)
        if control_scores is None
        else np.asarray(control_scores, float)
    )
    if scores.shape != (size,) or not np.all(np.isfinite(scores)):
        raise ValueError("one finite control score per phase word required")
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite")
    e = affine_estimator(
        params,
        input_bits,
        x0,
        step,
        axis,
        phase_bits,
        taper,
        fibre0,
        fibre_step,
        amplitudes,
    )
    target = e.num_qubits
    circuit = QuantumCircuit(target + 1, name="direct_phase_control")
    phase = list(range(input_bits, input_bits + phase_bits))
    circuit.compose(e, qubits=range(e.num_qubits), inplace=True)
    circuit.append(UCRZGate((2 * gamma * scores).tolist()), [target, *phase])
    if cleanup:
        circuit.compose(e.inverse(), qubits=range(e.num_qubits), inplace=True)
    return circuit, {
        "input": list(range(input_bits)),
        "target": target,
        "phase": phase,
        "system": input_bits + phase_bits,
        "signed": [],
        "work": list(range(input_bits, target)),
    }
