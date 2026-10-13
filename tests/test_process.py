import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Choi, Kraus, Statevector, SuperOp, diamond_norm

from qru_registerization.channel import coherent_channel
from qru_registerization.process import (
    choi_from_superoperator,
    diamond_choi,
    process_channel,
    process_metrics,
    superoperator_from_choi,
)
from qru_registerization.qru import random_qru_params
from qru_registerization.structural import affine_direct_interface

PARAMS = random_qru_params(2, seed=600, scale=0.6)
XS = [-0.7, 0.9]
AXIS = [0.3, -0.4, np.sqrt(0.75)]


def structural_step(m=2, target_axis="z", gamma=0.8, fibre_step=0.0, params=None):
    n = 2**m
    scores = -np.cos(2 * np.pi * np.arange(n) / n) / np.cos(np.pi / n)
    circuit, layout = affine_direct_interface(
        PARAMS if params is None else params,
        1,
        XS[0],
        XS[1] - XS[0],
        AXIS,
        phase_bits=m,
        gamma=gamma,
        taper="sine",
        fibre_step=fibre_step,
        control_scores=scores,
    )
    if target_axis == "x":
        framed = QuantumCircuit(circuit.num_qubits)
        framed.h(layout["target"])
        framed.compose(circuit, inplace=True)
        framed.h(layout["target"])
        circuit = framed
    return circuit, layout


def circuit_kraus(circuit, layout):
    d = 4
    env_size = 2 ** len(layout["work"])
    kraus = np.zeros((env_size, d, d), complex)
    for x in range(2):
        for t in range(2):
            initial = np.zeros(2**circuit.num_qubits, complex)
            initial[x + (t << layout["target"])] = 1
            state = Statevector(initial).evolve(circuit).data
            for xo in range(2):
                for to in range(2):
                    indices = xo + (to << layout["target"]) + 2 * np.arange(env_size)
                    kraus[:, 2 * xo + to, 2 * x + t] = state[indices]
    return kraus


@pytest.mark.parametrize("mix_input", [False, True])
def test_general_shared_channel_matches_expanded_noncommuting_circuit(mix_input):
    first, layout = structural_step(fibre_step=1.3)
    second, _ = structural_step(target_axis="x", fibre_step=1.3)
    circuit = first.copy()
    gate = np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))
    circuit.x(layout["target"])
    if mix_input:
        circuit.h(0)
        gate = np.kron(
            np.array([[1, 1], [1, -1]]) / np.sqrt(2), np.array([[0, 1], [1, 0]])
        )
    circuit.compose(second, inplace=True)
    expected = Choi(Kraus(list(circuit_kraus(circuit, layout)))).data
    result = process_channel(
        PARAMS,
        XS,
        AXIS,
        [([0, 0, 1], 0.8), ([1, 0, 0], 0.8)],
        interventions=[gate],
        phase_bits=2,
        fibre=[0, 1.3],
    )
    np.testing.assert_allclose(result.choi, expected, atol=3e-11)


def test_fresh_noncommuting_channel_matches_independent_circuit_channels():
    first, layout = structural_step()
    second, _ = structural_step(target_axis="x")
    s1 = SuperOp(Kraus(list(circuit_kraus(first, layout)))).data
    s2 = SuperOp(Kraus(list(circuit_kraus(second, layout)))).data
    result = process_channel(
        PARAMS,
        XS,
        AXIS,
        [([0, 0, 1], 0.8), ([1, 0, 0], 0.8)],
        phase_bits=2,
        mode="fresh",
    )
    np.testing.assert_allclose(
        result.choi,
        Choi(SuperOp(s2 @ s1)).data,
        atol=3e-11,
    )
    np.testing.assert_allclose(
        superoperator_from_choi(result.choi), s2 @ s1, atol=3e-11
    )
    np.testing.assert_allclose(
        choi_from_superoperator(s2 @ s1), result.choi, atol=3e-11
    )


def test_single_use_agrees_with_previous_schur_channel():
    old = coherent_channel(
        PARAMS,
        XS,
        AXIS,
        phase_bits=3,
        taper="sine",
        control_scores=-np.cos(2 * np.pi * np.arange(8) / 8) / np.cos(np.pi / 8),
    )
    new = process_channel(PARAMS, XS, AXIS, [([0, 0, 1], 0.8)], phase_bits=3)
    idx = np.arange(4) * 5
    np.testing.assert_allclose(new.choi[np.ix_(idx, idx)], old.gram, atol=1e-12)


def test_echo_is_exact_with_shared_work_but_not_with_fresh_work():
    gate = np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))
    kwargs = {
        "pulses": [([0, 0, 1], 0.8)] * 2,
        "interventions": [gate],
        "phase_bits": 2,
    }
    shared = process_channel(PARAMS, XS, AXIS, **kwargs, fibre=[0, 2.1])
    fresh = process_channel(PARAMS, XS, AXIS, **kwargs, mode="fresh")
    assert process_metrics(shared)["diamond_upper"] < 1e-10
    assert process_metrics(shared)["work_leakage_worst"] < 1e-12
    assert process_metrics(fresh, diamond=False)["choi_fidelity"] < 0.95


def test_general_diamond_normalization_and_qiskit_crosscheck():
    d = 4
    identity = np.eye(d).ravel(order="F")
    j = np.diag(identity) - np.outer(identity, identity)
    estimate = diamond_choi(j)
    assert abs(estimate["diamond_lower"] - 2 * (1 - 1 / d)) < 1e-6
    result = process_channel(
        PARAMS, XS, AXIS, [([0, 0, 1], 0.8), ([1, 0, 0], 0.8)], phase_bits=2
    )
    vector = result.ideal.ravel(order="F")
    delta = result.choi - np.outer(vector, vector.conj())
    estimate = diamond_choi(delta)
    assert (
        abs(estimate["diamond_lower"] - diamond_norm(Choi(delta), solver="CLARABEL"))
        < 3e-6
    )
    assert estimate["sdp_gap"] < 3e-6


def test_input_intervention_invalidates_exact_target_echo():
    gate = np.kron(np.array([[1, 1], [1, -1]]) / np.sqrt(2), np.array([[0, 1], [1, 0]]))
    result = process_channel(
        PARAMS,
        XS,
        AXIS,
        [([0, 0, 1], 0.8)] * 2,
        interventions=[gate],
        phase_bits=2,
    )
    assert process_metrics(result, diamond=False)["choi_fidelity"] < 0.99


def test_fibre_sector_bound_counterexample_outside_input_preserving_contract():
    """Independent expanded circuits verify the seed-603 boundary example."""
    params = random_qru_params(2, seed=603, scale=0.6)
    gate = np.kron(
        np.array([[1, 1], [1, -1]]) / np.sqrt(2),
        np.array([[0, 1], [1, 0]]),
    )
    results = []
    for phi in [0.0, 2.1]:
        step, layout = structural_step(m=3, fibre_step=phi, params=params)
        circuit = QuantumCircuit(step.num_qubits)
        for j in range(4):
            circuit.compose(step, inplace=True)
            if j < 3:
                circuit.h(0)
                circuit.x(layout["target"])
        result = process_channel(
            params,
            XS,
            AXIS,
            [([0, 0, 1], 0.8)] * 4,
            interventions=[gate] * 3,
            phase_bits=3,
            fibre=[0, phi],
        )
        expected = Choi(Kraus(list(circuit_kraus(circuit, layout)))).data
        np.testing.assert_allclose(result.choi, expected, atol=5e-11)
        results.append(result)
    p1 = np.linalg.eigvalsh(sum(k.conj().T @ k for k in results[0].kraus[1::2]))[-1]
    # A maximally entangled input is already a physical diamond lower witness.
    physical_lower = (
        np.sum(np.linalg.svd(results[1].choi - results[0].choi, compute_uv=False)) / 4
    )
    assert physical_lower > 2 * p1 + 0.05
