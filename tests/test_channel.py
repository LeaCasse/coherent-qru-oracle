import numpy as np
import pytest
from qiskit.quantum_info import Choi, Operator, Statevector, diamond_norm

from qru_registerization.channel import channel_metrics, coherent_channel, schur_diamond
from qru_registerization.interface import directional_amplitude_unitary
from qru_registerization.qru import qru_unitary, random_qru_params
from qru_registerization.structural import affine_input_oracle, affine_interface

PARAMS = random_qru_params(2, seed=42, scale=0.6)
AXIS = np.array([0.3, -0.4, np.sqrt(0.75)])
XS = np.linspace(-0.73, 0.91, 4)


@pytest.mark.parametrize("taper", ["tophat", "sine"])
@pytest.mark.parametrize("cleanup", [False, True])
def test_full_channel_matches_expanded_four_branch_circuit(taper, cleanup):
    circuit, layout = affine_interface(
        PARAMS,
        2,
        XS[0],
        XS[1] - XS[0],
        AXIS,
        phase_bits=2,
        magnitude_bits=2,
        taper=taper,
        cleanup=cleanup,
        fibre_step=0.7,
    )
    env = []
    for branch in range(4):
        for z in range(2):
            initial = np.zeros(2**circuit.num_qubits, complex)
            initial[branch + (z << layout["target"])] = 1
            state = Statevector(initial).evolve(circuit).data
            indices = (
                branch
                + (z << layout["target"])
                + 4 * np.arange(2 ** len(layout["work"]))
            )
            env.append(state[indices])
    env = np.asarray(env)
    actual = coherent_channel(
        PARAMS,
        XS,
        AXIS,
        phase_bits=2,
        magnitude_bits=2,
        taper=taper,
        cleanup=cleanup,
        fibre=0.7 * np.arange(4),
    )
    np.testing.assert_allclose(actual.gram, env @ env.conj().T, atol=2e-11)
    np.testing.assert_allclose(actual.zero_probability, abs(env[:, 0]) ** 2, atol=2e-11)


def test_affine_loading_matches_four_known_branches():
    matrix = Operator(affine_input_oracle(PARAMS, 2, XS[0], XS[1] - XS[0], AXIS)).data
    for branch, x in enumerate(XS):
        idx = [branch, branch + 4]
        np.testing.assert_allclose(
            matrix[np.ix_(idx, idx)],
            directional_amplitude_unitary(qru_unitary(PARAMS, x), AXIS),
            atol=1e-12,
        )


def test_zero_control_is_identity_and_clean_even_with_fibre():
    result = coherent_channel(PARAMS, XS, AXIS, gamma=0, fibre=np.arange(4) * 2)
    np.testing.assert_allclose(result.gram, np.ones((8, 8)), atol=1e-12)
    np.testing.assert_allclose(result.zero_probability, 1, atol=1e-12)


def test_dephasing_is_not_reversed_by_estimator_inverse():
    result = coherent_channel(PARAMS, XS, AXIS, gamma=0, dephase=1)
    assert channel_metrics(result, diamond=False)["choi_fidelity"] < 0.999
    assert min(result.zero_probability) < 0.9
    np.testing.assert_allclose(np.diag(result.gram), 1, atol=1e-12)
    assert np.linalg.eigvalsh(result.gram)[0] > -1e-12


def test_diamond_normalization_and_general_watrous_crosscheck():
    # Identity vs full dephasing on d dimensions has diamond norm 2*(1-1/d).
    for d in [2, 4, 8]:
        estimate = schur_diamond(np.eye(d) - np.ones((d, d)))
        assert abs(estimate["diamond_lower"] - 2 * (1 - 1 / d)) < 1e-6
        assert estimate["sdp_gap"] < 1e-6
    result = coherent_channel(PARAMS, XS[:2], AXIS, phase_bits=2)
    delta = result.gram - np.outer(result.ideal, result.ideal.conj())
    d = len(delta)
    choi = np.zeros((d * d, d * d), complex)
    indices = np.arange(d) * (d + 1)
    choi[np.ix_(indices, indices)] = delta
    full = diamond_norm(Choi(choi), solver="CLARABEL")
    assert abs(schur_diamond(delta)["diamond_lower"] - full) < 2e-6


def test_fibre_sensitivity_bound_by_system_one_leakage():
    for m in [2, 3, 4]:
        baseline = coherent_channel(PARAMS, XS, AXIS, phase_bits=m)
        changed = coherent_channel(
            PARAMS, XS, AXIS, phase_bits=m, fibre=np.arange(4) * 1.3
        )
        distance = schur_diamond(changed.gram - baseline.gram)["diamond_upper"]
        assert distance <= 2 * max(baseline.system_one_probability) + 2e-7
        np.testing.assert_allclose(
            changed.zero_probability, baseline.zero_probability, atol=1e-12
        )


def test_custom_preparation_has_same_retained_channel_and_work_reset():
    amplitudes = np.array([0.1, 0.4, 0.7, 0.2])
    amplitudes /= np.linalg.norm(amplitudes)
    circuit, layout = affine_interface(
        PARAMS,
        2,
        XS[0],
        XS[1] - XS[0],
        AXIS,
        phase_bits=2,
        magnitude_bits=2,
        amplitudes=amplitudes,
    )
    env = []
    for branch in range(4):
        for z in range(2):
            initial = np.zeros(2**circuit.num_qubits, complex)
            initial[branch + (z << layout["target"])] = 1
            state = Statevector(initial).evolve(circuit).data
            idx = (
                branch
                + (z << layout["target"])
                + 4 * np.arange(2 ** len(layout["work"]))
            )
            env.append(state[idx])
    env = np.array(env)
    result = coherent_channel(
        PARAMS, XS, AXIS, phase_bits=2, magnitude_bits=2, amplitudes=amplitudes
    )
    np.testing.assert_allclose(result.gram, env @ env.conj().T, atol=1e-11)
    np.testing.assert_allclose(result.zero_probability, abs(env[:, 0]) ** 2, atol=1e-11)


def test_dephasing_without_inverse_changes_no_retained_channel():
    clean = coherent_channel(PARAMS, XS, AXIS, cleanup=False)
    marked = coherent_channel(PARAMS, XS, AXIS, cleanup=False, dephase=1)
    np.testing.assert_allclose(clean.gram, marked.gram, atol=1e-12)


def test_inverse_changes_cross_branch_coherence_but_not_fixed_input_target():
    before = coherent_channel(PARAMS, XS, AXIS, cleanup=False)
    after = coherent_channel(PARAMS, XS, AXIS, cleanup=True)
    for x in range(4):
        idx = [2 * x, 2 * x + 1]
        np.testing.assert_allclose(
            before.gram[np.ix_(idx, idx)], after.gram[np.ix_(idx, idx)], atol=1e-12
        )
    assert np.linalg.norm(before.gram - after.gram) > 0.01


def test_clean_work_leakage_is_characteristic_variance_not_score_bias():
    for bits in [2, 4, 8]:
        result = coherent_channel(
            PARAMS, XS, AXIS, phase_bits=4, magnitude_bits=bits, gamma=0.8, taper="sine"
        )
        characteristic = result.distributions @ np.exp(-0.8j * result.estimates)
        exact_reset = np.repeat(abs(characteristic) ** 2, 2)
        np.testing.assert_allclose(result.zero_probability, exact_reset, atol=1e-12)
        mean = result.distributions @ result.estimates
        variance = result.distributions @ result.estimates**2 - mean**2
        assert np.max(1 - result.zero_probability) <= 0.8**2 * max(variance) + 1e-12
