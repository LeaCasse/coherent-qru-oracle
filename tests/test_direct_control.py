import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qru_registerization.channel import coherent_channel, schur_diamond
from qru_registerization.interface import phase_index_to_signed_code
from qru_registerization.qru import random_qru_params
from qru_registerization.structural import affine_direct_interface, affine_interface

PARAMS = random_qru_params(2, seed=42, scale=0.6)
AXIS = np.array([0.3, -0.4, np.sqrt(0.75)])
XS = np.linspace(-0.73, 0.91, 4)


def circuit_environment(circuit, layout):
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
    return np.asarray(env)


@pytest.mark.parametrize("taper", ["tophat", "sine"])
@pytest.mark.parametrize("cleanup", [False, True])
def test_direct_channel_matches_real_expanded_circuit(taper, cleanup):
    scores = -np.cos(2 * np.pi * np.arange(8) / 8)
    circuit, layout = affine_direct_interface(
        PARAMS,
        2,
        XS[0],
        XS[1] - XS[0],
        AXIS,
        phase_bits=3,
        taper=taper,
        cleanup=cleanup,
        fibre_step=0.7,
    )
    env = circuit_environment(circuit, layout)
    actual = coherent_channel(
        PARAMS,
        XS,
        AXIS,
        phase_bits=3,
        taper=taper,
        cleanup=cleanup,
        fibre=0.7 * np.arange(4),
        control_scores=scores,
    )
    np.testing.assert_allclose(actual.gram, env @ env.conj().T, atol=2e-11)
    np.testing.assert_allclose(actual.zero_probability, abs(env[:, 0]) ** 2, atol=2e-11)


def test_same_workspace_reuse_matches_dilation_not_channel_composition():
    n = 4
    scores = -np.cos(2 * np.pi * np.arange(n) / n) / np.cos(np.pi / n)
    circuit, layout = affine_direct_interface(
        PARAMS,
        2,
        XS[0],
        XS[1] - XS[0],
        AXIS,
        phase_bits=2,
        taper="sine",
        control_scores=scores,
    )
    repeated = circuit.compose(circuit)
    env = circuit_environment(repeated, layout)
    reused = coherent_channel(
        PARAMS,
        XS,
        AXIS,
        phase_bits=2,
        taper="sine",
        control_scores=scores,
        gamma=1.6,
    )
    base = coherent_channel(
        PARAMS,
        XS,
        AXIS,
        phase_bits=2,
        taper="sine",
        control_scores=scores,
    )
    np.testing.assert_allclose(env @ env.conj().T, reused.gram, atol=2e-11)
    assert np.linalg.norm(reused.gram - base.gram**2) > 0.01


def test_multiplexor_exactly_replaces_signed_contract():
    scores = np.array([phase_index_to_signed_code(y, 3, 4)[2] for y in range(8)])
    direct, ld = affine_direct_interface(
        PARAMS,
        2,
        XS[0],
        XS[1] - XS[0],
        AXIS,
        phase_bits=3,
        control_scores=scores,
    )
    signed, ls = affine_interface(
        PARAMS,
        2,
        XS[0],
        XS[1] - XS[0],
        AXIS,
        phase_bits=3,
        magnitude_bits=4,
    )
    ed, es = circuit_environment(direct, ld), circuit_environment(signed, ls)
    np.testing.assert_allclose(ed @ ed.conj().T, es @ es.conj().T, atol=2e-11)
    np.testing.assert_allclose(abs(ed[:, 0]) ** 2, abs(es[:, 0]) ** 2, atol=2e-11)
    assert signed.num_qubits - direct.num_qubits == 5


def test_direct_gauge_and_characteristic_variance_bounds():
    scores = -np.cos(2 * np.pi * np.arange(16) / 16)
    base = coherent_channel(PARAMS, XS, AXIS, phase_bits=4, control_scores=scores)
    changed = coherent_channel(
        PARAMS,
        XS,
        AXIS,
        phase_bits=4,
        control_scores=scores,
        fibre=1.3 * np.arange(4),
    )
    np.testing.assert_allclose(
        base.zero_probability, changed.zero_probability, atol=1e-12
    )
    assert schur_diamond(changed.gram - base.gram)["diamond_upper"] <= (
        2 * max(base.system_one_probability) + 2e-7
    )
    mean = base.distributions @ scores
    variance = base.distributions @ scores**2 - mean**2
    assert max(1 - base.zero_probability) <= 0.8**2 * max(variance) + 1e-12


def test_invalid_control_table_rejected():
    with pytest.raises(ValueError, match="control score"):
        coherent_channel(PARAMS, XS, AXIS, control_scores=[0, 1])
    with pytest.raises(ValueError, match="control score"):
        affine_direct_interface(PARAMS, 2, 0, 1, AXIS, control_scores=[np.nan] * 8)


@pytest.mark.parametrize("phase_bits", [2, 3, 4, 5])
def test_sine_calibration_is_unbiased_and_has_quadratic_channel_bound(phase_bits):
    n = 2**phase_bits
    scores = -np.cos(2 * np.pi * np.arange(n) / n) / np.cos(np.pi / n)
    from qru_registerization.channel import channel_metrics

    result = coherent_channel(
        PARAMS,
        XS,
        AXIS,
        phase_bits=phase_bits,
        taper="sine",
        control_scores=scores,
    )
    np.testing.assert_allclose(result.distributions @ scores, result.scores, atol=2e-12)
    metrics = channel_metrics(result)
    assert metrics["diamond_upper"] <= 2 * 0.8**2 * metrics["score_mse_max"] + 2e-7


def test_calibration_moments_and_uniform_variance_on_eigenphase_grid():
    # Independent Fourier calculation, including the score endpoints.
    phases = np.linspace(0, np.pi, 257)
    for m in [2, 3, 4, 5, 6]:
        n = 2**m
        a = np.sqrt(2 / n) * np.sin(np.pi * np.arange(n) / n)
        probability = (
            abs(
                np.fft.fft(
                    a[None, :] * np.exp(1j * phases[:, None] * np.arange(n)),
                    axis=1,
                )
            )
            ** 2
            / n
        )
        kappa = np.cos(np.pi / n)
        scores = -np.cos(2 * np.pi * np.arange(n) / n) / kappa
        mean = probability @ scores
        np.testing.assert_allclose(mean, -np.cos(phases), atol=2e-12)
        variance = probability @ scores**2 - mean**2
        assert max(variance) <= 2 * (1 - kappa) / kappa**2 + 2e-12


def test_calibrated_angles_match_expanded_circuit():
    n = 8
    scores = -np.cos(2 * np.pi * np.arange(n) / n) / np.cos(np.pi / n)
    circuit, layout = affine_direct_interface(
        PARAMS,
        2,
        XS[0],
        XS[1] - XS[0],
        AXIS,
        phase_bits=3,
        taper="sine",
        control_scores=scores,
    )
    env = circuit_environment(circuit, layout)
    actual = coherent_channel(
        PARAMS,
        XS,
        AXIS,
        phase_bits=3,
        taper="sine",
        control_scores=scores,
    )
    np.testing.assert_allclose(actual.gram, env @ env.conj().T, atol=2e-11)
