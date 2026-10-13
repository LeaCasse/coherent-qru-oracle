"""Fresh independent-model experiments; each completed row is durably flushed.

Run from repository: python experiments/channel_study.py
with PYTHONPATH=src and OPENBLAS_NUM_THREADS=1.
"""

import csv
import hashlib
import json
import platform
import time
import os
from pathlib import Path

import numpy as np
from qiskit import transpile
from scipy.signal.windows import dpss

from qru_registerization.channel import channel_metrics, coherent_channel, schur_diamond
from qru_registerization.interface import (
    phase_index_to_signed_code,
    phase_taper_amplitudes,
)
from qru_registerization.qru import qru_unitary, random_qru_params, signed_coordinate
from qru_registerization.structural import affine_interface

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("QRU_RESULTS_DIR", ROOT / "outputs/results")) / "channel_study"
AXIS = np.array([0.3, -0.4, np.sqrt(0.75)])
XS = np.linspace(-0.73, 0.91, 4)
GAMMA = 0.8


def write_rows(name, rows):
    rows = list(rows)
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            f.flush()


def learned_window(m):
    n = 2**m
    times = np.arange(n)
    bins = times[:, None]
    estimates = np.array([phase_index_to_signed_code(y, m, m - 1)[2] for y in times])
    kernel = np.zeros((n, n), complex)
    count = 0
    for seed in range(100, 106):
        params = random_qru_params(2, seed=seed, scale=0.6)
        for x in np.linspace(-1, 1, 17):
            score = signed_coordinate(qru_unitary(params, x)[:, 0], AXIS)
            theta = np.arcsin(np.sqrt((1 + score) / 2)) / np.pi
            loss = np.sin(GAMMA * (estimates - score)) ** 2
            for phase in [theta, 1 - theta]:
                transfer = np.exp(2j * np.pi * times * (phase - bins / n)) / np.sqrt(n)
                kernel += 0.5 * (transfer.conj().T @ (loss[:, None] * transfer))
            count += 1
    kernel /= count
    # Loss is even in y and the eigenphases are +/-theta: K is real symmetric.
    assert np.max(abs(kernel.imag)) < 1e-12
    values, vectors = np.linalg.eigh(kernel.real)
    a = vectors[:, 0]
    if a[np.argmax(abs(a))] < 0:
        a = -a
    return a, float(values[0]), float(values[1] - values[0])


def run():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    windows = {}
    training = []
    for m in range(2, 6):
        learned, eigenvalue, gap = learned_window(m)
        prolate = dpss(2**m, 1.0, Kmax=1, sym=False)[0]
        prolate /= np.linalg.norm(prolate)
        windows[m] = {
            "tophat": phase_taper_amplitudes(m),
            "sine": phase_taper_amplitudes(m, "sine"),
            "dpss": prolate,
            "learned_scalar": learned,
        }
        training.append(
            {
                "phase_bits": m,
                "minimum_train_scalar_loss": eigenvalue,
                "kernel_eigengap": gap,
            }
        )
    np.savez(
        OUT / "windows.npz",
        **{f"{name}_m{m}": a for m, w in windows.items() for name, a in w.items()},
    )
    write_rows("window_training.csv", training)
    print("Window training complete", flush=True)

    rows = []
    for split, depth, seeds in [
        ("test", 2, range(200, 212)),
        ("shift_depth", 4, range(300, 306)),
    ]:
        for seed in seeds:
            params = random_qru_params(depth, seed=seed, scale=0.6)
            for m in range(2, 6):
                for window, a in windows[m].items():
                    result = coherent_channel(
                        params, XS, AXIS, m, m - 1, GAMMA, amplitudes=a
                    )
                    metrics = channel_metrics(result)
                    scalar_loss = np.mean(
                        np.sum(
                            result.distributions
                            * np.sin(
                                GAMMA
                                * (result.estimates[None, :] - result.scores[:, None])
                            )
                            ** 2,
                            axis=1,
                        )
                    )
                    rows.append(
                        {
                            "split": split,
                            "depth": depth,
                            "seed": seed,
                            "window": window,
                            "phase_bits": m,
                            "magnitude_bits": m - 1,
                            "oracle_calls": 2 * (2 ** (m + 1) - 1),
                            "fixed_input_scalar_loss": scalar_loss,
                            "mse_diamond_bound": min(
                                2.0, 2 * GAMMA * np.sqrt(metrics["score_mse_max"])
                            ),
                            **metrics,
                        }
                    )
                    write_rows("channels.csv", rows)
            print(f"Channels {split} seed {seed} ({len(rows)} rows)", flush=True)

    gauge = []
    for seed in range(200, 206):
        params = random_qru_params(2, seed=seed, scale=0.6)
        for m in range(2, 6):
            for cleanup in [False, True]:
                base = coherent_channel(params, XS, AXIS, m, m - 1, cleanup=cleanup)
                changed = coherent_channel(
                    params,
                    XS,
                    AXIS,
                    m,
                    m - 1,
                    cleanup=cleanup,
                    fibre=np.arange(4) * 1.3,
                )
                gauge.append(
                    {
                        "seed": seed,
                        "phase_bits": m,
                        "cleanup": int(cleanup),
                        "system_one_bound": 2 * max(base.system_one_probability),
                        "work_leakage_bound": 2 * (1 - min(base.zero_probability)),
                        "zero_probability_change": max(
                            abs(base.zero_probability - changed.zero_probability)
                        ),
                        **schur_diamond(changed.gram - base.gram),
                    }
                )
    write_rows("gauge.csv", gauge)
    dephased = []
    for m in range(2, 6):
        for gamma in [0.0, 0.8]:
            for cleanup in [False, True]:
                params = random_qru_params(2, seed=200, scale=0.6)
                base = coherent_channel(
                    params, XS, AXIS, m, m - 1, gamma, cleanup=cleanup
                )
                full = coherent_channel(
                    params, XS, AXIS, m, m - 1, gamma, cleanup=cleanup, dephase=1
                )
                for rate in [0.0, 0.01, 0.1, 0.5, 1.0]:
                    result = coherent_channel(
                        params, XS, AXIS, m, m - 1, gamma, cleanup=cleanup, dephase=rate
                    )
                    dephased.append(
                        {
                            "phase_bits": m,
                            "gamma": gamma,
                            "cleanup": int(cleanup),
                            "dephase": rate,
                            "dephase_channel_change": schur_diamond(
                                result.gram - base.gram
                            )["diamond_lower"],
                            **channel_metrics(result),
                        }
                    )
                if not cleanup:
                    assert np.max(abs(full.gram - base.gram)) < 1e-11
    write_rows("dephasing.csv", dephased)
    print("Gauge and dephasing complete", flush=True)
    resources = []
    params = random_qru_params(2, seed=200, scale=0.6)
    for depth in [2, 4]:
        params = random_qru_params(depth, seed=200, scale=0.6)
        for m in range(2, 6):
            for window, a in windows[m].items():
                custom = a if window in ["dpss", "learned_scalar"] else None
                circuit, _ = affine_interface(
                    params,
                    2,
                    XS[0],
                    XS[1] - XS[0],
                    AXIS,
                    m,
                    m - 1,
                    taper=window if custom is None else "tophat",
                    amplitudes=custom,
                )
                compiled = transpile(
                    circuit,
                    basis_gates=["rz", "sx", "x", "cx"],
                    optimization_level=0,
                    seed_transpiler=17,
                )
                resources.append(
                    {
                        "depth": depth,
                        "phase_bits": m,
                        "window": window,
                        "qubits": circuit.num_qubits,
                        "cx": compiled.count_ops().get("cx", 0),
                        "depth_compiled": compiled.depth(),
                        "gates": sum(compiled.count_ops().values()),
                        "oracle_calls": 2 * (2 ** (m + 1) - 1),
                    }
                )
                write_rows("resources.csv", resources)
                print(f"Resource compilation L={depth} m={m} {window}", flush=True)
    config = {
        "date": "2026-10-10",
        "input_xs": XS.tolist(),
        "axis": AXIS.tolist(),
        "gamma": GAMMA,
        "training_seeds": list(range(100, 106)),
        "test_seeds": list(range(200, 212)),
        "shift_seeds": list(range(300, 306)),
        "phase_bits": [2, 3, 4, 5],
        "signed_bits": "m-1",
        "dpss_NW": 1.0,
        "taper_sine": "sin(pi*n/N)",
        "access": (
            "affine binary loading; 2 input qubits; gate-expanded Grover repetitions"
        ),
        "noise": "phase-word dephasing intervention, not hardware gate noise",
        "compilation": "rz,sx,x,cx; level 0; seed 17; no coupling map",
        "scope": "H=Z; known single-qubit QRU; no training of QRU parameters",
        "python": platform.python_version(),
        "seconds": time.time() - start,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                ROOT / "src/qru_registerization/channel.py",
                ROOT / "src/qru_registerization/structural.py",
            ]
        },
    }
    (OUT / "protocol.json").write_text(json.dumps(config, indent=2))
    print(f"Completed in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    run()
