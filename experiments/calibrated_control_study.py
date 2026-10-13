"""Analytic sine-window calibration; follow-up on the same fresh cohort."""

import csv
import json
import os
from pathlib import Path

import numpy as np
from qiskit import transpile

from qru_registerization.channel import channel_metrics, coherent_channel, schur_diamond
from qru_registerization.qru import random_qru_params
from qru_registerization.structural import affine_direct_interface

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("QRU_RESULTS_DIR", ROOT / "outputs/results")) / "direct_control"
AXIS = [0.3, -0.4, np.sqrt(0.75)]
XS = np.linspace(-0.73, 0.91, 4)


def write(name, rows):
    temporary = (OUT / name).with_suffix(".csv.partial")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUT / name)


def run(resume=False):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration_protocol.json").write_text(
        json.dumps(
            {
                "status": (
                    "analytic follow-up; same cohort as direct comparison; "
                    "no new holdout"
                ),
                "formula": "-cos(2*pi*y/N)/cos(pi/N)",
                "scope": (
                    "sine window sin(pi*n/N); no clipping; "
                    "not sine((n+1)*pi/(N+1))"
                ),
                "depth_2_seeds": list(range(400, 412)),
                "depth_4_seeds": list(range(500, 506)),
                "phase_bits": [2, 3, 4, 5],
                "gamma": 0.8,
            },
            indent=2,
        )
    )
    rows = []
    if resume and (OUT / "calibrated.csv").exists():
        with (OUT / "calibrated.csv").open() as file:
            rows = list(csv.DictReader(file))
    completed = {(int(r["seed"]), int(r["phase_bits"])) for r in rows}
    for depth, seeds in [(2, range(400, 412)), (4, range(500, 506))]:
        for seed in seeds:
            params = random_qru_params(depth, seed=seed, scale=0.6)
            for m in [2, 3, 4, 5]:
                if (seed, m) in completed:
                    continue
                n = 2**m
                scores = -np.cos(2 * np.pi * np.arange(n) / n) / np.cos(np.pi / n)
                result = coherent_channel(
                    params,
                    XS,
                    AXIS,
                    m,
                    taper="sine",
                    control_scores=scores,
                )
                bias = max(abs(result.distributions @ scores - result.scores))
                metrics = channel_metrics(result)
                changed = coherent_channel(
                    params,
                    XS,
                    AXIS,
                    m,
                    taper="sine",
                    control_scores=scores,
                    fibre=1.3 * np.arange(4),
                )
                rows.append(
                    {
                        "depth": depth,
                        "seed": seed,
                        "phase_bits": m,
                        "window": "sine",
                        "policy": "calibrated",
                        **metrics,
                        "bias_max": bias,
                        "bias_variance_bound": 2 * 0.8 * bias
                        + 2 * 0.8**2 * metrics["score_mse_max"],
                        "gauge_upper": schur_diamond(changed.gram - result.gram)[
                            "diamond_upper"
                        ],
                        "sector_bound": 2 * max(result.system_one_probability),
                    }
                )
                write("calibrated.csv", rows)
            print(f"Calibrated: depth {depth}, seed {seed}", flush=True)
    resources = []
    params = random_qru_params(2, seed=400, scale=0.6)
    for m in [2, 3, 4, 5]:
        n = 2**m
        scores = -np.cos(2 * np.pi * np.arange(n) / n) / np.cos(np.pi / n)
        circuit, _ = affine_direct_interface(
            params,
            2,
            XS[0],
            XS[1] - XS[0],
            AXIS,
            m,
            taper="sine",
            control_scores=scores,
        )
        compiled = transpile(
            circuit,
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=0,
            seed_transpiler=17,
        )
        resources.append(
            {
                "depth": 2,
                "phase_bits": m,
                "window": "sine",
                "policy": "calibrated",
                "qubits": circuit.num_qubits,
                "cx": compiled.count_ops().get("cx", 0),
                "rz": compiled.count_ops().get("rz", 0),
                "depth_compiled": compiled.depth(),
                "oracle_calls": 2 * (2 ** (m + 1) - 1),
            }
        )
        write("calibrated_resources.csv", resources)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    run(resume=parser.parse_args().resume)
