"""Fixed intervention study: whole input-target channels, not scalar scores."""

import csv
import json
import os
from pathlib import Path

import numpy as np

from qru_registerization.process import diamond_choi, process_channel, process_metrics
from qru_registerization.qru import random_qru_params

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("QRU_RESULTS_DIR", ROOT / "outputs/results")) / "process_study"
XS = [-0.7, 0.9]
AXIS = [0.3, -0.4, np.sqrt(0.75)]
Z = [0, 0, 1]
X = [1, 0, 0]
TX = np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))
HX = np.kron(np.array([[1, 1], [1, -1]]) / np.sqrt(2), np.eye(2))


def protocol(name, k):
    if name == "repeated_z":
        return [(Z, 0.8)] * k, [np.eye(4)] * (k - 1)
    if name == "echo":
        return [(Z, 0.8)] * k, [TX] * (k - 1)
    if name == "alternating":
        return [(Z if j % 2 == 0 else X, 0.8) for j in range(k)], [np.eye(4)] * (k - 1)
    if name == "input_mixed_echo":
        return [(Z, 0.8)] * k, [HX @ TX] * (k - 1)
    raise ValueError(name)


def save(rows, filename):
    path = OUT / filename
    temporary = path.with_suffix(".csv.partial")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    specification = {
        "depth_two_seeds": list(range(600, 606)),
        "depth_four_seeds": list(range(700, 703)),
        "inputs": XS,
        "axis": AXIS,
        "phase_bits": [2, 3, 4],
        "repetitions": [2, 4, 8],
        "protocols": ["repeated_z", "echo", "alternating", "input_mixed_echo"],
        "modes": ["shared", "fresh"],
        "window": "repository sine",
        "table": "unclipped analytic calibration",
        "diamond_subset": (
            "depth=2,seeds=600..602,m=2 or 4,k=2 or 8; all protocols and modes"
        ),
        "notes": [
            (
                "seed 600 used for implementation verification; "
                "seeds 601..605 and 700..702 new"
            ),
            (
                "remaining cases report normalized Choi trace norm, "
                "never relabelled diamond"
            ),
            "floating-point SDP witnesses; not interval certificates",
            "ideal gates; frozen random models; no external-task advantage",
        ],
    }
    (OUT / "protocol.json").write_text(json.dumps(specification, indent=2))
    rows = []
    for depth, seeds in [(2, range(600, 606)), (4, range(700, 703))]:
        for seed in seeds:
            params = random_qru_params(depth, seed=seed, scale=0.6)
            for m in [2, 3, 4]:
                for k in [2, 4, 8]:
                    for name in specification["protocols"]:
                        pulses, gates = protocol(name, k)
                        for mode in ["shared", "fresh"]:
                            use_sdp = depth == 2 and seed <= 602 and m != 3 and k != 4
                            result = process_channel(
                                params,
                                XS,
                                AXIS,
                                pulses,
                                gates,
                                phase_bits=m,
                                mode=mode,
                            )
                            metrics = process_metrics(result, diamond=use_sdp)
                            delta = result.estimates[None, :] - result.scores[:, None]
                            bias = float(
                                np.max(
                                    abs(np.sum(result.distributions * delta, axis=1))
                                )
                            )
                            mu = float(
                                np.max(np.sum(result.distributions * delta**2, axis=1))
                            )
                            budget = sum(abs(a) for _, a in pulses)
                            # Exact echo has F'(g)=F''(g)=0 for even k.
                            if name == "echo":
                                budget = 0.0
                            applicable = mode == "shared" and name != "input_mixed_echo"
                            rows.append(
                                {
                                    "depth": depth,
                                    "seed": seed,
                                    "phase_bits": m,
                                    "repetitions": k,
                                    "protocol": name,
                                    "mode": mode,
                                    "has_diamond_sdp": use_sdp,
                                    "bias": bias,
                                    "second_moment": mu,
                                    "smooth_protocol_bound": min(
                                        2.0, 2 * budget * bias + 2 * budget**2 * mu
                                    )
                                    if applicable
                                    else "",
                                    **metrics,
                                }
                            )
                save(rows, "channels.csv")
            print(
                f"Completed depth={depth}, seed={seed}: {len(rows)} cases", flush=True
            )
    gauge = []
    for seed in range(601, 604):
        params = random_qru_params(2, seed=seed, scale=0.6)
        for name in ["repeated_z", "echo", "alternating", "input_mixed_echo"]:
            pulses, gates = protocol(name, 4)
            baseline = process_channel(params, XS, AXIS, pulses, gates, phase_bits=3)
            p1 = float(
                np.linalg.eigvalsh(sum(k.conj().T @ k for k in baseline.kraus[1::2]))[
                    -1
                ]
            )
            for phi in [0.7, 2.1]:
                changed = process_channel(
                    params,
                    XS,
                    AXIS,
                    pulses,
                    gates,
                    phase_bits=3,
                    fibre=[0, phi],
                )
                gauge.append(
                    {
                        "seed": seed,
                        "protocol": name,
                        "phase": phi,
                        "input_preserved": name != "input_mixed_echo",
                        "system_one_operator_probability": p1,
                        "sector_bound": min(2.0, 2 * p1),
                        **diamond_choi(changed.choi - baseline.choi),
                    }
                )
        save(gauge, "gauge.csv")
    summary = {
        "channel_cases": len(rows),
        "diamond_cases": sum(r["has_diamond_sdp"] for r in rows),
        "gauge_cases": len(gauge),
        "max_sdp_gap": max(
            [r.get("sdp_gap", 0) for r in rows] + [r["sdp_gap"] for r in gauge]
        ),
        "max_tp_error": max(r["tp_max_error"] for r in rows),
        "min_cp_eigenvalue": min(r["cp_min_eigenvalue"] for r in rows),
        "max_bias": max(r["bias"] for r in rows),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
