"""Decouple phase bits from signed magnitude bits (exploratory ablation)."""

import csv
import json
import os
from pathlib import Path

import numpy as np
from qiskit import transpile

from qru_registerization.channel import channel_metrics, coherent_channel
from qru_registerization.qru import random_qru_params
from qru_registerization.structural import affine_interface

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("QRU_RESULTS_DIR", ROOT / "outputs/results")) / "channel_study"
AXIS = np.array([0.3, -0.4, np.sqrt(0.75)])
XS = np.linspace(-0.73, 0.91, 4)


def write(name, rows):
    temporary=(OUT/name).with_suffix(".csv.partial")
    with temporary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())

    temporary.replace(OUT/name)

def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(200, 206):
        p = random_qru_params(2, seed=seed, scale=0.6)
        for m in [3, 4, 5]:
            for b in [2, 4, 6, 8]:
                for window in ["tophat", "sine"]:
                    result = coherent_channel(p, XS, AXIS, m, b, taper=window)
                    rows.append(
                        {
                            "seed": seed,
                            "phase_bits": m,
                            "magnitude_bits": b,
                            "window": window,
                            **channel_metrics(result),
                        }
                    )
                    write("precision.csv", rows)
        print(f"Precision seed {seed}", flush=True)
    resources = []
    p = random_qru_params(2, seed=200, scale=0.6)
    for m in [3, 4, 5]:
        for b in [2, 4, 6, 8]:
            for window in ["tophat", "sine"]:
                c, _ = affine_interface(
                    p, 2, XS[0], XS[1] - XS[0], AXIS, m, b, taper=window
                )
                t = transpile(
                    c,
                    basis_gates=["rz", "sx", "x", "cx"],
                    optimization_level=0,
                    seed_transpiler=17,
                )
                resources.append(
                    {
                        "phase_bits": m,
                        "magnitude_bits": b,
                        "window": window,
                        "qubits": c.num_qubits,
                        "cx": t.count_ops().get("cx", 0),
                        "depth_compiled": t.depth(),
                        "oracle_calls": 2 * (2 ** (m + 1) - 1),
                    }
                )
                write("precision_resources.csv", resources)
    write("precision.csv",rows)
    assert len(list(csv.DictReader((OUT/"precision.csv").open())))==len(rows)
    (OUT / "precision_protocol.json").write_text(
        json.dumps(
            {
                "seeds": list(range(200, 206)),
                "phase_bits": [3, 4, 5],
                "magnitude_bits": [2, 4, 6, 8],
                "scope": (
                    "post hoc exploratory ablation; same 6 held-out QRUs reused; "
                    "no independent tuning claim"
                ),
                "encoding": "truncated signed magnitude; |1| saturates at 1-2^-b",
                "compilation": "level 0; rz,sx,x,cx; seed 17; no topology",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
