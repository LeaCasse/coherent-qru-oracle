"""Fresh-model comparison: signed ANF vs direct phase-word rotation.

Fixed protocol; no fitting or selection on these seeds. Every simulation is
the complete retained channel, every resource circuit expands all Q queries.
"""

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from qiskit import transpile

from qru_registerization.channel import channel_metrics, coherent_channel, schur_diamond
from qru_registerization.interface import phase_index_to_signed_code
from qru_registerization.qru import random_qru_params
from qru_registerization.structural import affine_direct_interface, affine_interface

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("QRU_RESULTS_DIR", ROOT / "outputs/results")) / "direct_control"
AXIS = np.array([0.3, -0.4, np.sqrt(0.75)])
XS = np.linspace(-0.73, 0.91, 4)
GAMMA = 0.8


def write(name, rows):
    temporary=(OUT/name).with_suffix(".csv.partial")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(OUT/name)

def run():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "depth_2_seeds": list(range(400, 412)),
        "depth_4_seeds": list(range(500, 506)),
        "phase_bits": [2, 3, 4, 5],
        "policies": ["signed4", "signed8", "direct"],
        "windows": ["tophat", "sine"],
        "xs": XS.tolist(),
        "axis": AXIS.tolist(),
        "gamma": GAMMA,
        "fibre": (1.3 * np.arange(4)).tolist(),
        "compilation": "level 0, rz/sx/x/cx, seed 17, no topology",
        "scope": (
            "fresh random QRUs; fixed model-score target; "
            "no training, no hardware noise"
        ),
        "source_sha256": {},
    }
    for path in [
        Path(__file__),
        ROOT / "src/qru_registerization/channel.py",
        ROOT / "src/qru_registerization/structural.py",
    ]:
        protocol["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    rows, gauges = [], []
    for depth, seeds in [(2, range(400, 412)), (4, range(500, 506))]:
        for seed in seeds:
            params = random_qru_params(depth, seed=seed, scale=0.6)
            for m in [2, 3, 4, 5]:
                scores = -np.cos(2 * np.pi * np.arange(2**m) / 2**m)
                for window in ["tophat", "sine"]:
                    for policy, b in [("signed4", 4), ("signed8", 8), ("direct", 4)]:
                        result = coherent_channel(
                            params,
                            XS,
                            AXIS,
                            m,
                            b,
                            gamma=GAMMA,
                            taper=window,
                            control_scores=scores if policy == "direct" else None,
                        )
                        rows.append(
                            {
                                "depth": depth,
                                "seed": seed,
                                "phase_bits": m,
                                "window": window,
                                "policy": policy,
                                **channel_metrics(result),
                            }
                        )
                        write("channels.csv", rows)
                        if policy == "direct":
                            changed = coherent_channel(
                                params,
                                XS,
                                AXIS,
                                m,
                                gamma=GAMMA,
                                taper=window,
                                fibre=1.3 * np.arange(4),
                                control_scores=scores,
                            )
                            mean = result.distributions @ scores
                            variance = result.distributions @ scores**2 - mean**2
                            gauges.append(
                                {
                                    "depth": depth,
                                    "seed": seed,
                                    "phase_bits": m,
                                    "window": window,
                                    **schur_diamond(changed.gram - result.gram),
                                    "sector_bound": 2
                                    * max(result.system_one_probability),
                                    "variance_bound": 2 * GAMMA**2 * max(variance),
                                    "reset_change": max(
                                        abs(
                                            changed.zero_probability
                                            - result.zero_probability
                                        )
                                    ),
                                }
                            )
                            write("gauge.csv", gauges)
            print(
                f"Channels: depth {depth}, seed {seed}, {len(rows)} cases", flush=True
            )
    run_resources()


def run_resources():
    OUT.mkdir(parents=True,exist_ok=True)
    resources = []
    params = random_qru_params(2, seed=400, scale=0.6)
    for m in [2, 3, 4, 5]:
        quantized = np.array(
            [phase_index_to_signed_code(y, m, 4)[2] for y in range(2**m)]
        )
        for window in ["tophat", "sine"]:
            for policy, b in [
                ("signed4", 4),
                ("signed8", 8),
                ("multiplex4", 4),
                ("direct", 4),
            ]:
                kwargs = {"phase_bits": m, "gamma": GAMMA, "taper": window}
                if policy.startswith("signed"):
                    circuit, _ = affine_interface(
                        params,
                        2,
                        XS[0],
                        XS[1] - XS[0],
                        AXIS,
                        magnitude_bits=b,
                        **kwargs,
                    )
                else:
                    circuit, _ = affine_direct_interface(
                        params,
                        2,
                        XS[0],
                        XS[1] - XS[0],
                        AXIS,
                        control_scores=quantized if policy == "multiplex4" else None,
                        **kwargs,
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
                        "window": window,
                        "policy": policy,
                        "qubits": circuit.num_qubits,
                        "cx": compiled.count_ops().get("cx", 0),
                        "rz": compiled.count_ops().get("rz", 0),
                        "depth_compiled": compiled.depth(),
                        "oracle_calls": 2 * (2 ** (m + 1) - 1),
                    }
                )
                write("resources.csv", resources)
            print(f"Compiled: m={m}, {window}", flush=True)


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--resources-only",action="store_true")
    if parser.parse_args().resources_only:run_resources()
    else:run()
