"""Analytic approximation audit, explicitly NOT a completed QSP baseline."""

import csv
import json
from math import exp, factorial
import os
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebval
from qiskit import transpile
from qiskit.quantum_info import Operator
from scipy.special import jv

from qru_registerization.protocol_circuits import score_block_encoding
from qru_registerization.qru import qru_unitary, random_qru_params, signed_coordinate

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("QRU_RESULTS_DIR", ROOT / "outputs/results")) / "process_study"


def tail_bound(t, degree):
    """2 sum_{j>D}|J_j(t)| <= exp(t²/4) times geometric factorial tail."""
    first = degree + 1
    ratio = abs(t) / (2 * (first + 1))
    if ratio >= 1:
        return float("inf")
    return 2 * exp(t * t / 4) * (abs(t) / 2) ** first / factorial(first) / (1 - ratio)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    axis = [0.3, -0.4, np.sqrt(0.75)]
    encodings = []
    for depth, seed in [(2, 610), (4, 710)]:
        params = random_qru_params(depth, seed=seed, scale=0.6)
        circuit = score_block_encoding(params, 1, -0.7, 1.6, axis, 1.3)
        matrix = Operator(circuit).data
        expected = [
            signed_coordinate(qru_unitary(params, x)[:, 0], axis) for x in [-0.7, 0.9]
        ]
        expanded = transpile(
            circuit,
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=0,
            seed_transpiler=27,
        )
        encodings.append(
            {
                "depth": depth,
                "seed": seed,
                "a_calls_per_block": 2,
                "qubits": circuit.num_qubits,
                "cx_count": expanded.count_ops().get("cx", 0),
                "score_block_error": float(
                    np.max(abs(matrix[:2, :2] - np.diag(expected)))
                ),
                "hermiticity_error": float(np.max(abs(matrix - matrix.conj().T))),
            }
        )
    rows = []
    grid = np.linspace(-1, 1, 20001)
    for t in [0.8, 1.6, 6.4]:
        for epsilon in [0.01, 0.001, 0.0001, 0.000001]:
            degree = next(d for d in range(1, 100) if tail_bound(t, d) <= epsilon)
            coefficients = np.array(
                [
                    jv(0, t) if j == 0 else 2 * (-1j) ** j * jv(j, t)
                    for j in range(degree + 1)
                ]
            )
            observed = float(
                np.max(abs(chebval(grid, coefficients) - np.exp(-1j * t * grid)))
            )
            rows.append(
                {
                    "time": t,
                    "epsilon": epsilon,
                    "degree": degree,
                    "analytic_tail_bound": tail_bound(t, degree),
                    "grid_max_scalar_error": observed,
                    "is_qsp_circuit": False,
                }
            )
            assert observed <= tail_bound(t, degree) + 1e-13
    with (OUT / "polynomial_audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "block_encoding.json").write_text(
        json.dumps(
            {
                "implementation": "B=-A† Z_system A",
                "zero_system_block": "diag(s_x)",
                "target_hamiltonian": "diag(s_x) tensor Z_target",
                "circuits": encodings,
                "status": (
                    "block encoding verified; polynomial approximation verified; "
                    "full QSP/OAA circuit absent"
                ),
                "caveat": (
                    "degrees are not compiled CX counts or complete QSP query counts"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
