"""Exact density-matrix pilot on structural circuits and a two-qubit reference."""

import csv
import json
import os
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import partial_trace
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qru_registerization.process import process_channel, trace_norm
from qru_registerization.protocol_circuits import alternating_protocol
from qru_registerization.qru import random_qru_params

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("QRU_RESULTS_DIR", ROOT / "outputs/results")) / "process_study"


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    axis = [0.3, -0.4, np.sqrt(0.75)]
    specification = {
        "seeds": [610, 611, 612],
        "depth": 2,
        "phase_bits": 2,
        "protocol": "two noncommuting calibrated Z,X controls",
        "noise": "Aer depolarizing_error(p,2) after every compiled CX",
        "parameters": [0.0, 0.00001, 0.0001, 0.001],
        "basis": ["rz", "sx", "x", "cx"],
        "optimization_level": 0,
        "perfect_operations": (
            "single-qubit gates, Bell preparation, reference, final trace"
        ),
        "no_sampling": True,
        "topology": None,
        "metric": "normalized Choi trace norm; not diamond",
        "purpose": (
            "small sensitivity pilot; not hardware validation "
            "or noise-general guarantee"
        ),
    }
    (OUT / "noise_protocol.json").write_text(json.dumps(specification, indent=2))
    for seed in specification["seeds"]:
        params = random_qru_params(2, seed=seed, scale=0.6)
        ideal_result = process_channel(
            params,
            [-0.7, 0.9],
            axis,
            [([0, 0, 1], 0.8), ([1, 0, 0], 0.8)],
            phase_bits=2,
        )
        vector = ideal_result.ideal.ravel(order="F")
        ideal_choi = np.outer(vector, vector.conj())
        for fused in [False, True]:
            circuit, layout = alternating_protocol(
                params,
                -0.7,
                1.6,
                axis,
                fused=fused,
            )
            expanded = transpile(
                circuit,
                basis_gates=specification["basis"],
                optimization_level=0,
                seed_transpiler=27,
            )
            total = circuit.num_qubits
            target = layout["target"]
            # Keep ideal Bell preparation separate so the noise only acts
            # on the service. Aer initial density matrix is supplied directly.
            prep = QuantumCircuit(total + 2)
            prep.h(0)
            prep.cx(0, total)
            prep.h(target)
            prep.cx(target, total + 1)
            from qiskit.quantum_info import DensityMatrix

            density = DensityMatrix.from_instruction(prep)
            service = QuantumCircuit(total + 2)
            service.set_density_matrix(density)
            service.compose(expanded, qubits=range(total), inplace=True)
            service.save_density_matrix()
            for p in specification["parameters"]:
                noise = NoiseModel()
                if p:
                    noise.add_all_qubit_quantum_error(depolarizing_error(p, 2), "cx")
                simulator = AerSimulator(
                    method="density_matrix",
                    noise_model=noise,
                    max_parallel_threads=1,
                    max_parallel_experiments=1,
                )
                output = simulator.run(service).result().data(0)["density_matrix"]
                reduced = np.asarray(partial_trace(output, layout["work"]).data)
                indices = [
                    x + 2 * t + 4 * rx + 8 * rt
                    for rx in range(2)
                    for rt in range(2)
                    for x in range(2)
                    for t in range(2)
                ]
                choi = 4 * reduced[np.ix_(indices, indices)]
                if not p:
                    discrepancy = float(np.max(abs(choi - ideal_result.choi)))
                    assert discrepancy < 3e-11, discrepancy
                rows.append(
                    {
                        "seed": seed,
                        "implementation": "fused" if fused else "unfused",
                        "cx_depolarizing_parameter": p,
                        "cx_count": expanded.count_ops().get("cx", 0),
                        "qubits_service": total,
                        "qubits_simulated": total + 2,
                        "choi_fidelity": float(
                            np.vdot(vector, choi @ vector).real / 16
                        ),
                        "choi_trace_norm": trace_norm(choi - ideal_choi) / 4,
                        "noise_increment_trace_norm": trace_norm(
                            choi - ideal_result.choi
                        )
                        / 4,
                    }
                )
            print(f"Noise seed={seed}, fused={fused}", flush=True)
            temporary = OUT / "gate_noise.csv.partial"
            with temporary.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(OUT / "gate_noise.csv")


if __name__ == "__main__":
    run()
