# Reproduce the paper

`python reproduce.py verify` checks the saved data, circuit identities and tests. `python reproduce.py figures` builds all 17 figures used across the compact and long manuscripts. No LaTeX installation is needed.

## Experiments

Run all V3 studies with `python reproduce.py experiments`, or choose one:

```bash
python reproduce.py experiments --study direct_memory_validation
```

| Study | Saved results in `reference_data/v3/` |
|---|---|
| `precision` | `precision.csv`: QPE/GQSP at common error requirements |
| `resources` | `resources.csv`, `gqsp_synthesis.json`: expanded circuits and phases |
| `fibre` | `fibre.csv`: prepared and invariant initialization |
| `renewal` | `renewal_validation.csv`, `renewal_scaling.csv`: certificates and partitions |
| `inverse` | `inverse.csv`: matched error, mismatch and drift |
| `amortization` | `amortization.csv`: one numerical estimate across multiple controls |
| `amortization_resources` | `amortization_resources.csv`: compiled sequence costs |
| `direct_memory` | `direct_memory.csv`: GQSP workspace policies |
| `direct_memory_validation` | `direct_memory_validation.csv`: new-model check |
| `noise` | `noise/gate_noise.csv`: optimized CX-noise pilot |

New runs go to `outputs/results/v3/`. To plot those results:

```bash
python reproduce.py figures --data outputs/results/v3
```

Each study writes its configuration alongside its tables. The saved `environment.json` records the environment used for the paper. Development pilots remain in `reference_data/v3/development/`.

## Conventions

Diamond errors use the full norm. Normalized Choi trace norms are separate witnesses. Literal return refers to the implemented clean-input isometry. The tables retain every declared model and workspace policy.

Preparation calls, CX counts, qubits and reset episodes are reported separately. Amortization compares shared numerical estimation with modular fresh GQSP. In `inverse.csv`, optional diamond fields measure the change from the unperturbed implementation, while `choi_error` measures distance from the exact-score ideal.

Certificates and numerical SDP witnesses are evaluated in floating point. The SDP routines retain feasibility repairs and witness gaps.

## Earlier results

The original tables remain in the other `reference_data/` folders. Their hashes are unchanged.

```bash
python reproduce.py experiments --version v2
python reproduce.py figures --version v2
python reproduce.py compare
```

The earlier worked notebook is optional:

```bash
python -m pip install -e ".[notebook]"
python reproduce.py notebook
```
