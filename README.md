# Coherent score oracles

Code and data for *Workspace contracts for coherent score oracles*.

The experiments cover QRU preparation geometry, QPE and GQSP, workspace reuse and certified renewal.

## Run

Use Python 3.12:

```bash
python -m pip install -e ".[test]"
python reproduce.py verify
python reproduce.py figures
```

The figures used in the paper are written to `outputs/figures/paper/`. This uses the saved results, so there is no need to rerun the experiments first.

To recompute the V3 studies:

```bash
python reproduce.py experiments
```

See [REPRODUCING.md](REPRODUCING.md) for individual studies, data files and earlier results.
