# Coherent QRU score oracles

Code for *Composing coherent score oracles: workspace contracts and quaternion covariance*.

Python 3.12:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python reproduce.py verify
python reproduce.py figures
python reproduce.py notebook
```

Full recalculation:

```bash
python reproduce.py experiments
python reproduce.py compare
python reproduce.py figures --data outputs/results
```

Run one campaign with `python reproduce.py experiments --study renewal_study`.
Results are saved during campaigns in `outputs/results/`. Figures use archived data by default.

- `src/`: QRUs, coherent channels, circuits and workspace contracts
- `experiments/`: nine campaigns
- `figures/`: generators for all 21 figures
- `reference_data/`: paper results and protocol specifications
- `notebooks/reproduce_paper.ipynb`: worked reproduction
- `tests/`: mathematical and circuit checks
