# Coherent QRU score oracles

For the Paper "When coherent score oracles compose: workspace memory and quaternion covariance" - Léa Cassé, Albert Bifet, Bernhard Pfahringer

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python reproduce.py verify
python reproduce.py figures
python reproduce.py notebook
```

Full recalculation and comparison:

```bash
python reproduce.py experiments
python reproduce.py compare
python reproduce.py figures --data outputs/results
```

Use `experiments --study process_study` to run one campaign. Results are saved during the campaigns, under `outputs/results/`. Numerical SDP witnesses may vary within the comparison tolerance.

- `src/`: quantum model, estimators, channels and structural circuits
- `experiments/`: the eight numerical campaigns used by the papers
- `figures/`: generators for all 20 figures, including the compact subset
- `reference_data/`: immutable paper tables and campaign specifications
- `notebooks/reproduce_paper.ipynb`: executed notebook
- `tests/`: channel, circuit and mathematical checks
