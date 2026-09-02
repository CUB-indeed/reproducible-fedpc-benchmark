# FedPC — Federated PC Algorithm for Constraint-Based Causal Discovery

Simulation and validation codebase for **FedPC**, a federated adaptation of the
PC algorithm that aggregates per-client separating sets (Sepsets) under
weighted voting, rather than aggregating adjacency votes alone.

**Author:** Widyasmoro Priatmojo

---

## Contents

| File | Purpose |
|---|---|
| `fedpc_core.py` | Core algorithm: CI test, skeleton search, Sepset aggregation, v-structure/Meek orientation, experiment runners, metrics, and figure generation. Everything else imports from this file. |
| `part1a_stage1_first_half.py` | Stage 1 (main sweep), scenarios 1–5 of 10. Writes `results/raw/main_sweep_raw.csv`. |
| `part1b_stage1_second_half.py` | Stage 1 (main sweep), scenarios 6–10 of 10. Appends to the same CSV as Part 1a. |
| `part2a_stage2_first_half.py` | Stage 2 (mechanism-shift sweep), scenarios 1–3 of 5. Writes `results/raw/mechshift_sweep_raw.csv`. |
| `part2b_stage2_second_half.py` | Stage 2 (mechanism-shift sweep), scenarios 4–5 of 5. Appends to the same CSV as Part 2a. |
| `part3_stage3_6.py` | Stage 3 (corrupted-client sweep), Stage 4 (factorial ablation), Stage 5 (scalability + Sachs real-data experiment), Stage 6 (statistics tables + figures). Loads the CSVs Parts 1–2 already wrote. |
| `independent_validation.py` | Standalone script validating the core PC implementation (CI test, Sepset handling, orientation rules) against the independent `causal-learn` package. Does not depend on and is not required by Parts 1–3. |

The pipeline is split into parts (`1a → 1b → 2a → 2b → 3`) purely so each run
stays under typical cloud-session time limits (~3 hours). Running all parts in
order produces results identical to a single monolithic run.

---

## Requirements

```bash
pip install numpy pandas networkx scipy joblib matplotlib
```

For `independent_validation.py` only, one additional dependency is needed:

```bash
pip install causal-learn
```

Tested with Python 3.10+.

---

## Running the main pipeline

Run the parts in order from the same working directory (each stage reads the
CSV(s) the previous stage wrote):

```bash
python part1a_stage1_first_half.py
python part1b_stage1_second_half.py
python part2a_stage2_first_half.py
python part2b_stage2_second_half.py
python part3_stage3_6.py
```

Outputs are written to `results/`:

```
results/
├── raw/         # per-scenario, per-replicate raw metric CSVs
├── statistics/  # paired significance tests, effect sizes, condition tables
├── tables/      # formatted summary tables
└── figures/     # generated plots
```

`part3_stage3_6.py` additionally runs the real-data experiment on the Sachs
protein-signaling dataset — place the Sachs CSV files (observational +
interventional conditions) in a local `sachs/` directory and point the
script's `sachs_dir` argument at it; if the directory is missing, that part of
Stage 5 is skipped automatically.

All stages seed `numpy`/`random` from a fixed `MASTER_SEED` (see
`fedpc_core.py`) for reproducibility.

---

## Independent validation (`independent_validation.py`)

This script isolates the three components of the from-scratch implementation
that are independent of FedPC's federation/aggregation logic — the
conditional-independence test, skeleton search, Sepset bookkeeping (including
the empty-separating-set / marginal-independence case), and v-structure/Meek
orientation — and checks them against the established `causal-learn` package
(PC algorithm, Fisher-Z test) on freshly simulated linear-Gaussian SEM data.

Place it next to `fedpc_core.py` and run:

```bash
python independent_validation.py
```

It writes:

- `independent_validation_results.csv` — per-configuration numbers (skeleton
  and directed-edge exact match / Jaccard overlap, Sepset exact-match rate,
  empty-Sepset agreement rate)
- `independent_validation_summary.txt` — a human-readable summary

Because the tested configurations are small (≤15 variables, no client
federation), the script runs in well under a minute — this is expected and
not indicative of a reduced or skipped check.

---

## Citation

If you use this code, please cite the associated paper (details to be added
on publication).
