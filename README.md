# FedPC — Federated PC Algorithm for Constraint-Based Causal Discovery

Simulation and validation codebase for **FedPC**, a federated adaptation of the PC algorithm that aggregates per-client separating sets (Sepsets) under weighted voting, rather than aggregating adjacency votes alone.

**Author:** Widyasmoro Priatmojo

---

## Contents

| File                                | Purpose                                                                                                                                                                                                    |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fedpc_core.py`                     | Core algorithm: CI test, skeleton search, Sepset aggregation, v-structure/Meek orientation, experiment runners, metrics, and figure generation. Everything else imports from this file.                    |
| `part1a_stage1_first_half.py`       | Stage 1 (main sweep), scenarios 1–5 of 10. Writes `results/raw/main_sweep_raw.csv`.                                                                                                                        |
| `part1b_stage1_second_half.py`      | Stage 1 (main sweep), scenarios 6–10 of 10. Appends to the same CSV as Part 1a.                                                                                                                            |
| `part2a_stage2_first_half.py`       | Stage 2 (mechanism-shift sweep), scenarios 1–3 of 5. Writes `results/raw/mechshift_sweep_raw.csv`.                                                                                                         |
| `part2b_stage2_second_half.py`      | Stage 2 (mechanism-shift sweep), scenarios 4–5 of 5. Appends to the same CSV as Part 2a.                                                                                                                   |
| `part3_stage3_6.py`                 | Stage 3 (corrupted-client sweep), Stage 4 (factorial ablation), Stage 5 (scalability + Sachs real-data experiment), Stage 6 (statistics tables + figures). Loads the CSVs Parts 1–2 already wrote.         |
| `run_sachs_centralized_baseline.py` | Runs a centralized PC baseline on the Sachs protein-signaling dataset and optionally merges the result with the local, naive, and FedPC consensus results produced by `part3_stage3_6.py`.                 |
| `independent_validation.py`         | Standalone script validating the core PC implementation (CI test, Sepset handling, orientation rules) against the independent `causal-learn` package. Does not depend on and is not required by Parts 1–3. |

The main simulation pipeline is split into parts (`1a → 1b → 2a → 2b → 3`) purely so each run stays under typical cloud-session time limits (~3 hours). Running all parts in order produces results identical to a single monolithic run.

The Sachs centralized baseline is a separate analysis script. It does **not** need to be run to complete the main simulation pipeline.

---

## Requirements

Install the main dependencies with:

```bash
pip install numpy pandas networkx scipy joblib matplotlib
```

For `independent_validation.py`, one additional dependency is needed:

```bash
pip install causal-learn
```

Tested with Python 3.10+.

---

## Running the main pipeline

Run the parts in order from the same working directory:

```bash
python part1a_stage1_first_half.py
python part1b_stage1_second_half.py
python part2a_stage2_first_half.py
python part2b_stage2_second_half.py
python part3_stage3_6.py
```

Outputs are written to `results/`:

```text
results/
├── raw/         # per-scenario, per-replicate raw metric CSVs
├── statistics/  # paired significance tests, effect sizes, condition tables
├── tables/      # formatted summary tables
└── figures/     # generated plots
```

All stages seed `numpy`/`random` from a fixed `MASTER_SEED` defined in `fedpc_core.py` for reproducibility.

---

## Sachs real-data experiment

`part3_stage3_6.py` includes the Sachs protein-signaling real-data experiment as part of Stage 5.

The Sachs experiment expects the dataset files to be placed in a local directory containing the Sachs condition CSV files. The directory is commonly named `7681811` in this codebase.

For example:

```text
project/
├── fedpc_core.py
├── part3_stage3_6.py
├── run_sachs_centralized_baseline.py
├── ...
└── 7681811/
    ├── cd3cd28.csv
    ├── ...
    └── other Sachs condition files
```

The exact set of files is discovered automatically by the Sachs helper functions in `fedpc_core.py`.

If the Sachs directory is missing, the Sachs portion of Stage 5 in `part3_stage3_6.py` is skipped automatically.

### Running the Sachs experiment through Stage 5

If the Sachs data directory is available, run the normal pipeline:

```bash
python part3_stage3_6.py
```

The script will automatically discover the observational and interventional Sachs condition files and run the corresponding real-data experiments.

---

## Centralized Sachs baseline (`run_sachs_centralized_baseline.py`)

`run_sachs_centralized_baseline.py` provides a **centralized PC baseline** for the Sachs real-data experiment.

Unlike the federated experiments, the centralized baseline:

* pools the available Sachs data into a single dataset,
* runs one global PC skeleton search,
* stores the resulting Sepsets locally,
* orients v-structures using the resulting graph and Sepsets,
* does not partition the data into clients,
* does not perform federated aggregation,
* does not incur communication cost,
* uses the same canonical PC configuration as the main FedPC experiments.

Because the centralized PC fit is deterministic under the fixed configuration, only a single run (`replicate_id = 0`) is performed.

The script evaluates two Sachs condition sets when available:

1. **Observational data only**
2. **All discovered Sachs conditions pooled together**

The resulting centralized metrics can then be compared with the `local`, `naive`, and `fedpc_consensus` methods generated by `part3_stage3_6.py`.

### Running the centralized baseline

From the project root:

```bash
python run_sachs_centralized_baseline.py
```

The script automatically searches for the Sachs directory in the following locations:

```text
./7681811
/home/coder/project/7681811
../7681811
```

If the dataset is stored somewhere else, provide the path explicitly:

```bash
python run_sachs_centralized_baseline.py --sachs-dir /path/to/7681811
```

For example:

```bash
python run_sachs_centralized_baseline.py --sachs-dir /data/sachs/7681811
```

The output directory defaults to the same `results/` directory used by the main pipeline.

To use a different output directory:

```bash
python run_sachs_centralized_baseline.py \
    --sachs-dir /path/to/7681811 \
    --out-dir /path/to/results
```

### Recommended order

For a complete comparison between the centralized and federated methods, first run the main pipeline:

```bash
python part1a_stage1_first_half.py
python part1b_stage1_second_half.py
python part2a_stage2_first_half.py
python part2b_stage2_second_half.py
python part3_stage3_6.py
```

Then run:

```bash
python run_sachs_centralized_baseline.py
```

This allows the centralized baseline script to find and merge the existing Sachs results from `part3_stage3_6.py`.

If those existing result files are not available, the script still runs successfully, but the comparison table and figure will contain only the centralized baseline for the corresponding condition.

### Input discovery

The script uses the Sachs discovery and loading functions already implemented in `fedpc_core.py`:

```text
_discover_sachs_conditions()
_load_sachs_files()
_sachs_ground_truth()
```

Therefore, the centralized baseline does not implement a separate Sachs file parser or ground-truth definition. It uses the same Sachs data handling and ground truth as the main Sachs experiment.

At least one usable Sachs condition set must be available. If no usable Sachs CSV files are found, the script terminates with an error message.

### Outputs

The centralized baseline creates the following files under `results/`:

```text
results/
├── raw/
│   └── sachs_centralized_raw.csv
│
├── tables/
│   └── sachs_with_centralized_baseline.csv
│
└── figures/
    └── fig_sachs_centralized_baseline.png
```

#### `sachs_centralized_raw.csv`

Contains the raw metrics from the centralized PC runs, including:

* SHD
* Skeleton Precision
* Skeleton Recall
* Skeleton F1
* Number of oriented edges
* Directional Precision
* Directional Recall
* Directional F1
* Runtime
* Communication cost

The centralized baseline has:

```text
method = centralized
K = NaN
communication_bytes = 0
replicate_id = 0
```

because there is no client partitioning or federated communication.

#### `sachs_with_centralized_baseline.csv`

If the corresponding Sachs results from `part3_stage3_6.py` already exist, the script combines them with the centralized baseline.

The resulting table can contain:

```text
local
naive
fedpc_consensus
centralized
```

for each available Sachs condition.

The summary reports the mean and standard deviation for:

```text
SHD
F1
Dir_F1
```

#### `fig_sachs_centralized_baseline.png`

The script generates a comparison figure showing:

* SHD, where lower is better
* Skeleton F1
* Directional F1

for the available methods and Sachs condition sets.

The figure is saved at 300 DPI.

### Important: non-destructive behavior

`run_sachs_centralized_baseline.py` is designed to be **non-destructive**.

It writes:

```text
results/raw/sachs_centralized_raw.csv
results/tables/sachs_with_centralized_baseline.csv
results/figures/fig_sachs_centralized_baseline.png
```

It does not overwrite or re-run the existing federated Sachs experiments.

If `sachs_observational_raw.csv` and/or `sachs_pooled_raw.csv` already exist, they are read only for comparison and are not modified.

---

## Independent validation (`independent_validation.py`)

This script isolates the three components of the from-scratch implementation that are independent of FedPC's federation/aggregation logic:

1. Conditional-independence testing
2. Skeleton search and Sepset bookkeeping
3. V-structure and Meek orientation

It checks these components against the established `causal-learn` package (PC algorithm and Fisher-Z test) on freshly simulated linear-Gaussian SEM data.

Place it next to `fedpc_core.py` and run:

```bash
python independent_validation.py
```

It writes:

```text
independent_validation_results.csv
independent_validation_summary.txt
```

The results include:

* skeleton exact match / Jaccard overlap,
* directed-edge exact match / Jaccard overlap,
* Sepset exact-match rate,
* empty-Sepset agreement rate.

Because the tested configurations are small (≤15 variables, with no client federation), the script runs in well under a minute. This short runtime is expected and does not indicate that the validation is reduced or skipped.

---

## Relationship between the validation and Sachs baseline

The two standalone scripts serve different purposes.

### `independent_validation.py`

Validates the **implementation of the core PC components** against an independent implementation:

```text
FedPC implementation
        │
        ├── CI test
        ├── skeleton search
        ├── Sepsets
        └── orientation
                 │
                 ▼
          causal-learn PC
```

It uses simulated linear-Gaussian data and does not use the Sachs dataset.

### `run_sachs_centralized_baseline.py`

Provides a **real-data centralized reference point** for the Sachs experiment:

```text
Sachs data
    │
    ├── Centralized PC
    │       └── global graph
    │
    └── FedPC / federated methods
            ├── local
            ├── naive
            └── fedpc_consensus
```

Thus, the centralized Sachs baseline is not an independent software-validation test. It is a baseline for evaluating the federated methods on the same real-world dataset.

---

## Reproducibility

All main experiments use fixed random seeds controlled by `MASTER_SEED` in `fedpc_core.py`.

The centralized Sachs baseline itself is deterministic and performs a single global PC fit, so no replicate sweep is required.

For the most complete and reproducible set of results, run the scripts in this order:

```bash
python part1a_stage1_first_half.py
python part1b_stage1_second_half.py
python part2a_stage2_first_half.py
python part2b_stage2_second_half.py
python part3_stage3_6.py
python run_sachs_centralized_baseline.py
```

The independent validation can be run separately:

```bash
python independent_validation.py
```

It does not depend on the results generated by Parts 1–3.

---

## Expected project structure

After running the complete pipeline and the standalone analyses, the project may look like:

```text
project/
├── fedpc_core.py
├── part1a_stage1_first_half.py
├── part1b_stage1_second_half.py
├── part2a_stage2_first_half.py
├── part2b_stage2_second_half.py
├── part3_stage3_6.py
├── run_sachs_centralized_baseline.py
├── independent_validation.py
│
├── 7681811/
│   ├── cd3cd28.csv
│   └── ...
│
├── results/
│   ├── raw/
│   │   ├── main_sweep_raw.csv
│   │   ├── mechshift_sweep_raw.csv
│   │   ├── sachs_observational_raw.csv
│   │   ├── sachs_pooled_raw.csv
│   │   └── sachs_centralized_raw.csv
│   │
│   ├── statistics/
│   │   └── ...
│   │
│   ├── tables/
│   │   ├── ...
│   │   └── sachs_with_centralized_baseline.csv
│   │
│   └── figures/
│       ├── ...
│       └── fig_sachs_centralized_baseline.png
│
├── independent_validation_results.csv
└── independent_validation_summary.txt
```

---

## Citation

If you use this code, please cite the associated paper (details to be added on publication).
