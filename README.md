# FedPC-Style Benchmark Framework

A reproducible benchmark for aggregation strategies in federated constraint-based causal discovery, using a FedPC-style pipeline under controlled synthetic and real-world (Sachs) data scenarios.

## Overview

Reliable causal discovery in federated environments remains challenging due to privacy constraints, heterogeneous client data, fragmented statistical evidence, and the difficulty of aggregating local causal structures without sharing raw data.

This repository provides a Python-based **FedPC-style benchmarking framework**, scoped as a **controlled aggregation-ablation study within a single PC-based pipeline** — not a general benchmark of state-of-the-art federated causal discovery algorithms, and not a proposal of a new causal discovery algorithm. It isolates, through factorial ablation, how skeleton weighting and separation-set (Sepset) aggregation each contribute to structural and orientation recovery under federated conditions.

The framework combines synthetic DAG generation, federated data partitioning, client-side conditional independence (CI) testing, server-side aggregation of structural summaries, and CPDAG reconstruction using standard v-structure and Meek orientation rules, while raw data remain local at all times.

## Key Features

- Modular FedPC-style constraint-based causal discovery pipeline.
- Synthetic DAG-based data generation using linear-Gaussian SEMs.
- Controlled federated data partitioning.
- Configurable client heterogeneity: homogeneous, mild noise, strong noise, and structural mechanism-shift regimes.
- Client-side conditional independence testing.
- Server-side skeleton and separation-set aggregation, with a ground-truth-free consensus reliability mechanism (pairwise Jaccard similarity).
- CPDAG reconstruction using v-structure identification restricted to unshielded triples, followed by Meek rules R1–R4.
- Factorial ablation isolating skeleton-weighting from Sepset-aggregation effects.
- Real-world validation on the Sachs protein-signaling dataset, including a per-condition breakdown across all 16 experimental conditions.
- Reproducible statistical validation: paired/one-sample Wilcoxon signed-rank tests, matched-pairs rank-biserial correlation, and Holm-corrected significance across 16 scenarios.

## Evaluation Metrics

- Structural Hamming Distance (SHD).
- Skeleton precision, recall, and F1-score.
- Directional precision, recall, and F1-score (computed only on explicitly oriented CPDAG edges; undirected edges are excluded).
- Runtime (simulated per-client CPU time, sequential single-VM execution — not parallel wall-clock latency).
- Communication cost (bytes, based on skeleton indicators and Sepset payload).
- Scalability across federated clients and graph dimensionality.
- Stability under repeated experiments (20 repetitions per configuration).

## Experimental Settings

The benchmark enables systematic evaluation under controlled synthetic settings by varying:

- Number of federated clients (K).
- Sample size (N).
- Graph density (φ) and dimensionality (p).
- Conditioning depth (ℓ_max).
- Distributional heterogeneity, including client-specific structural mechanism-shift.
- Aggregation threshold (τ) and weighting scheme (equal vs. consensus).

## Benchmark Scope

The framework focuses on **benchmarking and component-wise ablation**, not on proposing a new causal discovery algorithm. The implemented baselines are:

- **Centralized PC** — reference upper bound using pooled data.
- **Local PC** — no collaboration between clients.
- **FedPC-Naive** — equal-weight skeleton voting, no Sepset aggregation.
- **FedPC-Consensus** (proposed, practical) — ground-truth-free consensus reliability weighting for skeleton and Sepset aggregation.
- **FedPC-Oracle** — ground-truth-informed weighting, included **solely as a synthetic upper-bound diagnostic reference**, not a deployable method.

A factorial ablation (equal/consensus skeleton weighting × none/equal/consensus Sepset aggregation) shows that orientation improvement over FedPC-Naive is attributable to **Sepset aggregation**, not to the consensus weighting scheme itself; consensus weighting is evaluated as a coarse client-reliability filter (effective at down-weighting corrupted clients) rather than as a source of measurable accuracy gain in the tested configurations.

## Repository Structure

```text
fedpc-benchmark/
├── 7681811/                             # Sachs dataset
├── results/                             # Experimental outputs, raw per-run CSVs, and plots
├── diagrams/
│   ├── fedpc_architecture.drawio
│   ├── fedpc_workflow.png
│   └── fedpc_architecture.png
├── run_and_log.py                       # runtime
└── FedPC-style.py                       # Core implementation of the FedPC-style framework
```


## Reproducibility

All synthetic experiments use fixed random seeds (`SEED = 42`) with 20 repeated runs (`REPS = 20`) per configuration. Sachs experiments use 10 random client-partition seeds per condition. Raw per-run CSVs, aggregated summary tables, and figures are generated directly from the experimental pipeline.

The implementation is a research benchmarking platform, not a production-ready or formally privacy-preserving system: only structural summaries (skeletons and Sepsets) are exchanged between clients and server, and no secure aggregation or differential privacy mechanism is implemented. Client-side computation is currently simulated sequentially on a single machine; reported runtime reflects summed simulated CPU time rather than parallel deployment latency.

## Running Experiments

All experiments should be executed via `run_and_log.py`, which wraps the pipeline 
and logs runtime, configuration, and seed information for reproducibility:

    python run_and_log.py --config <config_name>

Direct execution of `FedPC-style.py` without this wrapper will skip runtime logging 
and is not recommended for reproducing reported results.

## Limitations

- Restricted to linear-Gaussian SEMs under causal sufficiency (horizontal federation only).
- Validated on one real-world dataset (Sachs); generalization to domains with naturally defined client structure is untested.
- Does not include a head-to-head comparison against contemporary federated causal discovery methods (e.g., FedCDH, FedCSL, FedECD), which differ in local discovery procedure and evaluation protocol.

## Author

- **Name:** Widyasmoro Priatmojo
- **Role:** PhD Student
- **Research Interests:** Federated Learning, Causal AI, Distributed Causal Discovery, Reproducible Machine Learning, Sustainability
