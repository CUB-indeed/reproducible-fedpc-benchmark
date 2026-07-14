# FedPC-Style Benchmark Framework

A reproducible benchmarking framework for evaluating federated constraint-based causal discovery using a FedPC-style pipeline under controlled synthetic data scenarios.

## Overview

Reliable causal discovery in federated environments remains challenging due to privacy constraints, heterogeneous client data, fragmented statistical evidence, and the difficulty of aggregating local causal structures without sharing raw data.

This repository provides a Python-based **FedPC-style benchmarking framework** for systematic evaluation of federated causal discovery. Rather than proposing a fundamentally new causal discovery algorithm, the framework offers a modular and reproducible experimental pipeline for studying how different components of a PC-based federated workflow contribute to structural recovery.

The framework combines synthetic DAG generation, federated data partitioning, client-side conditional independence (CI) testing, server-side aggregation of structural summaries, and CPDAG reconstruction using standard orientation rules. It enables controlled benchmarking under varying federation settings while preserving the assumption that raw data remain local.

## Key Features

- Modular FedPC-style constraint-based causal discovery pipeline.
- Synthetic DAG-based data generation using linear-Gaussian SEMs.
- Controlled federated data partitioning.
- Configurable client heterogeneity simulation.
- Variance-based perturbations.
- Optional mechanism-shift scenarios.
- Client-side conditional independence testing.
- Server-side skeleton and separation-set aggregation.
- CPDAG reconstruction using v-structure identification and Meek orientation rules.
- Reproducible benchmark evaluation framework.

## Evaluation Metrics

The framework supports evaluation of multiple aspects including:

- Structural Hamming Distance (SHD).
- Skeleton precision, recall, and F1-score.
- Orientation precision, recall, and F1-score.
- Runtime.
- Communication cost.
- Scalability across federated clients.
- Stability under repeated experiments.

## Experimental Settings

The benchmark enables systematic evaluation under controlled synthetic settings by varying:

- Number of federated clients.
- Sample size per client.
- Graph density.
- Graph topology.
- Conditioning depth.
- Distributional heterogeneity.
- Aggregation strategy.

These controlled configurations make it possible to analyze how decentralized data, sample fragmentation, and client heterogeneity influence different stages of the causal discovery pipeline.

## Benchmark Scope

The framework focuses on benchmarking and component-wise analysis rather than proposing a new causal discovery algorithm.

The implemented pipeline includes:

- Local PC-style skeleton estimation.
- Separation-set aggregation.
- Consensus-based structural aggregation.
- CPDAG reconstruction.
- Oracle-weighted aggregation (synthetic upper-bound reference only).

The oracle weighting strategy is included solely for controlled synthetic experiments to estimate the theoretical upper bound of aggregation performance. It is not intended for deployment in practical federated environments, where ground-truth causal graphs are unavailable.

## Repository Structure

```text
fedpc-benchmark/
│
├── src/                # Core implementation of the FedPC-style framework
├── experiments/        # Benchmark experiment scripts
├── data/               # Synthetic data generation
├── results/            # Experimental outputs and plots
├── requirements.txt    # Python dependencies
└── README.md
```

## Reproducibility

This repository is designed to provide a fully reproducible benchmark for federated constraint-based causal discovery.

The framework uses controlled synthetic linear-Gaussian SEMs, configurable federation settings, and modular evaluation components to enable consistent comparison across different aggregation strategies and experimental configurations.

The implementation is intended as a research benchmarking platform and does not include production-oriented privacy mechanisms such as secure aggregation or differential privacy.

## Author

- **Name:** Widyasmoro Priatmojo
- **Role:** PhD Student
- **Research Interests:** Federated Learning, Causal AI, Distributed Causal Discovery, Reproducible Machine Learning, Sustainability
