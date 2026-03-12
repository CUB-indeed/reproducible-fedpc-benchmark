# FedPC Benchmark
A reproducible benchmarking framework for evaluating the **FedPC algorithm** for federated constraint-based causal discovery under controlled synthetic data scenarios.

## Overview
Reliable causal discovery in federated environments remains challenging due to privacy constraints, heterogeneous client data, and the difficulty of aggregating local causal structures across distributed institutions.
This repository introduces **FedPC**, a Python-based framework that integrates federated causal discovery with a systematic synthetic benchmarking environment. The framework combines DAG-based data generation, controlled heterogeneity simulation, client-level conditional independence testing, and server-side aggregation of causal structures.
The goal of this project is to provide a **reproducible evaluation framework** for studying how federated causal discovery behaves under different experimental conditions.

## Key Features
- Implementation of **FedPC (Federated PC Algorithm)**
- Synthetic **DAG-based data generation**
- Controlled **client heterogeneity simulation**
- Client-side **conditional independence testing**
- Server-side **skeleton and edge orientation aggregation**
- Reproducible **benchmark evaluation pipeline**
The framework evaluates multiple performance aspects including:
- Structural Hamming Distance (SHD)
- Skeleton F1-score
- Orientation F1-score
- Communication efficiency
- Scalability across clients

## Experimental Settings
The benchmark evaluates FedPC under varying experimental conditions:
- Number of clients
- Graph density
- Conditioning depth
- Sample size per client
- Data heterogeneity levels
These controlled settings allow systematic analysis of how federated constraints affect causal structure recovery.

## Results Summary
Empirical experiments show that:
- Skeleton F1 ranges from **0.64 – 0.91** under moderate federation
- Orientation F1 ranges from **0.30 – 0.58**
- FedPC outperforms naive aggregation approaches
- In several settings it surpasses centralized PC in **Structural Hamming Distance (SHD)**

Performance degradation occurs predictably under:
- smaller per-client sample sizes
- stronger data heterogeneity
- denser causal graphs
Communication cost scales approximately **linearly with the number of clients and conditioning depth**.
Weighted sepset aggregation and reliability-based voting improve orientation accuracy by **0.02 – 0.08** compared to standard majority voting.

## Repository Structure
```
fedpc-benchmark/
│
├── src/                # Core implementation of FedPC
├── experiments/        # Experiment scripts
├── data/               # Synthetic data generation
├── results/            # Plots and experiment outputs
├── requirements.txt    # Python dependencies
└── README.md
```

## Reproducibility
This repository aims to provide a **fully reproducible benchmark** for federated causal discovery research. All experiments are designed to run with controlled synthetic settings to ensure consistent evaluation across different configurations.
