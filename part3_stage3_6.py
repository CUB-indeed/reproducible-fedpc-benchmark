"""
PART 3 / 5 -- Run this after part1a_stage1_first_half.py, part1b_stage1_second_half.py, part2a_stage2_first_half.py, part2b_stage2_second_half.py has finished and produced
results/raw/main_sweep_raw.csv and results/raw/mechshift_sweep_raw.csv.

Runs Stage 3 (corrupted-client sweep), Stage 4 (factorial ablation),
Stage 5 (scalability + Sachs real-data experiment), and Stage 6
(statistics tables + figures) -- exactly as in the original monolithic
script's `if __name__ == "__main__":` block. No logic changed: df_main
and df_mech are simply loaded from the CSVs Part 1 already wrote,
instead of being recomputed, and main_registry/mech_registry are
rebuilt (they are pure, deterministic functions of no random state, so
rebuilding them is identical to what Part 1 already had in memory).
"""
import os
import time
import random
import numpy as np
import pandas as pd

from fedpc_core import (
    MASTER_SEED,
    build_main_scenario_registry,
    build_mechshift_registry,
    run_corruption_sweep,
    run_factorial_ablation,
    run_scalability,
    run_sachs,
    build_condition_table,
    generate_figures,
)

if __name__ == "__main__":
    np.random.seed(MASTER_SEED)
    random.seed(MASTER_SEED)

    # Same replicate counts as the original single-file script.
    REPS_CORRUPTION = 15
    REPS_FACTORIAL = 20
    REPS_SCALE = 3
    REPS_SACHS = 10
    SACHS_DIR = "/home/coder/project/7681811"  # adjust to your environment

    OUT_DIR = "results"
    for sub in ["raw", "figures", "tables", "statistics"]:
        os.makedirs(f"{OUT_DIR}/{sub}", exist_ok=True)

    # ---- Load Part 1's outputs instead of recomputing them ----
    main_csv = f"{OUT_DIR}/raw/main_sweep_raw.csv"
    mech_csv = f"{OUT_DIR}/raw/mechshift_sweep_raw.csv"
    if not (os.path.exists(main_csv) and os.path.exists(mech_csv)):
        raise FileNotFoundError(
            f"Expected {main_csv} and {mech_csv} from part1a_stage1_first_half.py, part1b_stage1_second_half.py, part2a_stage2_first_half.py, part2b_stage2_second_half.py. "
            "Run part1a_stage1_first_half.py, part1b_stage1_second_half.py, part2a_stage2_first_half.py, part2b_stage2_second_half.py first."
        )
    df_main = pd.read_csv(main_csv)
    df_mech = pd.read_csv(mech_csv)
    # Deterministic, side-effect-free -- identical to what part1 built in memory.
    main_registry = build_main_scenario_registry()
    mech_registry = build_mechshift_registry()

    t_start = time.time()

    print("=" * 70)
    print("STAGE 3/6: Corrupted-client sweep (with the equal-vs-consensus control arm)")
    print("=" * 70)
    df_corr, corr_registry = run_corruption_sweep(REPS_CORRUPTION, OUT_DIR)

    print("\n" + "=" * 70)
    print("STAGE 4/6: Factorial ablation (Sepset-aggregation x weighting)")
    print("=" * 70)
    df_fact, fact_scen_id = run_factorial_ablation(REPS_FACTORIAL, OUT_DIR)

    print("\n" + "=" * 70)
    print("STAGE 5/6: Scalability (p in {20, 30, 50}) + Sachs real-data experiment")
    print("=" * 70)
    df_scale = run_scalability(REPS_SCALE, OUT_DIR)
    sachs_out = run_sachs(SACHS_DIR, REPS_SACHS, OUT_DIR)

    print("\n" + "=" * 70)
    print("STAGE 6/6: Statistics tables + figures")
    print("=" * 70)

    metrics_std = ["SHD", "F1", "Dir_F1", "n_oriented", "communication_bytes",
                   "runtime_sum", "runtime_parallel_est"]

    # Main sweep: consensus vs naive, per scenario (directional metrics excluded
    # automatically for methods in STRUCTURAL_ZERO_METHODS).
    stat_main = build_condition_table(df_main, "scenario_id", "naive", "fedpc_consensus", metrics_std)
    stat_main.to_csv(f"{OUT_DIR}/statistics/main_sweep_naive_vs_consensus.csv", index=False)

    # Mechanism-shift: consensus vs naive, per deletion_probability.
    df_mech2 = df_mech.merge(mech_registry[["scenario_id", "deletion_probability"]].drop_duplicates(),
                              on="scenario_id", suffixes=("", "_reg"))
    stat_mech = build_condition_table(df_mech, "scenario_id", "naive", "fedpc_consensus", metrics_std)
    stat_mech.to_csv(f"{OUT_DIR}/statistics/mechshift_naive_vs_consensus.csv", index=False)

    # Corruption: THE key control-arm comparison -- agg_equal vs agg_consensus,
    # BOTH with Sepset aggregation on, per corruption_fraction.
    stat_corr = build_condition_table(df_corr, "corruption_fraction", "agg_equal", "agg_consensus", metrics_std)
    stat_corr.to_csv(f"{OUT_DIR}/statistics/corruption_equal_vs_consensus.csv", index=False)

    # Factorial: isolates Sepset-aggregation effect from weighting effect.
    df_fact_c = df_fact.copy(); df_fact_c["dummy_condition"] = "canonical_default"
    stat_fact_1 = build_condition_table(df_fact_c, "dummy_condition", "no_agg_equal", "agg_equal", metrics_std)
    stat_fact_2 = build_condition_table(df_fact_c, "dummy_condition", "agg_equal", "agg_consensus", metrics_std)
    stat_fact_1.to_csv(f"{OUT_DIR}/statistics/factorial_noagg_vs_aggequal.csv", index=False)
    stat_fact_2.to_csv(f"{OUT_DIR}/statistics/factorial_aggequal_vs_aggconsensus.csv", index=False)

    generate_figures(df_main, df_mech, df_corr, df_fact, fact_scen_id, df_scale, sachs_out, OUT_DIR)

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"PART 3/5 DONE in {elapsed/60:.1f} minutes.")
    print(f"Raw results:      {OUT_DIR}/raw/")
    print(f"Statistics:       {OUT_DIR}/statistics/")
    print(f"Figures:          {OUT_DIR}/figures/")
    print("=" * 70)

    print("\nKey answer to Reviewer #3, point 3 (corruption control arm):")
    print(stat_corr[["corruption_fraction", "F1_mean_diff", "F1_cohens_d", "F1_p_value", "F1_p_value_bh",
                      "n_replicates"]].to_string(index=False))

    print("\nKey answer to Reviewer #3, point 1 (Sepset-aggregation vs weighting):")
    print("  no_agg_equal -> agg_equal (Sepset-aggregation effect):")
    print(stat_fact_1[[c for c in stat_fact_1.columns if "Dir_F1" in c or c == "dummy_condition"]].to_string(index=False))
    print("  agg_equal -> agg_consensus (weighting effect, Sepset-agg held fixed=on):")
    print(stat_fact_2[[c for c in stat_fact_2.columns if "Dir_F1" in c or c == "dummy_condition"]].to_string(index=False))
