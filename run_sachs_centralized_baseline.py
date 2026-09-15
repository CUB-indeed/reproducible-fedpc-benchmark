
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- import only, no edits to fedpc_core.py -------------------------------
from fedpc_core import (
    CANONICAL_DEFAULT,
    PALETTE,
    METHOD_LABELS,
    _discover_sachs_conditions,
    _load_sachs_files,
    _sachs_ground_truth,
    pc_skeleton_with_sepsets,
    orient_v_structures,
    skeleton_edges_from_graph,
    compute_metrics,
    orientation_metrics_cpdag,
)

PALETTE.setdefault("centralized", "#2c3e50")
METHOD_LABELS.setdefault("centralized", "Centralized")


def find_sachs_dir(user_supplied):
    if user_supplied:
        return user_supplied
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "7681811"),
        "/home/coder/project/7681811",     # same default as part3_stage3_6.py
        os.path.join(here, "..", "7681811"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]  # will trigger a clear "not found" message downstream


def run_centralized_on_condition_set(sachs_dir, filenames, experiment_name, cfg):
    """Mirrors what run_sachs_condition_set() does for 'local'/'naive'/
    'fedpc_consensus', but for a single centralized PC fit on the pooled
    data -- deterministic, so a single run is sufficient (no replicates)."""
    X, col_names = _load_sachs_files(sachs_dir, filenames)
    if X is None or len(col_names) < 2:
        return None
    true_skel, true_dir = _sachs_ground_truth(col_names)
    p = X.shape[1]

    t0 = time.time()
    Gc, seps_c = pc_skeleton_with_sepsets(X, cfg.alpha, cfg.ell)
    pred_skel_c = skeleton_edges_from_graph(Gc)
    cpdag_c = orient_v_structures(Gc, seps_c)
    runtime = time.time() - t0

    shd, prec, rec, f1 = compute_metrics(true_skel, pred_skel_c)
    n_or, dp, dr, df1 = orientation_metrics_cpdag(true_dir, cpdag_c)

    return {
        "experiment_name": experiment_name, "scenario_id": experiment_name,
        "replicate_id": 0, "method": "centralized", "evaluation_unit": "global_graph",
        "evaluation_target": "sachs_ground_truth", "n_vars": p, "n_samples": X.shape[0],
        "K": np.nan, "tau": cfg.tau, "ell": cfg.ell,
        "SHD": shd, "Precision": prec, "Recall": rec, "F1": f1,
        "n_oriented": n_or, "Dir_Prec": dp, "Dir_Rec": dr, "Dir_F1": df1,
        "communication_bytes": 0.0, "runtime_sum": runtime, "runtime_parallel_est": runtime,
        "server_runtime": runtime, "n_clients_timed": 0,
        "note": "single global PC fit, no client partitioning (deterministic, n=1 run)",
    }


def load_existing_sachs_results(out_dir):
    """Best-effort load of what part3_stage3_6.py already produced, so the
    comparison table/figure include local / naive / fedpc_consensus too.
    Returns {} if those files don't exist yet -- script still works."""
    paths = {
        "sachs_observational": f"{out_dir}/raw/sachs_observational_raw.csv",
        "sachs_pooled_all_conditions": f"{out_dir}/raw/sachs_pooled_raw.csv",
    }
    found = {}
    for exp_name, path in paths.items():
        if os.path.exists(path):
            found[exp_name] = pd.read_csv(path)
    return found


def summarize(df):
    """Mean +/- std per method for the metrics we care about, matching the
    style already used throughout the paper's tables/figures."""
    metrics = ["SHD", "F1", "Dir_F1"]
    rows = []
    for method, g in df.groupby("method"):
        row = {"method": method}
        for m in metrics:
            if m in g.columns:
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_std"] = g[m].std() if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_comparison(summaries_by_condition, fig_path):
    n_panels = len(summaries_by_condition)
    metrics = [("SHD", "SHD (lower better)"), ("F1", "Skeleton F1"), ("Dir_F1", "Directional F1")]

    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]
    for ax, (cond_name, summ) in zip(axes, summaries_by_condition.items()):
        method_order = [m for m in ["local", "naive", "fedpc_consensus", "centralized"] if m in summ["method"].values]
        x = np.arange(len(metrics))
        width = 0.8 / max(len(method_order), 1)
        for i, method in enumerate(method_order):
            row = summ[summ["method"] == method].iloc[0]
            means = [row.get(f"{m}_mean", np.nan) for m, _ in metrics]
            stds = [row.get(f"{m}_std", 0.0) for m, _ in metrics]
            ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
                   color=PALETTE.get(method, "#95a5a6"), label=METHOD_LABELS.get(method, method),
                   edgecolor="white", linewidth=0.8)
        ax.set_xticks(x + width * (len(method_order) - 1) / 2)
        ax.set_xticklabels([lbl for _, lbl in metrics])
        ax.set_title(cond_name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, framealpha=0.7)
    fig.suptitle("Sachs real data: centralized PC baseline vs. federated methods", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sachs-dir", default=None, help="Path to the Sachs '7681811' data folder")
    parser.add_argument("--out-dir", default="results", help="Same results/ folder used by part3_stage3_6.py")
    args = parser.parse_args()

    sachs_dir = find_sachs_dir(args.sachs_dir)
    out_dir = args.out_dir
    for sub in ["raw", "tables", "figures"]:
        os.makedirs(f"{out_dir}/{sub}", exist_ok=True)

    if not os.path.isdir(sachs_dir):
        print(f"[ERROR] Sachs data folder not found at: {sachs_dir}")
        print("        Pass it explicitly: --sachs-dir /path/to/7681811")
        sys.exit(1)

    disc = _discover_sachs_conditions(sachs_dir)
    print(f"[Sachs] dir={sachs_dir}")
    print(f"[Sachs] files={len(disc['files'])} observational={disc['observational']} "
          f"interventional_n={len(disc['interventional'])}")

    cfg = CANONICAL_DEFAULT
    centralized_rows = []

    if disc["observational"]:
        r = run_centralized_on_condition_set(sachs_dir, disc["observational"], "sachs_observational", cfg)
        if r:
            centralized_rows.append(r)
            print(f"  [centralized | observational] SHD={r['SHD']:.2f}  F1={r['F1']:.3f}  "
                  f"Dir_F1={r['Dir_F1']:.3f}  runtime={r['runtime_sum']:.2f}s")

    if disc["files"]:
        r = run_centralized_on_condition_set(sachs_dir, disc["files"], "sachs_pooled_all_conditions", cfg)
        if r:
            centralized_rows.append(r)
            print(f"  [centralized | pooled_all_conditions] SHD={r['SHD']:.2f}  F1={r['F1']:.3f}  "
                  f"Dir_F1={r['Dir_F1']:.3f}  runtime={r['runtime_sum']:.2f}s")

    if not centralized_rows:
        print("[ERROR] No usable Sachs condition set found (need at least 'cd3cd28.csv' or other CSVs).")
        sys.exit(1)

    df_centralized = pd.DataFrame(centralized_rows)
    df_centralized.to_csv(f"{out_dir}/raw/sachs_centralized_raw.csv", index=False)
    print(f"\nSaved: {out_dir}/raw/sachs_centralized_raw.csv (does not overwrite any existing file)")

    # ---- merge with existing local/naive/fedpc_consensus results, if present ----
    existing = load_existing_sachs_results(out_dir)
    summaries_by_condition = {}
    merged_tables = []

    for exp_name in ["sachs_observational", "sachs_pooled_all_conditions"]:
        cent_part = df_centralized[df_centralized["experiment_name"] == exp_name]
        if cent_part.empty:
            continue
        if exp_name in existing:
            combined = pd.concat([existing[exp_name], cent_part], ignore_index=True)
            print(f"  Merged centralized results into existing '{exp_name}' results "
                  f"(methods present: {sorted(combined['method'].unique())}).")
        else:
            combined = cent_part
            expected_file = "sachs_observational_raw.csv" if exp_name == "sachs_observational" else "sachs_pooled_raw.csv"
            print(f"  NOTE: results/raw/{expected_file} not found -- only the "
                  f"centralized baseline is available for '{exp_name}'. "
                  f"Run part1a/1b/2a/2b/3 first for the full local/naive/consensus comparison.")
        merged_tables.append(combined)
        summaries_by_condition[exp_name] = summarize(combined)

    if merged_tables:
        full_table = pd.concat(merged_tables, ignore_index=True)
        table_path = f"{out_dir}/tables/sachs_with_centralized_baseline.csv"
        full_table.to_csv(table_path, index=False)
        print(f"Saved merged table: {table_path}")

    if summaries_by_condition:
        fig_path = f"{out_dir}/figures/fig_sachs_centralized_baseline.png"
        plot_comparison(summaries_by_condition, fig_path)
        print(f"Saved figure: {fig_path}")

    print("\nDone. This only added new files under results/raw, results/tables, "
          "results/figures -- nothing existing was modified or re-run.")


if __name__ == "__main__":
    main()
