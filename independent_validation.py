"""
Independent validation of FedPC's from-scratch causal-discovery implementation
against the established `causal-learn` package (PC algorithm, Fisher-Z CI test).

Addresses reviewer comment:
  "Independent validation of the implementation remains outstanding and is,
   in my view, the decisive item. The authors state candidly that the
   from-scratch CI test, Sepset handling, and orientation rules have not
   been checked against an established causal discovery package... the
   ambiguity above regarding empty separating sets is exactly the kind of
   defect such a check would surface."

This script does NOT touch FedPC's federation/aggregation layer at all --
it isolates and independently checks exactly the three components the
reviewer named:
  1. CI test + skeleton search       -> pc_skeleton_with_sepsets()
  2. Sepset handling (incl. empty
     separating sets / marginal
     independence)                   -> get_sepset() / stored Sepsets
  3. Orientation rules (v-structures
     + Meek)                         -> orient_v_structures()

against causal-learn's PC() reference implementation, on freshly simulated
data the FedPC codebase was never fit to.

Requires (only new dependency vs. the rest of the FedPC codebase):
    pip install causal-learn

Place this file in the same folder as fedpc_core.py, then run:
    python independent_validation.py

Outputs (written next to this script):
    independent_validation_results.csv   -- per-configuration numbers
    independent_validation_summary.txt   -- human-readable summary
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from fedpc_core import (
    pc_skeleton_with_sepsets,
    orient_v_structures,
    skeleton_edges_from_graph,
    directed_edges_from_cpdag,
    get_sepset,
    generate_random_dag,
    simulate_linear_sem,
)

from causallearn.search.ConstraintBased.PC import pc as cl_pc
from causallearn.utils.cit import fisherz

ALPHA = 0.05
ELL_MAX = 2


def causallearn_skeleton_and_dir(cg, p):
    gm = cg.G.graph
    skel, dir_edges = set(), set()
    for i in range(p):
        for j in range(i + 1, p):
            if gm[i, j] != 0 or gm[j, i] != 0:
                skel.add((i, j))
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if gm[j, i] == 1 and gm[i, j] == -1:
                dir_edges.add((i, j))
    return skel, dir_edges


def causallearn_sepset(cg, i, j):
    """causal-learn stores cg.sepset[i, j] as an array of candidate
    separating sets found during the search; take their union so this is
    comparable to FedPC's single stored Sepset per removed pair."""
    entries = cg.sepset[i, j]
    if entries is None:
        return None
    found = [e for e in entries if e is not None]
    if not found:
        return None
    union = set()
    for e in found:
        union |= set(e)
    return union


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def run_case(p, edge_prob, n, seed, tag=""):
    rng = np.random.default_rng(seed)
    B = generate_random_dag(p, edge_prob, rng)
    X = simulate_linear_sem(B, n, rng)

    # --- FedPC (ours), single-site / centralized use of the core routines ---
    G_ours, seps_ours = pc_skeleton_with_sepsets(X, alpha=ALPHA, ell_max=ELL_MAX)
    skel_ours = skeleton_edges_from_graph(G_ours)
    cpdag_ours = orient_v_structures(G_ours, seps_ours)
    dir_ours = directed_edges_from_cpdag(cpdag_ours)

    # --- causal-learn (independent reference implementation) ---
    cg = cl_pc(X, alpha=ALPHA, indep_test=fisherz, stable=True, uc_rule=0, uc_priority=2,
               show_progress=False, verbose=False)
    skel_cl, dir_cl = causallearn_skeleton_and_dir(cg, p)

    skel_jac = jaccard(skel_ours, skel_cl)
    dir_jac = jaccard(dir_ours, dir_cl)

    # --- Sepset cross-check, restricted to edges BOTH implementations remove
    #     (an apples-to-apples comparison of *why* an edge was removed) ---
    removed_both = {(i, j) for i in range(p) for j in range(i + 1, p)
                     if (i, j) not in skel_ours and (i, j) not in skel_cl}

    sep_match = sep_total = empty_cases = empty_match = 0
    for (i, j) in removed_both:
        s_ours = get_sepset(seps_ours, i, j)
        s_cl = causallearn_sepset(cg, i, j)
        if s_ours is None or s_cl is None:
            continue
        sep_total += 1
        if s_ours == s_cl:
            sep_match += 1
        if len(s_ours) == 0 or len(s_cl) == 0:
            empty_cases += 1
            if s_ours == s_cl:
                empty_match += 1

    return dict(
        tag=tag, p=p, edge_prob=edge_prob, n=n, seed=seed,
        skel_exact=(skel_ours == skel_cl), skel_jaccard=skel_jac,
        skel_ours_size=len(skel_ours), skel_cl_size=len(skel_cl),
        dir_exact=(dir_ours == dir_cl), dir_jaccard=dir_jac,
        dir_ours_size=len(dir_ours), dir_cl_size=len(dir_cl),
        n_removed_both=len(removed_both),
        sepset_exact_match=sep_match, sepset_total_compared=sep_total,
        n_empty_sepset_cases=empty_cases, empty_sepset_agree=empty_match,
    )


def main():
    configs = []
    # Main sweep, same regime as the paper's own simulations (Gaussian linear SEM).
    for p, edge_prob in [(6, 0.3), (8, 0.25), (10, 0.2), (12, 0.15), (15, 0.12)]:
        for n in [1000, 3000]:
            for seed in range(1, 11):
                configs.append((p, edge_prob, n, seed, "main_sweep"))

    # Sparse stress test: low edge density forces many marginal-independence
    # (empty separating set) removals -- exactly the case the reviewer flagged.
    for p in [8, 10, 12]:
        for seed in range(1, 11):
            configs.append((p, 0.06, 3000, seed, "sparse_empty_sepset_stress"))

    results = [run_case(*c) for c in configs]
    df = pd.DataFrame(results)
    df.to_csv(Path(__file__).resolve().parent / "independent_validation_results.csv", index=False)

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=== Independent validation of FedPC's from-scratch PC implementation ===")
    log("Reference: causal-learn PC algorithm, Fisher-Z CI test, stable=True\n")

    for tag, sub in df.groupby("tag"):
        log(f"--- {tag}  (n={len(sub)} configs) ---")
        log(f"Skeleton exact match:      {sub['skel_exact'].sum()}/{len(sub)}  "
            f"(mean Jaccard={sub['skel_jaccard'].mean():.4f})")
        log(f"Directed-edge exact match: {sub['dir_exact'].sum()}/{len(sub)}  "
            f"(mean Jaccard={sub['dir_jaccard'].mean():.4f})")
        sep_sub = sub[sub["sepset_total_compared"] > 0]
        if len(sep_sub):
            tot_match = sep_sub["sepset_exact_match"].sum()
            tot_cmp = sep_sub["sepset_total_compared"].sum()
            log(f"Sepset exact-match rate (edges both remove): {tot_match}/{tot_cmp} "
                f"({tot_match / tot_cmp:.4f})")
        n_empty = sub["n_empty_sepset_cases"].sum()
        if n_empty:
            n_empty_ok = sub["empty_sepset_agree"].sum()
            log(f"Empty-Sepset (marginal-independence) agreement: {n_empty_ok}/{n_empty} "
                f"({n_empty_ok / n_empty:.4f})")
        log()

    summary_path = Path(__file__).resolve().parent / "independent_validation_summary.txt"
    summary_path.write_text("\n".join(lines))
    print(f"Saved: independent_validation_results.csv, {summary_path.name}")


if __name__ == "__main__":
    main()
