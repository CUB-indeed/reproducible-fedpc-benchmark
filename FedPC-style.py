import numpy as np
import networkx as nx
from itertools import combinations
from scipy.stats import pearsonr, norm, wilcoxon, t as t_dist
import random
import time
import os
import pandas as pd
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# =======================
# GLOBAL SETTINGS
# =======================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

REPS = 20
REPS_SCALE = 3
ELL_MAX = 2
ALPHA = 0.05
TAU_DEFAULT = 0.5
SACHS_DIR = "/home/coder/project/7681811"

METHODS = ["centralized", "fedpc_consensus", "fedpc_oracle", "naive"]

# =======================
# PLOT STYLE CONSTANTS
# =======================
PALETTE = {
    "centralized":     "#2c3e50",
    "local":           "#7f8c8d",
    "naive":           "#e67e22",
    "fedpc_consensus": "#27ae60",
    "fedpc_oracle":    "#3498db",
}
METHOD_LABELS = {
    "centralized":     "Centralized",
    "local":           "Local PC",
    "naive":           "FedPC-Naive",
    "fedpc_consensus": "FedPC-Consensus",
    "fedpc_oracle":    "FedPC-Oracle",
}
PLOT_METHODS = ["centralized", "local", "naive", "fedpc_consensus", "fedpc_oracle"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})


# =======================
# CORE FUNCTIONS
# =======================

def safe_pearsonr(x, y):
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0, 1.0
    r, p = pearsonr(x, y)
    if not np.isfinite(r):
        return 0.0, 1.0
    return float(np.clip(r, -0.999999, 0.999999)), p


def generate_random_dag(p, edge_prob):
    G = nx.DiGraph()
    G.add_nodes_from(range(p))
    B = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            if np.random.rand() < edge_prob:
                G.add_edge(i, j)
                B[i, j] = np.random.uniform(0.5, 2.0) * random.choice([-1, 1])
    return B


def simulate_linear_sem(B, n_samples, hetero=None, B_perturbed=None):
    Buse = B_perturbed if B_perturbed is not None else B
    p = Buse.shape[0]
    X = np.zeros((n_samples, p))
    # Build DiGraph from Buse for topological ordering
    G_temp = nx.DiGraph()
    G_temp.add_nodes_from(range(p))
    for i in range(p):
        for j in range(p):
            if Buse[i, j] != 0:
                G_temp.add_edge(i, j)
    order = list(nx.topological_sort(G_temp))
    for i in order:
        parents = np.where(Buse[:, i] != 0)[0]
        noise = np.random.normal(0, 1, n_samples)
        if hetero == "mild":
            # Sample-wise scaling: eta_t ~ U(0.8, 1.2)
            eta = np.random.uniform(0.8, 1.2, n_samples)
            noise = noise * eta
        elif hetero == "strong":
            # Sample-wise scaling: eta_t ~ U(0.5, 1.5)
            eta = np.random.uniform(0.5, 1.5, n_samples)
            noise = noise * eta
        if len(parents):
            X[:, i] = X[:, parents] @ Buse[parents, i] + noise
        else:
            X[:, i] = noise
    return X


def perturb_B(B, drop_frac=0.07, coeff_noise=0.2):
    Bp = B.copy()
    edges = list(zip(*np.where(B != 0)))
    if not edges:
        return Bp
    n_drop = max(1, int(len(edges) * drop_frac))
    drop_idx = random.sample(edges, min(n_drop, len(edges)))
    for i, j in drop_idx:
        Bp[i, j] = 0.0
    for i, j in zip(*np.where(Bp != 0)):
        Bp[i, j] *= np.random.uniform(1 - coeff_noise, 1 + coeff_noise)
    return Bp


def ci_test(X, i, j, cond_set, alpha=0.05):
    n = X.shape[0]
    k = len(cond_set)
    if n - k - 3 <= 0:
        return False
    if k == 0:
        r, _ = safe_pearsonr(X[:, i], X[:, j])
    else:
        Z = X[:, list(cond_set)]
        beta_i, *_ = np.linalg.lstsq(Z, X[:, i], rcond=None)
        beta_j, *_ = np.linalg.lstsq(Z, X[:, j], rcond=None)
        r, _ = safe_pearsonr(X[:, i] - Z @ beta_i, X[:, j] - Z @ beta_j)
    z = 0.5 * np.log((1 + r) / (1 - r))
    stat = np.sqrt(n - k - 3) * abs(z)
    return stat < norm.ppf(1 - alpha / 2)


def pc_skeleton_with_sepsets(X, alpha=0.05, ell_max=2):
    p = X.shape[1]
    G = nx.complete_graph(p)
    sepsets = {(i, j): set() for i in range(p) for j in range(p) if i != j}
    l = 0
    while True:
        adj_snapshot = {i: set(G.neighbors(i)) for i in range(p)}
        max_adj = max((len(nb) for nb in adj_snapshot.values()), default=0)
        if max_adj < l:
            break
        for i in range(p):
            for j in list(adj_snapshot[i]):
                if j <= i:
                    continue
                if not G.has_edge(i, j):
                    continue
                adj_ij = [v for v in adj_snapshot[i] if v != j]
                if len(adj_ij) < l:
                    continue
                for cond in combinations(adj_ij, l):
                    if ci_test(X, i, j, cond, alpha):
                        if G.has_edge(i, j):
                            G.remove_edge(i, j)
                            sepsets[(i, j)] = set(cond)
                            sepsets[(j, i)] = set(cond)
                        break
        l += 1
        if l > ell_max:
            break
    return G, sepsets


# ====== CPDAG / Meek ======

def make_cpdag_from_skeleton(G):
    cpdag = nx.DiGraph()
    cpdag.add_nodes_from(G.nodes())
    for u, v in G.edges():
        cpdag.add_edge(u, v)
        cpdag.add_edge(v, u)
    return cpdag


def is_undirected(cpdag, u, v):
    return cpdag.has_edge(u, v) and cpdag.has_edge(v, u)


def orient_edge(cpdag, u, v):
    if cpdag.has_edge(v, u):
        cpdag.remove_edge(v, u)


def apply_meek_rules(cpdag):
    changed = True
    while changed:
        changed = False
        nodes = list(cpdag.nodes())

        # R1: Orient b-c into b->c if a->b, a-c not adjacent
        for b in nodes:
            for a in list(cpdag.predecessors(b)):
                if is_undirected(cpdag, a, b):
                    continue
                for c in list(cpdag.successors(b)):
                    if c == a:
                        continue
                    if not is_undirected(cpdag, b, c):
                        continue
                    if not cpdag.has_edge(a, c) and not cpdag.has_edge(c, a):
                        orient_edge(cpdag, b, c)
                        changed = True

        # R2: Orient a-c into a->c if a->b->c exists
        for b in nodes:
            for a in list(cpdag.predecessors(b)):
                if is_undirected(cpdag, a, b):
                    continue
                for c in list(cpdag.successors(b)):
                    if c == a:
                        continue
                    if is_undirected(cpdag, b, c):
                        continue
                    if is_undirected(cpdag, a, c):
                        orient_edge(cpdag, a, c)
                        changed = True

        # R3: Orient a-c into a->c
        for c in nodes:
            parents_c = [x for x in cpdag.predecessors(c) if not is_undirected(cpdag, x, c)]
            for b, d in combinations(parents_c, 2):
                if cpdag.has_edge(b, d) or cpdag.has_edge(d, b):
                    continue
                for a in nodes:
                    if a in (b, c, d):
                        continue
                    if (is_undirected(cpdag, a, b) and
                            is_undirected(cpdag, a, d) and
                            is_undirected(cpdag, a, c)):
                        orient_edge(cpdag, a, c)
                        changed = True

        # R4
        for c in nodes:
            for b in list(cpdag.predecessors(c)):
                if is_undirected(cpdag, b, c):
                    continue
                for d in list(cpdag.successors(c)):
                    if d == b:
                        continue
                    if is_undirected(cpdag, c, d):
                        continue
                    for a in nodes:
                        if a in (b, c, d):
                            continue
                        if (is_undirected(cpdag, a, b) and
                                is_undirected(cpdag, a, d) and
                                not cpdag.has_edge(a, c) and
                                not cpdag.has_edge(c, a)):
                            orient_edge(cpdag, a, d)
                            changed = True

    return cpdag


def orient_v_structures(G, sepsets, use_meek=True):
    cpdag = make_cpdag_from_skeleton(G)
    for b in G.nodes():
        neighbors_b = list(G.neighbors(b))
        for a, c in combinations(neighbors_b, 2):
            if G.has_edge(a, c):
                continue
            sep_ac = sepsets.get((a, c), None)
            sep_ca = sepsets.get((c, a), None)

            if sep_ac is None and sep_ca is None:
                continue

            combined_sep = set()
            if sep_ac is not None:
                combined_sep |= sep_ac
            if sep_ca is not None:
                combined_sep |= sep_ca

            if b not in combined_sep:
                orient_edge(cpdag, a, b)
                orient_edge(cpdag, c, b)

    if use_meek:
        cpdag = apply_meek_rules(cpdag)
    return cpdag


# =======================
# METRICS
# =======================

def true_skeleton_edges(B):
    return set(tuple(sorted((i, j))) for i, j in zip(*np.where(B != 0)))


def true_directed_edges(B):
    return set(zip(*np.where(B != 0)))


def skeleton_edges_from_graph(G):
    return set(tuple(sorted(e)) for e in G.edges())


def directed_edges_from_cpdag(cpdag):
    return {(i, j) for i, j in cpdag.edges() if not cpdag.has_edge(j, i)}


def compute_metrics(true_set, pred_set):
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    shd = fp + fn
    return shd, precision, recall, f1


def orientation_metrics_cpdag(true_dir, cpdag):
    pred_dir = directed_edges_from_cpdag(cpdag)
    n_oriented = len(pred_dir)
    _, dp, dr, df1 = compute_metrics(true_dir, pred_dir)
    return n_oriented, dp, dr, df1


def communication_cost(K, p, ell):
    """Communication cost approximation (§3.2.8.4, Eq. 36): K * p^2 * 2^ell_max."""
    return K * (p ** 2) * (2 ** ell)


# =======================
# FEDERATED HELPERS
# =======================

def federated_split(X, K):
    idx = np.random.permutation(X.shape[0])
    return [X[idx[i::K]] for i in range(K)]


def local_pc_clients(client_data, alpha=0.05, ell_max=2):
    def run_pc(Xc):
        Gc, sepsc = pc_skeleton_with_sepsets(Xc, alpha, ell_max)
        return skeleton_edges_from_graph(Gc), sepsc
    results = Parallel(n_jobs=-1)(delayed(run_pc)(Xc) for Xc in client_data)
    return [r[0] for r in results], [r[1] for r in results]


def naive_majority_aggregation(client_edge_sets, tau=0.5):
    K = len(client_edge_sets)
    counter = {}
    for edges in client_edge_sets:
        for e in edges:
            counter[e] = counter.get(e, 0) + 1
    return {e for e, c in counter.items() if c / K >= tau}


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def compute_consensus_reliability(client_edge_sets):
    K = len(client_edge_sets)
    weights = []
    for k in range(K):
        js = [jaccard(client_edge_sets[k], client_edge_sets[j]) for j in range(K) if j != k]
        weights.append(np.mean(js) if js else 1.0 / K)
    weights = np.array(weights, dtype=float)
    weights = weights ** 2
    total = weights.sum()
    if total < 1e-12:
        weights = np.ones(K) / K
    else:
        weights /= total
    return weights


def compute_oracle_reliability(client_data, client_edge_sets, true_skel, p):
    weights = []
    for Xc, edges in zip(client_data, client_edge_sets):
        _, _, _, f1 = compute_metrics(true_skel, edges)
        # Pure oracle: weight = f1 score directly, clipped to avoid zero
        r = np.clip(f1, 0.05, 1.0)
        weights.append(r)
    weights = np.array(weights, dtype=float)
    weights = weights ** 2  # Squared to amplify differences (Eq. 24 analog)
    total = weights.sum()
    if total < 1e-12:
        weights = np.ones(len(client_edge_sets)) / len(client_edge_sets)
    else:
        weights /= total
    return weights


def weighted_aggregation_with_weights(client_edge_sets, weights, tau=0.5):
    total_weight = sum(weights)
    if total_weight < 1e-12:
        return set()
    counter = {}
    for k, edges in enumerate(client_edge_sets):
        for e in edges:
            counter[e] = counter.get(e, 0.0) + weights[k]
    # Compare fractional weighted support against tau
    return {e for e, w in counter.items() if (w / total_weight) >= tau}


def aggregate_sepsets(client_seps, weights, tau=0.5):
    total_weight = sum(weights)
    if total_weight < 1e-12:
        return {}
    # Lenient threshold for sepset: tau/2
    sep_tau = tau / 2.0

    all_keys = set()
    for seps in client_seps:
        all_keys.update(seps.keys())
    agg_seps = {}
    for key in all_keys:
        votes = {}
        for k, seps_k in enumerate(client_seps):
            if key in seps_k:
                for z in seps_k[key]:
                    votes[z] = votes.get(z, 0.0) + weights[k]
        if votes:
            included = {z for z, w in votes.items() if (w / total_weight) >= sep_tau}
            if included:  # Only add entry if non-empty (avoid spurious empty sepsets)
                agg_seps[key] = included
    return agg_seps


# =======================
# SINGLE EXPERIMENT
# =======================

def run_single_experiment(
    p=20, N=5000, K=5, edge_prob=0.2, hetero="mild",
    mechanism_shift=False,
    alpha=ALPHA, tau=TAU_DEFAULT, ell=ELL_MAX
):
    B = generate_random_dag(p, edge_prob)
    true_skel = true_skeleton_edges(B)
    true_dir = true_directed_edges(B)

    if mechanism_shift:
        client_data = []
        for _ in range(K):
            Bc = perturb_B(B)
            n_k = N // K
            Xk = simulate_linear_sem(B, n_k, hetero, B_perturbed=Bc)
            client_data.append(Xk)
    else:
        X_pool = simulate_linear_sem(B, N, hetero)
        client_data = federated_split(X_pool, K)

    # Centralized: pool all client data (§4.5: upper-bound non-federated reference)
    X_global = np.vstack(client_data)
    results = {}

    # ---- Centralized PC ----
    t0 = time.time()
    Gc, seps_c = pc_skeleton_with_sepsets(X_global, alpha, ell)
    pred_skel_c = skeleton_edges_from_graph(Gc)
    cpdag_c = orient_v_structures(Gc, seps_c)
    shd, prec, rec, f1 = compute_metrics(true_skel, pred_skel_c)
    n_or, dp, dr, df1 = orientation_metrics_cpdag(true_dir, cpdag_c)
    results["centralized"] = {
        "SHD": shd, "F1": f1, "Precision": prec, "Recall": rec,
        "n_oriented": n_or, "Dir_Prec": dp, "Dir_Rec": dr, "Dir_F1": df1,
        "Comm": 0, "Runtime": time.time() - t0
    }

    # ---- Local PC (per client, averaged) ----
    t0 = time.time()
    client_edges, client_seps = local_pc_clients(client_data, alpha, ell)
    local_shds, local_f1s, local_df1s, local_precs, local_recs = [], [], [], [], []
    for edges, seps in zip(client_edges, client_seps):
        Gcl = nx.Graph()
        Gcl.add_nodes_from(range(p))
        Gcl.add_edges_from(edges)
        cpdag_l = orient_v_structures(Gcl, seps)
        shd_l, prec_l, rec_l, f1_l = compute_metrics(true_skel, edges)
        _, _, _, df1_l = orientation_metrics_cpdag(true_dir, cpdag_l)
        local_shds.append(shd_l)
        local_f1s.append(f1_l)
        local_df1s.append(df1_l)
        local_precs.append(prec_l)
        local_recs.append(rec_l)
    results["local"] = {
        "SHD": float(np.mean(local_shds)), "F1": float(np.mean(local_f1s)),
        "Precision": float(np.mean(local_precs)), "Recall": float(np.mean(local_recs)),
        "n_oriented": 0, "Dir_Prec": 0, "Dir_Rec": 0,
        "Dir_F1": float(np.mean(local_df1s)),
        "Comm": communication_cost(K, p, ell), "Runtime": time.time() - t0
    }

    # ---- FedPC-Naive (§4.5 / §3.2.8.1) ----
    # Majority voting, no sepset aggregation, no reliability weighting
    t0 = time.time()
    pred_naive = naive_majority_aggregation(client_edges, tau)
    Gn = nx.Graph()
    Gn.add_nodes_from(range(p))
    Gn.add_edges_from(pred_naive)
    cpdag_n = orient_v_structures(Gn, {})
    shd, prec, rec, f1 = compute_metrics(true_skel, pred_naive)
    n_or, dp, dr, df1 = orientation_metrics_cpdag(true_dir, cpdag_n)
    results["naive"] = {
        "SHD": shd, "F1": f1, "Precision": prec, "Recall": rec,
        "n_oriented": n_or, "Dir_Prec": dp, "Dir_Rec": dr, "Dir_F1": df1,
        "Comm": communication_cost(K, p, ell), "Runtime": time.time() - t0
    }

    # ---- FedPC-Consensus (Proposed) ----
    t0 = time.time()
    w_con = compute_consensus_reliability(client_edges)
    pred_con = weighted_aggregation_with_weights(client_edges, w_con, tau)
    agg_seps_con = aggregate_sepsets(client_seps, w_con, tau)
    Gcon = nx.Graph()
    Gcon.add_nodes_from(range(p))
    Gcon.add_edges_from(pred_con)
    cpdag_con = orient_v_structures(Gcon, agg_seps_con)
    shd, prec, rec, f1 = compute_metrics(true_skel, pred_con)
    n_or, dp, dr, df1 = orientation_metrics_cpdag(true_dir, cpdag_con)
    results["fedpc_consensus"] = {
        "SHD": shd, "F1": f1, "Precision": prec, "Recall": rec,
        "n_oriented": n_or, "Dir_Prec": dp, "Dir_Rec": dr, "Dir_F1": df1,
        "Comm": communication_cost(K, p, ell), "Runtime": time.time() - t0
    }

    # ---- FedPC-Oracle (Upper-bound ablation only, §4.5) ----
    t0 = time.time()
    w_ora = compute_oracle_reliability(client_data, client_edges, true_skel, p)
    pred_ora = weighted_aggregation_with_weights(client_edges, w_ora, tau)
    agg_seps_ora = aggregate_sepsets(client_seps, w_ora, tau)
    Gora = nx.Graph()
    Gora.add_nodes_from(range(p))
    Gora.add_edges_from(pred_ora)
    cpdag_ora = orient_v_structures(Gora, agg_seps_ora)
    shd, prec, rec, f1 = compute_metrics(true_skel, pred_ora)
    n_or, dp, dr, df1 = orientation_metrics_cpdag(true_dir, cpdag_ora)
    results["fedpc_oracle"] = {
        "SHD": shd, "F1": f1, "Precision": prec, "Recall": rec,
        "n_oriented": n_or, "Dir_Prec": dp, "Dir_Rec": dr, "Dir_F1": df1,
        "Comm": communication_cost(K, p, ell), "Runtime": time.time() - t0
    }

    return results


# =======================
# STATISTICS HELPERS
# =======================

def wilcoxon_ci(a, b, alpha=0.05, method="fedpc_consensus vs naive"):
    """Paired Wilcoxon signed-rank test with 95% CI and Cohen's d (§3.2.8.6, Eq. 40-41)."""
    diffs = np.array(a) - np.array(b)
    n = len(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    se = std_diff / np.sqrt(n)
    t_crit = t_dist.ppf(1 - alpha / 2, df=n - 1)
    ci_lo = mean_diff - t_crit * se
    ci_hi = mean_diff + t_crit * se
    cohen_d = mean_diff / (std_diff + 1e-12)
    try:
        stat, p_val = wilcoxon(diffs, alternative="two-sided")
    except Exception:
        stat, p_val = float("nan"), float("nan")
    return {
        "comparison": method,
        "mean_diff": mean_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "cohen_d": cohen_d,
        "wilcoxon_stat": stat,
        "p_value": p_val,
        "n": n
    }


# =======================
# SACHS DATASET
# =======================

SACHS_GROUND_TRUTH_EDGES = [
    ("praf", "pmek"),
    ("pmek", "p44.42"),
    ("pmek", "pakts473"),
    ("PIP3", "PIP2"),
    ("PIP3", "pakts473"),
    ("plcg", "PIP2"),
    ("plcg", "PIP3"),
    ("PKA", "p44.42"),
    ("PKA", "pakts473"),
    ("PKA", "pjnk"),
    ("PKA", "P38"),
    ("PKA", "praf"),
    ("PKC", "praf"),
    ("PKC", "pmek"),
    ("PKC", "pjnk"),
    ("PKC", "P38"),
    ("PKC", "PKA"),
]

SACHS_CANONICAL = [
    "praf", "pmek", "p44.42", "pakts473",
    "PKA", "PKC", "P38", "pjnk", "plcg", "PIP2", "PIP3"
]

SACHS_COL_MAP = {
    "raf": "praf", "praf": "praf",
    "mek": "pmek", "pmek": "pmek",
    "erk": "p44.42", "p44.42": "p44.42",
    "akt": "pakts473", "pakts473": "pakts473",
    "pka": "PKA", "PKA": "PKA",
    "pkc": "PKC", "PKC": "PKC",
    "p38": "P38", "P38": "P38",
    "jnk": "pjnk", "pjnk": "pjnk",
    "plcg": "plcg",
    "pip2": "PIP2", "PIP2": "PIP2",
    "pip3": "PIP3", "PIP3": "PIP3",
}


def load_sachs(sachs_dir):
    """
    Load & merge all Sachs CSV files, normalise column names.
    Returns (X, true_skel_edges, true_dir_edges, col_names) or (None,...) on failure.
    """
    if not os.path.isdir(sachs_dir):
        return None, None, None, None

    csvs = [
        f for f in os.listdir(sachs_dir)
        if f.endswith(".csv") and f.lower() not in ("groundtruth.csv",)
    ]
    if not csvs:
        print("  [WARN] No CSV files found in Sachs dir.")
        return None, None, None, None

    dfs = []
    for fn in csvs:
        fpath = os.path.join(sachs_dir, fn)
        for sep in [",", "\t", ";"]:
            try:
                df = pd.read_csv(fpath, sep=sep)
                if df.shape[1] > 1:
                    dfs.append(df)
                    break
            except Exception:
                continue

    if not dfs:
        print("  [WARN] Could not parse any Sachs CSV.")
        return None, None, None, None

    combined = pd.concat(dfs, ignore_index=True)

    rename = {}
    for col in combined.columns:
        key = col.strip().lower()
        if key in SACHS_COL_MAP:
            rename[col] = SACHS_COL_MAP[key]
        else:
            for canon in SACHS_CANONICAL:
                if col.strip().lower() == canon.lower():
                    rename[col] = canon
                    break

    combined = combined.rename(columns=rename)
    available = [c for c in SACHS_CANONICAL if c in combined.columns]

    if len(available) < 5:
        numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
        available = numeric_cols
        print(f"  [WARN] Few canonical columns found; using {len(available)} numeric cols.")

    X = combined[available].dropna().values.astype(float)
    col_names = available

    gt_skel = set()
    gt_dir = set()
    for u, v in SACHS_GROUND_TRUTH_EDGES:
        if u in col_names and v in col_names:
            ui = col_names.index(u)
            vi = col_names.index(v)
            gt_skel.add(tuple(sorted((ui, vi))))
            gt_dir.add((ui, vi))

    print(f"  Columns used ({len(col_names)}): {col_names}")
    print(f"  X shape: {X.shape} | true_skel={len(gt_skel)} | true_dir={len(gt_dir)}")
    return X, gt_skel, gt_dir, col_names


def run_sachs_experiment(sachs_dir, K=5, tau=TAU_DEFAULT, ell=ELL_MAX, alpha=ALPHA, reps=10):
    print("  Loading Sachs data ...")
    result = load_sachs(sachs_dir)
    if result[0] is None:
        print("  [WARN] Could not load Sachs data. Skipping.")
        return [], None

    X, true_skel, true_dir, col_names = result
    p = X.shape[1]
    rows = []

    for rep in range(reps):
        np.random.seed(SEED + rep)

        client_data = federated_split(X, K)

        # Centralized
        Gc, seps_c = pc_skeleton_with_sepsets(X, alpha, ell)
        pred_c = skeleton_edges_from_graph(Gc)
        cpdag_c = orient_v_structures(Gc, seps_c)
        shd_c, prec_c, rec_c, f1_c = compute_metrics(true_skel, pred_c)
        _, dp_c, dr_c, df1_c = orientation_metrics_cpdag(true_dir, cpdag_c)

        client_edges, client_seps = local_pc_clients(client_data, alpha, ell)

        # Local (averaged across clients)
        local_shds, local_f1s, local_df1s = [], [], []
        for edges, seps in zip(client_edges, client_seps):
            Gcl = nx.Graph(); Gcl.add_nodes_from(range(p)); Gcl.add_edges_from(edges)
            cpdag_l = orient_v_structures(Gcl, seps)
            s, _, _, f = compute_metrics(true_skel, edges)
            _, _, _, df = orientation_metrics_cpdag(true_dir, cpdag_l)
            local_shds.append(s); local_f1s.append(f); local_df1s.append(df)

        # Naive (no sepset aggregation)
        pred_naive = naive_majority_aggregation(client_edges, tau)
        Gn = nx.Graph(); Gn.add_nodes_from(range(p)); Gn.add_edges_from(pred_naive)
        cpdag_n = orient_v_structures(Gn, {})  # Fixed: no sepsets → no collider orientation
        shd_n, prec_n, rec_n, f1_n = compute_metrics(true_skel, pred_naive)
        _, dp_n, dr_n, df1_n = orientation_metrics_cpdag(true_dir, cpdag_n)

        # Consensus
        w_con = compute_consensus_reliability(client_edges)
        pred_con = weighted_aggregation_with_weights(client_edges, w_con, tau)
        agg_seps_con = aggregate_sepsets(client_seps, w_con, tau)
        Gcon = nx.Graph(); Gcon.add_nodes_from(range(p)); Gcon.add_edges_from(pred_con)
        cpdag_con = orient_v_structures(Gcon, agg_seps_con)
        shd_con, prec_con, rec_con, f1_con = compute_metrics(true_skel, pred_con)
        _, dp_con, dr_con, df1_con = orientation_metrics_cpdag(true_dir, cpdag_con)

        # Oracle
        w_ora = compute_oracle_reliability(client_data, client_edges, true_skel, p)
        pred_ora = weighted_aggregation_with_weights(client_edges, w_ora, tau)
        agg_seps_ora = aggregate_sepsets(client_seps, w_ora, tau)
        Gora = nx.Graph(); Gora.add_nodes_from(range(p)); Gora.add_edges_from(pred_ora)
        cpdag_ora = orient_v_structures(Gora, agg_seps_ora)
        shd_ora, prec_ora, rec_ora, f1_ora = compute_metrics(true_skel, pred_ora)
        _, dp_ora, dr_ora, df1_ora = orientation_metrics_cpdag(true_dir, cpdag_ora)

        rows.append({
            "rep": rep,
            # centralized
            "centralized_SHD": shd_c, "centralized_F1": f1_c,
            "centralized_Precision": prec_c, "centralized_Recall": rec_c,
            "centralized_Dir_F1": df1_c, "centralized_Dir_Prec": dp_c, "centralized_Dir_Rec": dr_c,
            # local (averaged across clients)
            "local_SHD": float(np.mean(local_shds)),
            "local_F1": float(np.mean(local_f1s)),
            "local_Dir_F1": float(np.mean(local_df1s)),
            # naive
            "naive_SHD": shd_n, "naive_F1": f1_n,
            "naive_Precision": prec_n, "naive_Recall": rec_n,
            "naive_Dir_F1": df1_n, "naive_Dir_Prec": dp_n, "naive_Dir_Rec": dr_n,
            # consensus
            "fedpc_consensus_SHD": shd_con, "fedpc_consensus_F1": f1_con,
            "fedpc_consensus_Precision": prec_con, "fedpc_consensus_Recall": rec_con,
            "fedpc_consensus_Dir_F1": df1_con, "fedpc_consensus_Dir_Prec": dp_con,
            "fedpc_consensus_Dir_Rec": dr_con,
            # oracle
            "fedpc_oracle_SHD": shd_ora, "fedpc_oracle_F1": f1_ora,
            "fedpc_oracle_Precision": prec_ora, "fedpc_oracle_Recall": rec_ora,
            "fedpc_oracle_Dir_F1": df1_ora, "fedpc_oracle_Dir_Prec": dp_ora,
            "fedpc_oracle_Dir_Rec": dr_ora,
        })

    return rows, col_names


# =======================
# AGGREGATE + COLLECT
# =======================

def aggregate_reps(rep_results, scenario_name):
    rows = []
    methods = ["centralized", "local", "naive", "fedpc_consensus", "fedpc_oracle"]
    metrics_keys = [
        "SHD", "F1", "Precision", "Recall",
        "Dir_F1", "Dir_Prec", "Dir_Rec",
        "n_oriented", "Comm", "Runtime"
    ]
    for m in methods:
        if m not in rep_results[0]:
            continue
        row = {"scenario": scenario_name, "method": m}
        for k in metrics_keys:
            vals = [r[m].get(k, np.nan) for r in rep_results]
            row[f"{k}_mean"] = float(np.nanmean(vals))
            row[f"{k}_std"] = float(np.nanstd(vals))
        rows.append(row)
    return rows


# =======================
# HELPER: bar chart with error bars
# =======================

def _bar_group(ax, methods, means, stds, ylabel, title, ylim=None):
    colors = [PALETTE[m] for m in methods]
    labels = [METHOD_LABELS[m] for m in methods]
    x = np.arange(len(methods))
    bars = ax.bar(x, means, color=colors, alpha=0.88,
                  edgecolor="white", linewidth=1.1,
                  yerr=stds, capsize=4,
                  error_kw={"elinewidth": 1.4, "ecolor": "#444"})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=20, ha="right")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    for bar, val, std in zip(bars, means, stds):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    return bars


def _line_group(ax, x_vals, data_dict, ylabel, title, xlabel="", marker="o"):
    for m, vals in data_dict.items():
        means = [v[0] for v in vals]
        stds  = [v[1] for v in vals]
        ax.plot(x_vals, means, marker=marker, color=PALETTE[m],
                label=METHOD_LABELS[m], linewidth=1.8, markersize=5)
        ax.fill_between(x_vals,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color=PALETTE[m], alpha=0.12)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, framealpha=0.7)


# =======================
# FIG helpers: extract from df_main
# =======================

def _get(df, scenario, method, metric):
    row = df[(df["scenario"] == scenario) & (df["method"] == method)]
    if row.empty:
        return np.nan, np.nan
    return float(row[f"{metric}_mean"].values[0]), float(row[f"{metric}_std"].values[0])


# =======================
# VISUALIZATION SECTION 5
# =======================

def generate_all_figures(df_main, df_stat, out_dir="results/figures"):
    os.makedirs(out_dir, exist_ok=True)
    methods_full = PLOT_METHODS

    base = "p20_hetero_mild"

    # ============================================================
    # FIG 5.2 — Overall Structural Recovery
    # ============================================================
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle("§5.2 Overall Structural Recovery\n"
                 f"(baseline: p=20, N=5000, K=5, hetero=mild, τ=0.5)",
                 fontsize=11, fontweight="bold")

    for ax, metric, ylabel, title, ylim in [
        (axes[0], "SHD",       "SHD (↓ better)",   "Structural\nHamming Distance",  None),
        (axes[1], "Precision",  "Precision (↑)",    "Skeleton\nPrecision",           (0, 1.25)),
        (axes[2], "Recall",     "Recall (↑)",       "Skeleton\nRecall",              (0, 1.25)),
        (axes[3], "F1",         "F1 (↑)",           "Skeleton F1",                   (0, 1.25)),
    ]:
        means = [_get(df_main, base, m, metric)[0] for m in methods_full]
        stds  = [_get(df_main, base, m, metric)[1] for m in methods_full]
        _bar_group(ax, methods_full, means, stds, ylabel, title, ylim)

    plt.tight_layout()
    p_path = os.path.join(out_dir, "fig5_2_overall_structural_recovery.png")
    fig.savefig(p_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p_path}")

    # ============================================================
    # FIG 5.3 — Orientation Recovery
    # ============================================================
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle("§5.3 Orientation Recovery (CPDAG directed edges)\n"
                 f"(baseline: p=20, N=5000, K=5, hetero=mild, τ=0.5)",
                 fontsize=11, fontweight="bold")

    for ax, metric, ylabel, title, ylim in [
        (axes[0], "Dir_Prec", "Dir-Precision (↑)", "Directional\nPrecision", (0, 1.25)),
        (axes[1], "Dir_Rec",  "Dir-Recall (↑)",    "Directional\nRecall",    (0, 1.25)),
        (axes[2], "Dir_F1",   "Dir-F1 (↑)",        "Directional F1",         (0, 1.25)),
        (axes[3], "n_oriented","Count",             "# Oriented\nEdges",      None),
    ]:
        means = [_get(df_main, base, m, metric)[0] for m in methods_full]
        stds  = [_get(df_main, base, m, metric)[1] for m in methods_full]
        _bar_group(ax, methods_full, means, stds, ylabel, title, ylim)

    plt.tight_layout()
    p_path = os.path.join(out_dir, "fig5_3_orientation_recovery.png")
    fig.savefig(p_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p_path}")

    # ============================================================
    # FIG 5.4a — Effect of Federation Scale: K
    # ============================================================
    K_vals = [2, 5, 10]
    K_scenarios = [f"p20_K{k}" for k in K_vals]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("§5.4 Effect of Federation Scale: Number of Clients K\n(p=20, N=5000, hetero=mild)",
                 fontsize=11, fontweight="bold")

    for ax, metric, ylabel, title, ylim in [
        (axes[0], "SHD", "SHD (↓)", "Structural Hamming Distance vs K", None),
        (axes[1], "F1",  "F1 (↑)",  "Skeleton F1 vs K",                 (0, 1.1)),
        (axes[2], "Dir_F1", "Dir-F1 (↑)", "Directional F1 vs K",        (0, 1.1)),
    ]:
        data_d = {m: [_get(df_main, sc, m, metric) for sc in K_scenarios]
                  for m in methods_full}
        _line_group(ax, K_vals, data_d, ylabel, title, xlabel="K (clients)")

    plt.tight_layout()
    p_path = os.path.join(out_dir, "fig5_4a_federation_scale_K.png")
    fig.savefig(p_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p_path}")

    # ============================================================
    # FIG 5.4b — Effect of Sample Size N
    # ============================================================
    N_vals = [100, 3000, 5000]
    N_scenarios = [f"p20_N{n}" for n in N_vals]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("§5.4 Effect of Sample Size N (per-client fragmentation)\n(p=20, K=5, hetero=mild)",
                 fontsize=11, fontweight="bold")

    for ax, metric, ylabel, title, ylim in [
        (axes[0], "SHD",    "SHD (↓)",    "SHD vs N",         None),
        (axes[1], "F1",     "F1 (↑)",     "Skeleton F1 vs N", (0, 1.1)),
        (axes[2], "Dir_F1", "Dir-F1 (↑)", "Dir-F1 vs N",      (0, 1.1)),
    ]:
        data_d = {m: [_get(df_main, sc, m, metric) for sc in N_scenarios]
                  for m in methods_full}
        _line_group(ax, N_vals, data_d, ylabel, title, xlabel="Total samples N")

    plt.tight_layout()
    p_path = os.path.join(out_dir, "fig5_4b_sample_size_N.png")
    fig.savefig(p_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p_path}")

    # ============================================================
    # FIG 5.4c — Communication Cost vs K
    # ============================================================
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("§5.4 Communication Cost vs K (p=20, ell=2)", fontsize=11, fontweight="bold")

    for m in ["naive", "fedpc_consensus", "fedpc_oracle"]:
        comms = [_get(df_main, sc, m, "Comm")[0] for sc in K_scenarios]
        ax.plot(K_vals, comms, marker="o", color=PALETTE[m],
                label=METHOD_LABELS[m], linewidth=1.8, markersize=6)

    ax.set_xlabel("K (clients)", fontsize=9)
    ax.set_ylabel("Communication cost (# messages)", fontsize=9)
    ax.set_title("Communication overhead scales linearly with K", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    p_path = os.path.join(out_dir, "fig5_4c_comm_cost_K.png")
    fig.savefig(p_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p_path}")

    # ============================================================
    # FIG 5.5a — Graph Density (edge_prob)
    # ============================================================
    phi_vals = [0.1, 0.2]
    phi_scenarios = [f"p20_phi{phi}" for phi in phi_vals]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("§5.5 Effect of Graph Density (edge probability φ)\n(p=20, N=5000, K=5)",
                 fontsize=11, fontweight="bold")

    for ax, metric, ylabel, title in [
        (axes[0], "SHD",    "SHD (↓)",    "SHD vs φ"),
        (axes[1], "F1",     "F1 (↑)",     "Skeleton F1 vs φ"),
        (axes[2], "Dir_F1", "Dir-F1 (↑)", "Dir-F1 vs φ"),
    ]:
        data_d = {m: [_get(df_main, sc, m, metric) for sc in phi_scenarios]
                  for m in methods_full}
        _line_group(ax, phi_vals, data_d, ylabel, title, xlabel="Edge probability φ")

    plt.tight_layout()
    p_path = os.path.join(out_dir, "fig5_5a_graph_density.png")
    fig.savefig(p_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p_path}")

    # ============================================================
    # FIG 5.5b — Conditioning depth (ell)
    # ============================================================
    ell_vals = [1, 2]
    ell_scenarios = [f"p20_ell{e}" for e in ell_vals]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("§5.5 Effect of Max Conditioning Depth ℓ\n(p=20, N=5000, K=5, hetero=mild)",
                 fontsize=11, fontweight="bold")

    for ax, metric, ylabel, title in [
        (axes[0], "SHD",    "SHD (↓)",    "SHD vs ℓ"),
        (axes[1], "F1",     "F1 (↑)",     "Skeleton F1 vs ℓ"),
        (axes[2], "Runtime","Runtime (s)","Runtime vs ℓ"),
    ]:
        data_d = {m: [_get(df_main, sc, m, metric) for sc in ell_scenarios]
                  for m in methods_full}
        _line_group(ax, ell_vals, data_d, ylabel, title, xlabel="Max conditioning depth ℓ")

    plt.tight_layout()
    p_path = os.path.join(out_dir, "fig5_5b_conditioning_depth.png")
    fig.savefig(p_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p_path}")

    # ============================================================
    # FIG 5.6 — Effect of Heterogeneity
    # ============================================================
    hetero_labels = ["Homogeneous\n(None)", "Mild hetero", "Strong hetero", "Mechanism\nShift"]
    hetero_scenarios = [
        "p20_hetero_None", "p20_hetero_mild", "p20_hetero_strong",
        "p20_mechanism_shift"
    ]

    existing = [sc for sc in hetero_scenarios if not df_main[df_main["scenario"] == sc].empty]
    ex_labels = [hetero_labels[hetero_scenarios.index(sc)] for sc in existing]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("§5.6 Effect of Heterogeneity & Mechanism Shift\n(p=20, N=5000, K=5)",
                 fontsize=11, fontweight="bold")

    for ax, metric, ylabel, title, ylim in [
        (axes[0], "SHD",    "SHD (↓)",    "SHD vs heterogeneity",    None),
        (axes[1], "F1",     "F1 (↑)",     "Skeleton F1",              (0, 1.1)),
        (axes[2], "Dir_F1", "Dir-F1 (↑)", "Directional F1",           (0, 1.1)),
    ]:
        x = np.arange(len(existing))
        for m in methods_full:
            means = [_get(df_main, sc, m, metric)[0] for sc in existing]
            stds  = [_get(df_main, sc, m, metric)[1] for sc in existing]
            ax.plot(x, means, marker="o", color=PALETTE[m],
                    label=METHOD_LABELS[m], linewidth=1.8, markersize=5)
            ax.fill_between(x,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=PALETTE[m], alpha=0.10)
        ax.set_xticks(x)
        ax.set_xticklabels(ex_labels, fontsize=8, rotation=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, framealpha=0.7)
        if ylim:
            ax.set_ylim(*ylim)

    plt.tight_layout()
    p_path = os.path.join(out_dir, "fig5_6_heterogeneity.png")
    fig.savefig(p_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p_path}")

    # ============================================================
    # FIG 5.7a — Runtime comparison (bar, baseline)
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("§5.7 Runtime per Experiment\n(baseline: p=20, N=5000, K=5, hetero=mild)",
                 fontsize=11, fontweight="bold")
    means = [_get(df_main, base, m, "Runtime")[0] for m in methods_full]
    stds  = [_get(df_main, base, m, "Runtime")[1] for m in methods_full]
    _bar_group(ax, methods_full, means, stds, "Runtime (s)", "Runtime (lower=better)")
    plt.tight_layout()
    p_path = os.path.join(out_dir, "fig5_7a_runtime_bar.png")
    fig.savefig(p_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p_path}")

    # ============================================================
    # FIG 5.7b — Accuracy-Efficiency trade-off scatter (F1 vs Runtime)
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("§5.7 Accuracy–Efficiency Trade-off\n(F1 vs Runtime, baseline scenario)",
                 fontsize=11, fontweight="bold")
    for m in methods_full:
        f1_mu, f1_sd = _get(df_main, base, m, "F1")
        rt_mu, rt_sd = _get(df_main, base, m, "Runtime")
        ax.errorbar(rt_mu, f1_mu,
                    xerr=rt_sd, yerr=f1_sd,
                    fmt="o", color=PALETTE[m], markersize=10,
                    label=METHOD_LABELS[m], capsize=4, linewidth=1.5)
        ax.annotate(METHOD_LABELS[m], (rt_mu, f1_mu),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=PALETTE[m])
    ax.set_xlabel("Runtime (s)", fontsize=10)
    ax.set_ylabel("Skeleton F1 (↑)", fontsize=10)
    ax.set_title("Upper-right = better (high F1, low runtime)", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.7)
    plt.tight_layout()
    p_path = os.path.join(out_dir, "fig5_7b_accuracy_efficiency_scatter.png")
    fig.savefig(p_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p_path}")

    # ============================================================
    # FIG 5.7c — Communication cost vs Skeleton F1 scatter
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("§5.7 Communication Cost vs F1\n(all K scenarios, federated methods only)",
                 fontsize=11, fontweight="bold")
    fed_methods = ["naive", "fedpc_consensus", "fedpc_oracle"]
    for m in fed_methods:
        comms = [_get(df_main, sc, m, "Comm")[0] for sc in K_scenarios]
        f1s   = [_get(df_main, sc, m, "F1")[0]   for sc in K_scenarios]
        ax.plot(comms, f1s, marker="o", color=PALETTE[m],
                label=METHOD_LABELS[m], linewidth=1.6, markersize=7)
        for k_v, (c, f) in zip(K_vals, zip(comms, f1s)):
            ax.annotate(f"K={k_v}", (c, f), textcoords="offset points",
                        xytext=(5, 3), fontsize=7.5, color=PALETTE[m])
    ax.set_xlabel("Communication cost", fontsize=10)
    ax.set_ylabel("Skeleton F1", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    p_path = os.path.join(out_dir, "fig5_7c_comm_cost_vs_f1.png")
    fig.savefig(p_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p_path}")

    # ============================================================
    # FIG 5.8a — Wilcoxon forest plot (F1)
    # ============================================================
    _plot_forest(df_stat[df_stat["metric"] == "F1"],
                 "§5.8 Statistical Comparison: FedPC-Consensus vs Naive\n"
                 "Mean F1 difference (95% CI, Wilcoxon signed-rank)",
                 os.path.join(out_dir, "fig5_8a_wilcoxon_forest_F1.png"),
                 "Mean diff (Consensus − Naive), Skeleton F1")
    print(f"  Saved: {os.path.join(out_dir, 'fig5_8a_wilcoxon_forest_F1.png')}")

    # ============================================================
    # FIG 5.8b — Wilcoxon forest plot (Dir_F1)
    # ============================================================
    _plot_forest(df_stat[df_stat["metric"] == "Dir_F1"],
                 "§5.8 Statistical Comparison: FedPC-Consensus vs Naive\n"
                 "Mean Dir-F1 difference (95% CI, Wilcoxon signed-rank)",
                 os.path.join(out_dir, "fig5_8b_wilcoxon_forest_DirF1.png"),
                 "Mean diff (Consensus − Naive), Directional F1")
    print(f"  Saved: {os.path.join(out_dir, 'fig5_8b_wilcoxon_forest_DirF1.png')}")

    # ============================================================
    # FIG 5.8c — Effect size (Cohen's d) heatmap
    # ============================================================
    _plot_effect_size(df_stat, out_dir)
    print(f"  Saved: {os.path.join(out_dir, 'fig5_8c_effect_size_heatmap.png')}")

    # ============================================================
    # FIG 5.8d — p-value significance dot chart
    # ============================================================
    _plot_pvalue_dot(df_stat, out_dir)
    print(f"  Saved: {os.path.join(out_dir, 'fig5_8d_pvalue_significance.png')}")

    print(f"\nAll §5 figures saved to: {out_dir}")


def _plot_forest(df_sub, title, path, xlabel):
    df_sub = df_sub.dropna(subset=["mean_diff", "ci_lo", "ci_hi"]).copy()
    df_sub = df_sub.sort_values("mean_diff", ascending=True).reset_index(drop=True)
    if df_sub.empty:
        return
    n = len(df_sub)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.38)))
    y = np.arange(n)
    xerr_lo = (df_sub["mean_diff"] - df_sub["ci_lo"]).abs()
    xerr_hi = (df_sub["ci_hi"] - df_sub["mean_diff"]).abs()
    ax.barh(y, df_sub["mean_diff"],
            xerr=[xerr_lo, xerr_hi],
            color=["#27ae60" if v > 0 else "#e74c3c" for v in df_sub["mean_diff"]],
            alpha=0.80, capsize=3, height=0.55,
            error_kw={"elinewidth": 1.2, "ecolor": "#333"})
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df_sub["scenario"], fontsize=7.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for i, row in df_sub.iterrows():
        p_str = f"p={row['p_value']:.3f}" if pd.notna(row['p_value']) else "p=?"
        sig = "*" if (pd.notna(row["p_value"]) and row["p_value"] < 0.05) else ""
        ax.text(df_sub["ci_hi"].max() * 1.02, i,
                f"{p_str}{sig}", va="center", fontsize=6.5)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_effect_size(df_stat, out_dir):
    pivot = df_stat.pivot_table(index="scenario", columns="metric", values="cohen_d")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2.5),
                                     max(4, len(pivot) * 0.35 + 1)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    plt.colorbar(im, ax=ax, label="Cohen's d")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(val) < 1.5 else "white")
    ax.set_title("§5.8 Effect Size (Cohen's d): FedPC-Consensus vs Naive\n"
                 "Positive = Consensus better", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_8c_effect_size_heatmap.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_pvalue_dot(df_stat, out_dir):
    df_f1  = df_stat[df_stat["metric"] == "F1"].copy()
    df_dir = df_stat[df_stat["metric"] == "Dir_F1"].copy()
    if df_f1.empty and df_dir.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(df_f1) * 0.38 + 1)), sharey=True)
    fig.suptitle("§5.8 Wilcoxon p-values: FedPC-Consensus vs Naive\n"
                 "Green dot = p<0.05 (significant), Red = not significant",
                 fontsize=10, fontweight="bold")
    for ax, df_sub, title in [
        (axes[0], df_f1,  "Skeleton F1"),
        (axes[1], df_dir, "Directional F1"),
    ]:
        df_sub = df_sub.dropna(subset=["p_value"]).sort_values("p_value").reset_index(drop=True)
        if df_sub.empty:
            ax.set_title(title + " (no data)"); continue
        y = np.arange(len(df_sub))
        colors = ["#27ae60" if pv < 0.05 else "#e74c3c" for pv in df_sub["p_value"]]
        ax.scatter(df_sub["p_value"], y, c=colors, s=60, zorder=3)
        ax.axvline(0.05, color="gray", linestyle="--", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(df_sub["scenario"], fontsize=7)
        ax.set_xlabel("p-value", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlim(-0.01, max(df_sub["p_value"].max() * 1.1, 0.12))
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_8d_pvalue_significance.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Scalability figure
# ============================================================

def generate_scalability_figure(df_scale, out_dir="results/figures"):
    os.makedirs(out_dir, exist_ok=True)
    p_vals = [20, 30, 50]
    p_scenarios = [f"scale_p{pv}" for pv in p_vals]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("§5.7 Scalability: p ∈ {20, 30, 50}\n(N=5000, K=5, hetero=mild)",
                 fontsize=11, fontweight="bold")

    for ax, metric, ylabel, title, ylim in [
        (axes[0], "F1",     "F1 (↑)",     "Skeleton F1 vs p",    (0, 1.1)),
        (axes[1], "SHD",    "SHD (↓)",    "SHD vs p",            None),
        (axes[2], "Runtime","Runtime (s)","Runtime vs p",         None),
    ]:
        data_d = {}
        for m in PLOT_METHODS:
            vals = []
            for sc in p_scenarios:
                row = df_scale[(df_scale["scenario"] == sc) & (df_scale["method"] == m)]
                if row.empty:
                    vals.append((np.nan, np.nan))
                else:
                    vals.append((float(row[f"{metric}_mean"].values[0]),
                                 float(row[f"{metric}_std"].values[0])))
            data_d[m] = vals
        _line_group(ax, p_vals, data_d, ylabel, title, xlabel="p (variables)")
        if ylim:
            ax.set_ylim(*ylim)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig5_7d_scalability_p.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Sachs visualization
# ============================================================

def generate_sachs_figure(sachs_raw_path, out_dir="results/figures"):
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(sachs_raw_path):
        print("  [WARN] Sachs raw CSV not found; skipping Sachs figure.")
        return

    df = pd.read_csv(sachs_raw_path)
    methods_sachs = ["centralized", "local", "naive", "fedpc_consensus", "fedpc_oracle"]
    metrics_sachs = [
        ("SHD",    "SHD (↓ better)"),
        ("F1",     "Skeleton F1 (↑)"),
        ("Dir_F1", "Directional F1 (↑)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("§5 Sachs Real-Data Experiment\n(K=5 clients, 10 splits, ℓ=2, τ=0.5)",
                 fontsize=11, fontweight="bold")

    for ax, (metric_key, ylabel) in zip(axes, metrics_sachs):
        means, stds, valid_methods = [], [], []
        for m in methods_sachs:
            col = f"{m}_{metric_key}"
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            if len(vals) == 0:
                continue
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            valid_methods.append(m)
        if not valid_methods:
            ax.set_title(f"{ylabel}\n(no data)"); continue
        _bar_group(ax, valid_methods, means, stds, ylabel, f"Sachs — {ylabel}",
                   ylim=None if "SHD" in metric_key else (0, 1.25))

    plt.tight_layout()
    path = os.path.join(out_dir, "fig_sachs_results.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =======================
# GRAPH VISUALIZATION
# =======================

def _edge_colors_and_styles(pred_edges, true_edges_skel):
    colors = []
    for e in pred_edges:
        e_u = tuple(sorted(e))
        colors.append("#2ecc71" if e_u in true_edges_skel else "#e74c3c")
    return colors


def _draw_cpdag_panel(ax, cpdag, title, true_skel, true_dir, pos, node_labels=None):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")
    nodes = list(cpdag.nodes())
    labels = node_labels if node_labels else {n: str(n) for n in nodes}

    oriented_nodes = set()
    for u, v in cpdag.edges():
        if not cpdag.has_edge(v, u):
            oriented_nodes.add(u)
            oriented_nodes.add(v)

    node_colors = ["#2c3e50" if n in oriented_nodes else "#7f8c8d" for n in nodes]
    nx.draw_networkx_nodes(cpdag, pos, ax=ax, nodelist=nodes,
                           node_size=380, node_color=node_colors, alpha=0.92)
    nx.draw_networkx_labels(cpdag, pos, ax=ax, labels=labels,
                            font_size=7, font_color="white", font_weight="bold")

    undir_edges = [(u, v) for u, v in cpdag.edges() if cpdag.has_edge(v, u) and u < v]
    undir_colors = _edge_colors_and_styles(undir_edges, true_skel)
    for (u, v), col in zip(undir_edges, undir_colors):
        nx.draw_networkx_edges(cpdag, pos, ax=ax, edgelist=[(u, v)],
                               edge_color=[col], width=1.4, alpha=0.65,
                               arrows=False, style="dashed")

    dir_edges = [(u, v) for u, v in cpdag.edges() if not cpdag.has_edge(v, u)]
    dir_colors = []
    for u, v in dir_edges:
        e_u = tuple(sorted((u, v)))
        is_tp = e_u in true_skel
        is_correct_dir = (u, v) in true_dir
        if is_tp and is_correct_dir:
            dir_colors.append("#27ae60")
        elif is_tp and not is_correct_dir:
            dir_colors.append("#f39c12")
        else:
            dir_colors.append("#e74c3c")
    if dir_edges:
        nx.draw_networkx_edges(cpdag, pos, ax=ax, edgelist=dir_edges,
                               edge_color=dir_colors, width=1.8, alpha=0.85,
                               arrows=True, arrowsize=12,
                               connectionstyle="arc3,rad=0.08")

    pred_skel2 = skeleton_edges_from_graph(cpdag.to_undirected())
    shd, _, _, f1 = compute_metrics(true_skel, pred_skel2)
    n_or, _, _, df1 = orientation_metrics_cpdag(true_dir, cpdag)
    ax.text(0.02, 0.02,
            f"SHD={shd:.0f}  F1={f1:.2f}\nDir-F1={df1:.2f}  n_or={n_or}",
            transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#bdc3c7", alpha=0.85))


def visualize_single_run(
    p=20, N=5000, K=5, edge_prob=0.2, hetero="mild",
    alpha=ALPHA, tau=TAU_DEFAULT, ell=ELL_MAX,
    out_dir="results/graphs",
    n_viz_reps=5
):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(SEED)
    random.seed(SEED)

    B = generate_random_dag(p, edge_prob)
    true_skel = true_skeleton_edges(B)
    true_dir = true_directed_edges(B)

    true_G = nx.DiGraph()
    true_G.add_nodes_from(range(p))
    true_G.add_edges_from(true_dir)

    # FIX BUG #4: Use single pool + partition for visualization too
    X_pool = simulate_linear_sem(B, N, hetero)
    client_data = federated_split(X_pool, K)
    X_global = np.vstack(client_data)

    Gc, seps_c = pc_skeleton_with_sepsets(X_global, alpha, ell)
    cpdag_c = orient_v_structures(Gc, seps_c)

    client_edges, client_seps = local_pc_clients(client_data, alpha, ell)

    w_con = compute_consensus_reliability(client_edges)
    pred_con = weighted_aggregation_with_weights(client_edges, w_con, tau)
    agg_seps_con = aggregate_sepsets(client_seps, w_con, tau)
    Gcon = nx.Graph(); Gcon.add_nodes_from(range(p)); Gcon.add_edges_from(pred_con)
    cpdag_con = orient_v_structures(Gcon, agg_seps_con)

    w_ora = compute_oracle_reliability(client_data, client_edges, true_skel, p)
    pred_ora = weighted_aggregation_with_weights(client_edges, w_ora, tau)
    agg_seps_ora = aggregate_sepsets(client_seps, w_ora, tau)
    Gora = nx.Graph(); Gora.add_nodes_from(range(p)); Gora.add_edges_from(pred_ora)
    cpdag_ora = orient_v_structures(Gora, agg_seps_ora)

    pred_naive = naive_majority_aggregation(client_edges, tau)
    Gn = nx.Graph(); Gn.add_nodes_from(range(p)); Gn.add_edges_from(pred_naive)
    cpdag_n = orient_v_structures(Gn, {})  # Fixed: no sepsets

    pos = nx.kamada_kawai_layout(
        true_G.to_undirected() if true_G.number_of_edges() > 0
        else nx.complete_graph(p)
    )

    # FIG 1 — All methods comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        f"Causal Graph Recovery (single run) — p={p}, N={N}, K={K}, hetero={hetero}",
        fontsize=13, fontweight="bold", y=1.01
    )
    ax = axes[0, 0]
    ax.set_title("True DAG", fontsize=12, fontweight="bold", pad=8)
    ax.axis("off")
    nx.draw_networkx_nodes(true_G, pos, ax=ax, node_size=380,
                           node_color="#2c3e50", alpha=0.92)
    nx.draw_networkx_labels(true_G, pos, ax=ax, font_size=10,
                            font_color="white", font_weight="bold")
    nx.draw_networkx_edges(true_G, pos, ax=ax, edge_color="#27ae60",
                           width=1.8, alpha=0.8, arrows=True, arrowsize=12,
                           connectionstyle="arc3,rad=0.08")
    ax.text(0.02, 0.02, f"Edges={len(true_dir)}  Nodes={p}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#bdc3c7", alpha=0.85))

    panels = [
        (axes[0, 1], cpdag_c,   "Centralized PC"),
        (axes[0, 2], cpdag_con, "FedPC-Consensus (proposed)"),
        (axes[1, 0], cpdag_ora, "FedPC-Oracle (ablation)"),
        (axes[1, 1], cpdag_n,   "FedPC-Naive (no sepsets)"),
    ]
    for ax, cpdag, title in panels:
        _draw_cpdag_panel(ax, cpdag, title, true_skel, true_dir, pos)

    ax_leg = axes[1, 2]
    ax_leg.axis("off")
    legend_items = [
        mpatches.Patch(color="#27ae60", label="Directed: correct edge + direction"),
        mpatches.Patch(color="#f39c12", label="Directed: correct edge, wrong direction"),
        mpatches.Patch(color="#e74c3c", label="Spurious edge (directed or undirected)"),
        mpatches.Patch(color="#2ecc71", label="Undirected: correct edge (dashed)"),
        mpatches.Patch(color="#2c3e50", label="Node with ≥1 oriented edge"),
        mpatches.Patch(color="#7f8c8d", label="Node with no oriented edges"),
    ]
    ax_leg.legend(handles=legend_items, loc="center", fontsize=10,
                  title="Legend", title_fontsize=11, framealpha=0.95)
    plt.tight_layout()
    path1 = os.path.join(out_dir, "fig1_all_methods.png")
    fig.savefig(path1, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path1}")

    # FIG 2 — Consensus vs Naive difference
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Edge Differences: FedPC-Consensus vs FedPC-Naive\n"
                 "(Naive uses no sepsets — Paper §4.5)",
                 fontsize=11, fontweight="bold")

    con_skel   = skeleton_edges_from_graph(cpdag_con.to_undirected())
    naive_skel = skeleton_edges_from_graph(cpdag_n.to_undirected())
    only_con   = con_skel - naive_skel
    only_naive = naive_skel - con_skel

    def draw_diff_panel(ax, cpdag, title, highlight_edges,
                        highlight_color, base_color="#95a5a6"):
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.axis("off")
        nx.draw_networkx_nodes(cpdag, pos, ax=ax, node_size=340,
                               node_color="#2c3e50", alpha=0.85)
        nx.draw_networkx_labels(cpdag, pos, ax=ax, font_size=9,
                                font_color="white", font_weight="bold")
        all_undir = [(u, v) for u, v in cpdag.edges()
                     if cpdag.has_edge(v, u) and u < v]
        all_dir = [(u, v) for u, v in cpdag.edges() if not cpdag.has_edge(v, u)]
        for elist, is_dir in [(all_undir, False), (all_dir, True)]:
            for u, v in elist:
                e_u = tuple(sorted((u, v)))
                col   = highlight_color if e_u in highlight_edges else base_color
                alpha_val = 0.9 if e_u in highlight_edges else 0.25
                w     = 2.0 if e_u in highlight_edges else 0.8
                if is_dir:
                    nx.draw_networkx_edges(cpdag, pos, ax=ax, edgelist=[(u, v)],
                                           edge_color=[col], width=w, alpha=alpha_val,
                                           arrows=True, arrowsize=11,
                                           connectionstyle="arc3,rad=0.08")
                else:
                    nx.draw_networkx_edges(cpdag, pos, ax=ax, edgelist=[(u, v)],
                                           edge_color=[col], width=w, alpha=alpha_val,
                                           arrows=False, style="dashed")
        pred_sk = skeleton_edges_from_graph(cpdag.to_undirected())
        shd2, _, _, f1_2 = compute_metrics(true_skel, pred_sk)
        ax.text(0.02, 0.02, f"SHD={shd2:.0f}  F1={f1_2:.2f}\nhighlighted={len(highlight_edges)}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#bdc3c7", alpha=0.85))

    draw_diff_panel(axes[0], cpdag_con,
                    "FedPC-Consensus\n(blue = unique to Consensus)", only_con, "#3498db")
    draw_diff_panel(axes[1], cpdag_n,
                    "FedPC-Naive\n(orange = unique to Naive)", only_naive, "#e67e22")

    ax = axes[2]
    ax.set_title("Edge Overlap\n(blue=Consensus-only, orange=Naive-only, gray=shared)",
                 fontsize=9, fontweight="bold", pad=6)
    ax.axis("off")
    nx.draw_networkx_nodes(true_G, pos, ax=ax, node_size=340,
                           node_color="#2c3e50", alpha=0.85)
    nx.draw_networkx_labels(true_G, pos, ax=ax, font_size=9,
                            font_color="white", font_weight="bold")
    shared = con_skel & naive_skel
    for e in shared:
        nx.draw_networkx_edges(true_G, pos, ax=ax, edgelist=[e],
                               edge_color=["#95a5a6"], width=1.2, alpha=0.4, arrows=False)
    for e in only_con:
        nx.draw_networkx_edges(true_G, pos, ax=ax, edgelist=[e],
                               edge_color=["#3498db"], width=2.0, alpha=0.85, arrows=False)
    for e in only_naive:
        nx.draw_networkx_edges(true_G, pos, ax=ax, edgelist=[e],
                               edge_color=["#e67e22"], width=2.0, alpha=0.85, arrows=False)
    ax.text(0.02, 0.02,
            f"Shared={len(shared)}  Con-only={len(only_con)}  Naive-only={len(only_naive)}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#bdc3c7", alpha=0.85))

    plt.tight_layout()
    path2 = os.path.join(out_dir, "fig2_consensus_vs_naive_diff.png")
    fig.savefig(path2, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path2}")

    # FIG 3 — Mean ± std bar over n_viz_reps
    print(f"  Computing mean±std over {n_viz_reps} reps for fig3 ...")
    rep_results = [
        run_single_experiment(p=p, N=N, K=K, edge_prob=edge_prob, hetero=hetero,
                              alpha=alpha, tau=tau, ell=ell)
        for _ in range(n_viz_reps)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Metrics Comparison — Mean±Std over {n_viz_reps} runs\n"
        f"(p={p}, N={N}, K={K}, hetero={hetero}, τ={tau})\n"
        "Note: Naive uses no sepsets by design (§4.5)",
        fontsize=10, fontweight="bold"
    )

    methods_p = ["centralized", "local", "naive", "fedpc_consensus", "fedpc_oracle"]
    for ax, metric, ylabel, title, ylim in [
        (axes[0], "SHD",    "SHD (↓ better)", "Structural Hamming Distance", None),
        (axes[1], "F1",     "F1 (↑)",          "Skeleton F1",                 (0, 1.25)),
        (axes[2], "Dir_F1", "Directed F1 (↑)", "Orientation F1",              (0, 1.25)),
    ]:
        shd_v = [np.mean([r[m][metric] for r in rep_results]) for m in methods_p]
        shd_s = [np.std([r[m][metric]  for r in rep_results]) for m in methods_p]
        _bar_group(ax, methods_p, shd_v, shd_s, ylabel, title, ylim)

    plt.tight_layout()
    path3 = os.path.join(out_dir, "fig3_metrics_bar.png")
    fig.savefig(path3, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path3}")


# =======================
# MAIN
# =======================

if __name__ == "__main__":

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/graphs", exist_ok=True)

    all_summary_rows = []
    all_raw_rows = []
    stat_rows = []

    # ---- Define scenarios ----
    scenarios = []
    for K in [2, 5, 10]:
        scenarios.append((f"p20_K{K}", {"p": 20, "K": K}))
    for N in [100, 3000, 5000]:
        scenarios.append((f"p20_N{N}", {"p": 20, "N": N}))
    for phi in [0.1, 0.2]:
        scenarios.append((f"p20_phi{phi}", {"p": 20, "edge_prob": phi}))
    for h in [None, "mild", "strong"]:
        scenarios.append((f"p20_hetero_{h}", {"p": 20, "hetero": h}))
    for tau in [0.3, 0.5]:
        scenarios.append((f"p20_tau{tau}", {"p": 20, "tau": tau}))
    for ell in [1, 2]:
        scenarios.append((f"p20_ell{ell}", {"p": 20, "ell": ell}))
    scenarios.append(("p20_mechanism_shift", {"p": 20, "mechanism_shift": True, "hetero": "mild"}))

    total = len(scenarios)
    print(f"Running {total} main scenarios × {REPS} reps ...")

    for si, (name, kwargs) in enumerate(scenarios):
        t_sc = time.time()
        print(f"  [{si+1}/{total}] {name} ...", end=" ", flush=True)
        reps_res = [run_single_experiment(**kwargs) for _ in range(REPS)]
        elapsed = time.time() - t_sc
        print(f"done in {elapsed:.1f}s")

        all_summary_rows.extend(aggregate_reps(reps_res, name))
        pd.DataFrame(all_summary_rows).to_csv("results/checkpoint_main.csv", index=False)

        for ri, r in enumerate(reps_res):
            for m in METHODS + ["local"]:
                if m not in r:
                    continue
                row = {"scenario": name, "rep": ri, "method": m}
                row.update(r[m])
                all_raw_rows.append(row)

        con_f1    = [r["fedpc_consensus"]["F1"]     for r in reps_res]
        naive_f1  = [r["naive"]["F1"]               for r in reps_res]
        con_df1   = [r["fedpc_consensus"]["Dir_F1"] for r in reps_res]
        naive_df1 = [r["naive"]["Dir_F1"]           for r in reps_res]

        s1 = wilcoxon_ci(con_f1, naive_f1, method="fedpc_consensus vs naive")
        s1["scenario"] = name; s1["metric"] = "F1"
        s2 = wilcoxon_ci(con_df1, naive_df1, method="fedpc_consensus vs naive")
        s2["scenario"] = name; s2["metric"] = "Dir_F1"
        stat_rows.extend([s1, s2])

    # ---- Ablation: ell × tau ----
    print("\nAblation: reuse from main scenarios ...")
    ablation_rows = []
    for ell in [1, 2]:
        for tau in [0.3, 0.5]:
            src = f"p20_ell{ell}" if tau == 0.5 else f"p20_tau{tau}"
            rows_copy = [r.copy() for r in all_summary_rows if r["scenario"] == src]
            for r in rows_copy:
                r["scenario"] = f"ablation_ell{ell}_tau{tau}"
            ablation_rows.extend(rows_copy)

    # ---- Scalability: p ----
    print("\nScalability: p = [20, 30, 50] ...")
    scale_rows = []
    for p_val in [20, 30, 50]:
        name = f"scale_p{p_val}"
        print(f"  p={p_val} ...", end=" ", flush=True)
        t0 = time.time()
        reps_res = [run_single_experiment(p=p_val, N=5000, K=5) for _ in range(REPS_SCALE)]
        print(f"done in {time.time()-t0:.1f}s")
        scale_rows.extend(aggregate_reps(reps_res, name))

    # ---- Sachs ----
    print("\nSachs dataset ...")
    sachs_col_names = None
    if os.path.isdir(SACHS_DIR):
        sachs_raw, sachs_col_names = run_sachs_experiment(
            SACHS_DIR, K=5, tau=TAU_DEFAULT, ell=ELL_MAX, alpha=ALPHA, reps=10
        )
        if sachs_raw:
            sachs_df = pd.DataFrame(sachs_raw)
            sachs_df.to_csv("results/sachs_raw.csv", index=False)
            numeric_cols = sachs_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != "rep"]
            sachs_summary = sachs_df[numeric_cols].agg(["mean", "std"]).T
            sachs_summary.columns = ["mean", "std"]
            sachs_summary = sachs_summary.reset_index().rename(columns={"index": "metric"})
            sachs_summary.to_csv("results/sachs_summary.csv", index=False)
            print("  Sachs done.")
    else:
        print(f"  [WARN] Sachs dir not found: {SACHS_DIR}. Skipping.")

    # =======================
    # SAVE ALL CSVs
    # =======================
    df_main = pd.DataFrame(all_summary_rows)
    df_main.to_csv("results/main_results_summary.csv", index=False)
    print(f"\nSaved: results/main_results_summary.csv  ({len(df_main)} rows)")

    df_raw = pd.DataFrame(all_raw_rows)
    df_raw.to_csv("results/main_results_raw.csv", index=False)
    print(f"Saved: results/main_results_raw.csv  ({len(df_raw)} rows)")

    df_stat = pd.DataFrame(stat_rows)
    df_stat.to_csv("results/statistics_wilcoxon.csv", index=False)
    print(f"Saved: results/statistics_wilcoxon.csv  ({len(df_stat)} rows)")

    df_abl = pd.DataFrame(ablation_rows)
    df_abl.to_csv("results/ablation_ell_tau.csv", index=False)
    print(f"Saved: results/ablation_ell_tau.csv  ({len(df_abl)} rows)")

    df_scale = pd.DataFrame(scale_rows)
    df_scale.to_csv("results/scalability.csv", index=False)
    print(f"Saved: results/scalability.csv  ({len(df_scale)} rows)")

    df_local = df_raw[df_raw["method"] == "local"]
    df_local.to_csv("results/appendix_local_baseline.csv", index=False)
    print(f"Saved: results/appendix_local_baseline.csv")

    # =======================
    # GENERATE ALL FIGURES
    # =======================
    print("\n=== Generating §5 Result Figures ===")
    generate_all_figures(df_main, df_stat, out_dir="results/figures")

    print("\n=== Generating Scalability Figure ===")
    generate_scalability_figure(df_scale, out_dir="results/figures")

    print("\n=== Generating Sachs Figure ===")
    generate_sachs_figure("results/sachs_raw.csv", out_dir="results/figures")

    print("\n=== Generating Graph Visualizations ===")
    visualize_single_run(out_dir="results/graphs", n_viz_reps=5)

    # =======================
    # PRINT SUMMARY TABLE
    # =======================
    print("\n========== MAIN RESULTS SUMMARY ==========")
    pivot_cols = [
        "scenario", "method",
        "SHD_mean", "F1_mean", "Precision_mean", "Recall_mean",
        "Dir_F1_mean", "Dir_Prec_mean", "Dir_Rec_mean",
        "Runtime_mean"
    ]
    print(df_main[[c for c in pivot_cols if c in df_main.columns]].to_string(index=False))

    print("\n========== WILCOXON (FedPC-Consensus vs Naive) ==========")
    stat_cols = ["scenario", "metric", "mean_diff", "ci_lo", "ci_hi",
                 "cohen_d", "wilcoxon_stat", "p_value"]
    print(df_stat[[c for c in stat_cols if c in df_stat.columns]].to_string(index=False))

    print("\nAll results saved to results/")
    print("All figures saved to results/figures/")
    print("Graph visualizations saved to results/graphs/")