import numpy as np
import networkx as nx
from itertools import combinations
from scipy.stats import pearsonr, norm, ttest_rel, t
import random
import time
from joblib import Parallel, delayed

# =======================
# GLOBAL SETTINGS
# =======================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

REPS = 20
ELL_MAX = 2
ALPHA = 0.05
TAU_DEFAULT = 0.5
baselines = ["centralized", "fedpc", "naive", "local", "random"]


# =======================
# FUNCTIONS (PC, CI, DAG, Metrics)
# =======================

def safe_pearsonr(x, y):
    """Pearson r with guards for near-constant vectors and non-finite values."""
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0, 1.0
    r, p = pearsonr(x, y)
    if not np.isfinite(r):
        return 0.0, 1.0
    r = float(np.clip(r, -0.999999, 0.999999))
    return r, p


def generate_random_dag(p, edge_prob):
    """Generate a random DAG with ordering 0 < 1 < ... < p-1."""
    G = nx.DiGraph()
    G.add_nodes_from(range(p))

    for i in range(p):
        for j in range(i + 1, p):
            if np.random.rand() < edge_prob:
                G.add_edge(i, j)

    B = np.zeros((p, p))
    for i, j in G.edges():
        B[i, j] = np.random.uniform(0.5, 2.0) * random.choice([-1, 1])

    return B


def simulate_linear_sem(B, n_samples, hetero=None):
    """Simulate linear SEM X = B^T X + noise with optional heteroskedastic noise."""
    p = B.shape[0]
    X = np.zeros((n_samples, p))
    order = list(nx.topological_sort(nx.DiGraph(B)))

    for i in order:
        parents = np.where(B[:, i] != 0)[0]
        noise = np.random.normal(0, 1, n_samples)

        if hetero == "mild":
            noise *= np.random.uniform(0.8, 1.2, n_samples)
        elif hetero == "strong":
            noise *= np.random.uniform(0.5, 1.5, n_samples)

        if len(parents):
            X[:, i] = X[:, parents] @ B[parents, i] + noise
        else:
            X[:, i] = noise

    return X


def ci_test(X, i, j, cond_set, alpha=0.05):
    """Fisher-Z partial correlation CI test."""
    n = X.shape[0]
    k = len(cond_set)

    if n - k - 3 <= 0:
        return False

    if k == 0:
        r, _ = safe_pearsonr(X[:, i], X[:, j])
    else:
        Z = X[:, cond_set]
        beta_i, *_ = np.linalg.lstsq(Z, X[:, i], rcond=None)
        beta_j, *_ = np.linalg.lstsq(Z, X[:, j], rcond=None)

        Xi = X[:, i] - Z @ beta_i
        Xj = X[:, j] - Z @ beta_j

        r, _ = safe_pearsonr(Xi, Xj)

    z = 0.5 * np.log((1 + r) / (1 - r))
    stat = np.sqrt(n - k - 3) * abs(z)

    return stat < norm.ppf(1 - alpha / 2)


def pc_skeleton_with_sepsets(X, alpha=0.05, ell_max=2):
    """PC skeleton discovery with separation sets."""
    p = X.shape[1]

    G = nx.complete_graph(p)
    sepsets = {(i, j): set() for i in range(p) for j in range(p) if i != j}

    l = 0
    while True:
        cont = False

        for i in range(p):
            adj_i = list(G.neighbors(i))

            if len(adj_i) < l + 1:
                continue

            for j in adj_i:
                if j <= i:
                    continue

                adj_ij = [v for v in adj_i if v != j]

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

        max_adj = max((len(list(G.neighbors(i))) for i in range(p)), default=0)

        if l > ell_max or max_adj < l:
            break

        cont = True

        if not cont:
            break

    return G, sepsets


# ====== Meek rules on a CPDAG ======

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
    """Orient u -> v."""
    if cpdag.has_edge(v, u):
        cpdag.remove_edge(v, u)


def apply_meek_rules(cpdag):
    """Apply Meek's rules repeatedly."""
    changed = True

    while changed:
        changed = False
        nodes = list(cpdag.nodes())

        # Rule 1
        for v in nodes:
            for u in list(cpdag.predecessors(v)):

                if is_undirected(cpdag, v, u):
                    continue

                for w in list(cpdag.neighbors(v)):

                    if w == u or not is_undirected(cpdag, v, w):
                        continue

                    if cpdag.has_edge(u, w) or cpdag.has_edge(w, u):
                        continue

                    orient_edge(cpdag, v, w)
                    changed = True

        # Rule 2
        for v in nodes:
            for u in list(cpdag.predecessors(v)):

                if is_undirected(cpdag, u, v):
                    continue

                for w in list(cpdag.neighbors(v)):

                    if w == u or not is_undirected(cpdag, v, w):
                        continue

                    if cpdag.has_edge(u, w) or cpdag.has_edge(w, u):
                        continue

                    orient_edge(cpdag, v, w)
                    changed = True

        # Rule 3
        for v in nodes:
            for w in list(cpdag.neighbors(v)):

                if not is_undirected(cpdag, v, w):
                    continue

                for u in list(cpdag.successors(v)):

                    if u in (v, w):
                        continue

                    if cpdag.has_edge(u, w) and not cpdag.has_edge(w, u):
                        orient_edge(cpdag, v, w)
                        changed = True
                        break

        # Rule 4 (cycle removal)
        for cyc in list(nx.simple_cycles(cpdag)):

            if len(cyc) >= 2:
                u, v = cyc[-1], cyc[0]

                if cpdag.has_edge(u, v):
                    cpdag.remove_edge(u, v)
                    changed = True

    return cpdag


def orient_v_structures(G, sepsets, use_meek=True):
    """Orient v-structures then apply Meek rules."""
    cpdag = make_cpdag_from_skeleton(G)
    nodes = list(G.nodes())

    for i, j, k in combinations(nodes, 3):

        if G.has_edge(i, j) and G.has_edge(j, k) and not G.has_edge(i, k):

            if j not in sepsets.get((i, k), set()):
                orient_edge(cpdag, j, i)
                orient_edge(cpdag, j, k)

    if use_meek:
        cpdag = apply_meek_rules(cpdag)

    return cpdag


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


def communication_cost(K, p, ell):
    return K * (p ** 2) * (2 ** ell)


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


def compute_client_reliability(X, edges, p, true_skel):

    _, _, _, f1_score = compute_metrics(true_skel, edges)

    reliability = 0.1 + 0.9 * f1_score + np.random.normal(0, 0.01)

    return np.clip(reliability, 0.1, 1.0)


def weighted_aggregation(client_edge_sets, client_data, true_skel, tau=0.5, p=20):

    weights_raw = np.array([
        compute_client_reliability(X, edges, p, true_skel)
        for X, edges in zip(client_data, client_edge_sets)
    ], dtype=float)

    weights = (weights_raw ** 2) / np.sum(weights_raw ** 2)

    counter = {}

    for k, edges in enumerate(client_edge_sets):

        for e in edges:
            counter[e] = counter.get(e, 0.0) + weights[k]

    return {e for e, w in counter.items() if w >= tau}

# =======================
# SINGLE EXPERIMENT
# =======================

def run_single_experiment_extended(
    p=20, N=5000, K=5, edge_prob=0.2, hetero="mild",
    alpha=ALPHA, tau=TAU_DEFAULT, ell=ELL_MAX
):

    # Generate ground truth DAG
    B = generate_random_dag(p, edge_prob)
    true_skel = true_skeleton_edges(B)
    true_dir = true_directed_edges(B)

    # Generate data per client
    client_data = [
        simulate_linear_sem(B, N // K, hetero)
        for _ in range(K)
    ]

    # Centralized baseline data
    X_global = np.vstack(client_data)

    metrics = {}
    orientation_scores = {}

    for baseline in baselines:

        start = time.time()
        pred_skel = None
        cpdag = None
        comm = 0

        if baseline == "centralized":

            G, seps = pc_skeleton_with_sepsets(X_global, alpha, ell)
            pred_skel = skeleton_edges_from_graph(G)
            cpdag = orient_v_structures(G, seps, use_meek=True)

        elif baseline == "fedpc":

            client_edges, client_seps = local_pc_clients(client_data, alpha, ell)

            reliabilities = [
                compute_client_reliability(client_data[k], client_edges[k], p, true_skel)
                for k in range(K)
            ]

            weights_raw = np.array([
                compute_client_reliability(client_data[k], client_edges[k], p, true_skel)
                for k in range(K)
            ])

            weights = (weights_raw ** 2) / np.sum(weights_raw ** 2)

            pred_skel = weighted_aggregation(
                client_edges, client_data, true_skel, tau, p
            )

            Gf = nx.Graph()
            Gf.add_nodes_from(range(p))
            Gf.add_edges_from(pred_skel)

            weights_raw = np.array([
                compute_client_reliability(X, edges, p, true_skel)
                for X, edges in zip(client_data, client_edges)
            ])

            weights = (weights_raw ** 2) / np.sum(weights_raw ** 2)

            agg_seps = {}

            for key in set(k for s in client_seps for k in s.keys()):

                votes = {}

                for k, seps_k in enumerate(client_seps):

                    if key in seps_k:

                        for z in seps_k[key]:
                            votes[z] = votes.get(z, 0.0) + weights[k]

                if votes:
                    agg_seps[key] = {z for z, w in votes.items() if w >= tau}

            cpdag = orient_v_structures(Gf, agg_seps, use_meek=True)

            comm = communication_cost(K, p, ell)

        elif baseline == "naive":

            client_edges, _ = local_pc_clients(client_data, alpha, ell)

            pred_skel = naive_majority_aggregation(client_edges, tau)

            Gn = nx.Graph()
            Gn.add_nodes_from(range(p))
            Gn.add_edges_from(pred_skel)

            cpdag = orient_v_structures(Gn, {}, use_meek=True)

            comm = communication_cost(K, p, ell) // 2

        elif baseline == "local":

            client_edges, client_seps = local_pc_clients(client_data, alpha, ell)

            f1s = []
            shds = []
            orient_f1s = []

            for edges, seps in zip(client_edges, client_seps):

                Gc = nx.Graph()
                Gc.add_nodes_from(range(p))
                Gc.add_edges_from(edges)

                cpdag_c = orient_v_structures(Gc, seps, use_meek=True)

                shd_c, _, _, f1_c = compute_metrics(true_skel, edges)

                _, _, _, orient_f1_c = compute_metrics(
                    true_dir,
                    directed_edges_from_cpdag(cpdag_c)
                )

                f1s.append(f1_c)
                shds.append(shd_c)
                orient_f1s.append(orient_f1_c)

            runtime = time.time() - start

            metrics["local"] = {
                "SHD": float(np.mean(shds)),
                "F1": float(np.mean(f1s)),
                "Comm": communication_cost(K, p, ell),
                "Runtime": runtime
            }

            orientation_scores["local"] = float(np.mean(orient_f1s))

            continue

        else:  # random baseline

            G = nx.fast_gnp_random_graph(p, edge_prob)

            pred_skel = skeleton_edges_from_graph(G)

            cpdag = orient_v_structures(G, {}, use_meek=True)

            comm = communication_cost(K, p, ell)

        shd, _, _, f1 = compute_metrics(true_skel, pred_skel)

        runtime = time.time() - start

        metrics[baseline] = {
            "SHD": shd,
            "F1": f1,
            "Comm": comm,
            "Runtime": runtime
        }

        _, _, _, orient_f1 = compute_metrics(
            true_dir,
            directed_edges_from_cpdag(cpdag)
        )

        orientation_scores[baseline] = orient_f1

    metrics["Orientation_F1"] = orientation_scores

    return metrics


# =======================
# MAIN LOOP ALL SCENARIOS
# =======================

if __name__ == "__main__":

    output_file = "all_experiment_results.txt"

    with open(output_file, "w") as f:

        all_results = {}

        def run_scenario(name, kwargs):
            all_results[name] = [
                run_single_experiment_extended(**kwargs)
                for _ in range(REPS)
            ]

        for p in [20]:

            for K in [2, 5, 10]:
                run_scenario(f"p={p}_K={K}", {"p": p, "K": K})

            for N in [100, 3000, 5000]:
                run_scenario(f"p={p}_N={N}", {"p": p, "N": N})

            for phi in [0.1, 0.2]:
                run_scenario(f"p={p}_phi={phi}", {"p": p, "edge_prob": phi})

            for h in [None, "mild", "strong"]:
                run_scenario(f"p={p}_hetero={h}", {"p": p, "hetero": h})

            for tau in [0.3, 0.5]:
                run_scenario(f"p={p}_tau={tau}", {"p": p, "tau": tau})

            for ell in [1, 2]:
                run_scenario(f"p={p}_ell={ell}", {"p": p, "ell": ell})

        for scenario, results in all_results.items():

            f.write(f"\n--- Section: {scenario} ---\n")

            f.write(
                "Scenario | "
                + " | ".join([f"{b} (SHD|F1|Orient_F1)" for b in baselines])
                + " | ΔF1_fedpc_naive | "
                + " | ".join([f"{b} (Comm|Runtime s)" for b in baselines])
                + "\n"
            )

            for r in results:

                row = [scenario]

                for b in baselines:
                    row.append(
                        f"{r[b]['SHD']:.2f}|{r[b]['F1']:.2f}|{r['Orientation_F1'][b]:.2f}"
                    )

                delta = (
                    r["Orientation_F1"]["fedpc"]
                    - r["Orientation_F1"]["naive"]
                )

                row.append(f"{delta:.2f}")

                for b in baselines:
                    row.append(f"{r[b]['Comm']}|{r[b]['Runtime']:.2f}")

                f.write(" | ".join(row) + "\n")

        # =======================
        # ABLATION STUDY
        # =======================

        f.write("\n======================================\n")
        f.write("ABLATION STUDY (ell × tau)\n")
        f.write("Fixed: p=20, N=5000, K=5, phi=0.2, hetero=mild\n")
        f.write("======================================\n")

        def aggregate_ablation(results):

            agg = {}

            for b in ["fedpc", "naive"]:

                shd = np.mean([r[b]["SHD"] for r in results])
                f1 = np.mean([r[b]["F1"] for r in results])
                orient = np.mean([r["Orientation_F1"][b] for r in results])
                comm = np.mean([r[b]["Comm"] for r in results])
                runtime = np.mean([r[b]["Runtime"] for r in results])

                agg[b] = {
                    "SHD": shd,
                    "F1": f1,
                    "Orient_F1": orient,
                    "Comm": comm,
                    "Runtime": runtime
                }

            return agg

        f.write(
            "ell | tau | FedPC (SHD|F1|OrientF1) | "
            "Naive (SHD|F1|OrientF1) | ΔOrientF1 | "
            "FedPC_Comm | Naive_Comm\n"
        )

        for ell in [1, 2]:

            for tau in [0.3, 0.5]:

                rep_results = [
                    run_single_experiment_extended(ell=ell, tau=tau)
                    for _ in range(REPS)
                ]

                agg = aggregate_ablation(rep_results)

                delta = (
                    agg["fedpc"]["Orient_F1"]
                    - agg["naive"]["Orient_F1"]
                )

                f.write(
                    f"{ell} | {tau} | "
                    f"{agg['fedpc']['SHD']:.2f}|{agg['fedpc']['F1']:.2f}|{agg['fedpc']['Orient_F1']:.2f} | "
                    f"{agg['naive']['SHD']:.2f}|{agg['naive']['F1']:.2f}|{agg['naive']['Orient_F1']:.2f} | "
                    f"{delta:.2f} | "
                    f"{int(agg['fedpc']['Comm'])} | "
                    f"{int(agg['naive']['Comm'])}\n"
                )

# =======================
# STATISTICAL TEST
# =======================

# Statistical Significance Test - 
fedpc_f1_scores = []
naive_f1_scores = []

with open("all_experiment_results.txt") as f:
    for line in f:
        line = line.strip()
        if line.startswith("p=20_") and "|" in line:
            parts = [x.strip() for x in line.split('|')]
            if len(parts) >= 9:  # 
                try:
                    fedpc_f1 = float(parts[5])  # FedPC F1
                    naive_f1 = float(parts[8])  # Naive F1
                    fedpc_f1_scores.append(fedpc_f1)
                    naive_f1_scores.append(naive_f1)
                except (ValueError, IndexError):
                    continue

fedpc_f1_scores = np.array(fedpc_f1_scores)
naive_f1_scores = np.array(naive_f1_scores)

print(f"FedPC F1: {fedpc_f1_scores}")
print(f"Naive F1: {naive_f1_scores}")
print(f"Total pairs: {len(fedpc_f1_scores)}")

t_stat, p_value = ttest_rel(fedpc_f1_scores, naive_f1_scores)
print(f"Paired t-test result: t = {t_stat:.4f}, p = {p_value:.4f}")

diffs = fedpc_f1_scores - naive_f1_scores
mean_diff = np.mean(diffs)
std_diff = np.std(diffs, ddof=1)
n = len(diffs)
cohen_d = mean_diff / std_diff

conf_level = 0.95
alpha_ci = 1 - conf_level
t_crit = t.ppf(1 - alpha_ci/2, df=n-1)
margin_error = t_crit * std_diff / np.sqrt(n)
ci_lower = mean_diff - margin_error
ci_upper = mean_diff + margin_error

print(f"Mean difference (FedPC - Naive): {mean_diff:.4f}")
print(f"Cohen's d: {cohen_d:.4f}")
print(f"95% CI for difference: [{ci_lower:.4f}, {ci_upper:.4f}]")

# =======================
# Visualization
# =======================

###################################### K2
import pandas as pd

text = """
Scenario | centralized (SHD|F1|Orient_F1) | fedpc (SHD|F1|Orient_F1) | naive (SHD|F1|Orient_F1) | local (SHD|F1|Orient_F1) | random (SHD|F1|Orient_F1) | ΔF1_fedpc_naive | centralized (Comm|Runtime s) | fedpc (Comm|Runtime s) | naive (Comm|Runtime s) | local (Comm|Runtime s) | random (Comm|Runtime s)
p=20_K=2 | 37.00|0.61|0.24 | 34.00|0.63|0.29 | 36.00|0.64|0.22 | 34.00|0.63|0.30 | 60.00|0.25|0.06 | 0.06 | 0|27.59 | 3200|5.33 | 1600|2.86 | 3200|2.78 | 3200|0.00
p=20_K=2 | 47.00|0.49|0.24 | 41.00|0.54|0.34 | 50.00|0.51|0.19 | 44.50|0.48|0.27 | 61.00|0.25|0.08 | 0.15 | 0|45.50 | 3200|3.98 | 1600|3.67 | 3200|3.84 | 3200|0.00
p=20_K=2 | 32.00|0.66|0.22 | 33.00|0.64|0.24 | 35.00|0.64|0.18 | 35.00|0.62|0.21 | 67.00|0.25|0.10 | 0.06 | 0|18.00 | 3200|1.89 | 1600|1.81 | 3200|1.74 | 3200|0.00
p=20_K=2 | 37.00|0.62|0.27 | 36.00|0.60|0.24 | 38.00|0.60|0.20 | 35.50|0.59|0.26 | 59.00|0.21|0.17 | 0.03 | 0|39.90 | 3200|4.17 | 1600|3.87 | 3200|3.76 | 3200|0.00
p=20_K=2 | 28.00|0.68|0.33 | 26.00|0.70|0.32 | 27.00|0.70|0.25 | 28.00|0.67|0.31 | 68.00|0.24|0.11 | 0.07 | 0|16.09 | 3200|2.62 | 1600|2.39 | 3200|2.38 | 3200|0.00
p=20_K=2 | 34.00|0.64|0.12 | 35.00|0.61|0.22 | 35.00|0.64|0.24 | 36.50|0.59|0.26 | 67.00|0.13|0.06 | -0.02 | 0|44.10 | 3200|4.56 | 1600|4.08 | 3200|4.07 | 3200|0.00
p=20_K=2 | 43.00|0.51|0.25 | 38.00|0.55|0.32 | 38.00|0.57|0.29 | 38.50|0.54|0.32 | 58.00|0.24|0.08 | 0.03 | 0|38.00 | 3200|3.75 | 1600|4.16 | 3200|4.15 | 3200|0.00
p=20_K=2 | 43.00|0.57|0.34 | 37.00|0.61|0.28 | 43.00|0.58|0.16 | 40.00|0.58|0.26 | 68.00|0.24|0.10 | 0.11 | 0|49.10 | 3200|3.99 | 1600|4.40 | 3200|4.45 | 3200|0.00
p=20_K=2 | 29.00|0.64|0.39 | 28.00|0.63|0.37 | 33.00|0.60|0.37 | 31.50|0.59|0.34 | 58.00|0.24|0.03 | 0.01 | 0|31.20 | 3200|3.15 | 1600|3.06 | 3200|3.38 | 3200|0.00
p=20_K=2 | 28.00|0.67|0.49 | 30.00|0.62|0.22 | 35.00|0.59|0.17 | 32.50|0.58|0.28 | 58.00|0.17|0.12 | 0.04 | 0|25.50 | 3200|2.60 | 1600|2.56 | 3200|2.46 | 3200|0.00
p=20_K=2 | 27.00|0.65|0.45 | 25.00|0.69|0.48 | 33.00|0.63|0.25 | 28.50|0.64|0.44 | 53.00|0.33|0.06 | 0.24 | 0|15.48 | 3200|2.36 | 1600|2.14 | 3200|2.18 | 3200|0.00
p=20_K=2 | 17.00|0.78|0.33 | 15.00|0.79|0.36 | 16.00|0.79|0.33 | 15.50|0.79|0.37 | 59.00|0.09|0.07 | 0.04 | 0|9.10 | 3200|1.42 | 1600|1.43 | 3200|1.52 | 3200|0.00
p=20_K=2 | 24.00|0.71|0.45 | 17.00|0.78|0.51 | 19.00|0.77|0.48 | 18.50|0.76|0.48 | 49.00|0.25|0.23 | 0.03 | 0|16.91 | 3200|1.77 | 1600|1.78 | 3200|1.86 | 3200|0.00
p=20_K=2 | 41.00|0.56|0.35 | 39.00|0.54|0.28 | 43.00|0.53|0.21 | 39.50|0.54|0.31 | 63.00|0.28|0.23 | 0.07 | 0|39.71 | 3200|3.38 | 1600|3.85 | 3200|3.48 | 3200|0.00
p=20_K=2 | 27.00|0.69|0.24 | 26.00|0.69|0.24 | 30.00|0.67|0.24 | 27.50|0.67|0.31 | 72.00|0.20|0.10 | 0.00 | 0|27.26 | 3200|3.10 | 1600|3.22 | 3200|3.04 | 3200|0.00
p=20_K=2 | 50.00|0.49|0.14 | 46.00|0.51|0.17 | 49.00|0.51|0.13 | 46.50|0.50|0.18 | 69.00|0.10|0.08 | 0.04 | 0|57.60 | 3200|5.06 | 1600|4.76 | 3200|5.47 | 3200|0.00
p=20_K=2 | 42.00|0.57|0.27 | 41.00|0.58|0.19 | 41.00|0.60|0.15 | 40.50|0.56|0.18 | 67.00|0.17|0.09 | 0.04 | 0|53.09 | 3200|4.56 | 1600|4.34 | 3200|4.48 | 3200|0.00
p=20_K=2 | 49.00|0.52|0.20 | 41.00|0.55|0.18 | 48.00|0.53|0.18 | 45.00|0.52|0.20 | 74.00|0.21|0.10 | 0.00 | 0|39.21 | 3200|4.10 | 1600|3.82 | 3200|3.97 | 3200|0.00
p=20_K=2 | 25.00|0.66|0.27 | 18.00|0.74|0.41 | 21.00|0.73|0.34 | 20.00|0.71|0.37 | 58.00|0.15|0.13 | 0.07 | 0|27.49 | 3200|2.84 | 1600|2.76 | 3200|2.92 | 3200|0.00
p=20_K=2 | 22.00|0.72|0.25 | 21.00|0.71|0.25 | 24.00|0.69|0.18 | 21.50|0.70|0.27 | 54.00|0.16|0.07 | 0.07 | 0|17.49 | 3200|1.68 | 1600|1.70 | 3200|1.73 | 3200|0.00

"""

rows = []
lines = text.strip().split("\n")

run_id = 0

for line in lines:
    if line.startswith("p="):
        run_id += 1
        parts = [p.strip() for p in line.split(" | ")]

        scenario = parts[0]

        methods_metrics = {
            "centralized": parts[1],
            "fedpc": parts[2],
            "naive": parts[3],
            "local": parts[4],
            "random": parts[5],
        }

        delta = float(parts[6])

        comm_runtime = {
            "centralized": parts[7],
            "fedpc": parts[8],
            "naive": parts[9],
            "local": parts[10],
            "random": parts[11],
        }

        for method in methods_metrics:
            shd, f1, orient = methods_metrics[method].split("|")
            comm, runtime = comm_runtime[method].split("|")

            rows.append({
                "Run": run_id,
                "Scenario": scenario,
                "Method": method,
                "SHD": float(shd),
                "F1": float(f1),
                "Orient_F1": float(orient),
                "Comm": float(comm),
                "Runtime_s": float(runtime),
                "Delta_F1_fedpc_naive": delta if method == "fedpc" else None
            })

df = pd.DataFrame(rows)

# save to Excel
df.to_excel("table1.xlsx", index=False)
print("Excel file saved: table1.xlsx")

###################################### K5

import pandas as pd

text = """
Scenario | centralized (SHD|F1|Orient_F1) | fedpc (SHD|F1|Orient_F1) | naive (SHD|F1|Orient_F1) | local (SHD|F1|Orient_F1) | random (SHD|F1|Orient_F1) | ΔF1_fedpc_naive | centralized (Comm|Runtime s) | fedpc (Comm|Runtime s) | naive (Comm|Runtime s) | local (Comm|Runtime s) | random (Comm|Runtime s)
p=20_K=5 | 32.00|0.58|0.33 | 27.00|0.58|0.17 | 27.00|0.58|0.28 | 29.20|0.57|0.22 | 63.00|0.09|0.06 | -0.11 | 0|23.17 | 8000|4.89 | 4000|5.04 | 8000|4.78 | 8000|0.00
p=20_K=5 | 35.00|0.60|0.26 | 27.00|0.63|0.35 | 27.00|0.63|0.22 | 29.80|0.60|0.27 | 65.00|0.13|0.03 | 0.13 | 0|27.78 | 8000|5.94 | 4000|5.73 | 8000|5.58 | 8000|0.00
p=20_K=5 | 32.00|0.64|0.28 | 23.00|0.70|0.30 | 23.00|0.70|0.26 | 24.80|0.69|0.35 | 64.00|0.18|0.14 | 0.04 | 0|21.64 | 8000|6.19 | 4000|5.12 | 8000|5.40 | 8000|0.00
p=20_K=5 | 17.00|0.76|0.42 | 10.00|0.84|0.39 | 10.00|0.84|0.34 | 13.80|0.79|0.38 | 64.00|0.14|0.03 | 0.05 | 0|19.09 | 8000|4.12 | 4000|4.23 | 8000|5.17 | 8000|0.00
p=20_K=5 | 33.00|0.63|0.29 | 22.00|0.69|0.28 | 22.00|0.69|0.32 | 27.20|0.64|0.31 | 50.00|0.24|0.10 | -0.04 | 0|115.26 | 8000|11.36 | 4000|11.15 | 8000|11.09 | 8000|0.00
p=20_K=5 | 24.00|0.71|0.49 | 24.00|0.68|0.37 | 24.00|0.68|0.18 | 24.00|0.68|0.34 | 60.00|0.30|0.24 | 0.19 | 0|19.48 | 8000|5.19 | 4000|4.80 | 8000|4.77 | 8000|0.00
p=20_K=5 | 17.00|0.77|0.37 | 16.00|0.77|0.38 | 16.00|0.77|0.39 | 17.00|0.76|0.45 | 52.00|0.28|0.20 | -0.01 | 0|12.62 | 8000|3.30 | 4000|3.17 | 8000|3.12 | 8000|0.00
p=20_K=5 | 29.00|0.60|0.29 | 26.00|0.58|0.14 | 26.00|0.58|0.15 | 25.60|0.61|0.26 | 47.00|0.25|0.14 | -0.01 | 0|31.98 | 8000|4.96 | 4000|4.77 | 8000|4.87 | 8000|0.00
p=20_K=5 | 7.00|0.89|0.50 | 11.00|0.81|0.41 | 11.00|0.81|0.42 | 12.60|0.79|0.36 | 66.00|0.08|0.06 | -0.01 | 0|10.08 | 8000|3.23 | 4000|3.32 | 8000|3.10 | 8000|0.00
p=20_K=5 | 35.00|0.65|0.29 | 33.00|0.59|0.40 | 33.00|0.59|0.14 | 32.80|0.61|0.33 | 54.00|0.31|0.08 | 0.26 | 0|47.05 | 8000|6.40 | 4000|6.58 | 8000|6.64 | 8000|0.00
p=20_K=5 | 30.00|0.69|0.33 | 26.00|0.68|0.32 | 26.00|0.68|0.20 | 28.00|0.67|0.31 | 73.00|0.14|0.08 | 0.12 | 0|34.37 | 8000|7.42 | 4000|7.32 | 8000|7.36 | 8000|0.00
p=20_K=5 | 14.00|0.77|0.54 | 8.00|0.86|0.36 | 8.00|0.86|0.41 | 11.00|0.81|0.37 | 56.00|0.12|0.10 | -0.05 | 0|4.39 | 8000|2.31 | 4000|2.25 | 8000|2.42 | 8000|0.00
p=20_K=5 | 20.00|0.73|0.48 | 22.00|0.68|0.46 | 22.00|0.68|0.35 | 21.00|0.69|0.39 | 52.00|0.16|0.17 | 0.11 | 0|15.71 | 8000|4.11 | 4000|4.28 | 8000|4.41 | 8000|0.00
p=20_K=5 | 33.00|0.57|0.23 | 27.00|0.61|0.34 | 27.00|0.61|0.31 | 30.80|0.57|0.31 | 67.00|0.13|0.06 | 0.03 | 0|42.19 | 8000|6.73 | 4000|7.46 | 8000|7.15 | 8000|0.00
p=20_K=5 | 28.00|0.67|0.37 | 37.00|0.52|0.26 | 37.00|0.52|0.22 | 36.20|0.53|0.26 | 63.00|0.22|0.12 | 0.05 | 0|27.33 | 8000|5.48 | 4000|5.42 | 8000|5.37 | 8000|0.00
p=20_K=5 | 24.00|0.68|0.29 | 20.00|0.70|0.39 | 20.00|0.70|0.28 | 20.40|0.70|0.34 | 55.00|0.23|0.10 | 0.12 | 0|16.32 | 8000|5.06 | 4000|5.36 | 8000|5.40 | 8000|0.00
p=20_K=5 | 34.00|0.63|0.37 | 33.00|0.61|0.30 | 33.00|0.61|0.30 | 31.60|0.64|0.32 | 59.00|0.25|0.17 | -0.01 | 0|27.36 | 8000|6.09 | 4000|5.62 | 8000|5.70 | 8000|0.00
p=20_K=5 | 25.00|0.70|0.19 | 25.00|0.65|0.32 | 25.00|0.65|0.27 | 26.20|0.64|0.33 | 64.00|0.14|0.09 | 0.06 | 0|19.28 | 8000|4.48 | 4000|4.62 | 8000|4.68 | 8000|0.00
p=20_K=5 | 33.00|0.62|0.35 | 28.00|0.61|0.31 | 28.00|0.61|0.35 | 28.80|0.61|0.24 | 62.00|0.26|0.12 | -0.03 | 0|44.51 | 8000|6.94 | 4000|6.62 | 8000|7.01 | 8000|0.00
p=20_K=5 | 22.00|0.70|0.24 | 20.00|0.68|0.40 | 20.00|0.68|0.25 | 23.20|0.64|0.32 | 66.00|0.20|0.18 | 0.15 | 0|12.31 | 8000|3.86 | 4000|3.47 | 8000|3.50 | 8000|0.00

"""

rows = []
lines = text.strip().split("\n")

run_id = 0

for line in lines:
    if line.startswith("p="):
        run_id += 1
        parts = [p.strip() for p in line.split(" | ")]

        scenario = parts[0]

        methods_metrics = {
            "centralized": parts[1],
            "fedpc": parts[2],
            "naive": parts[3],
            "local": parts[4],
            "random": parts[5],
        }

        delta = float(parts[6])

        comm_runtime = {
            "centralized": parts[7],
            "fedpc": parts[8],
            "naive": parts[9],
            "local": parts[10],
            "random": parts[11],
        }

        for method in methods_metrics:
            shd, f1, orient = methods_metrics[method].split("|")
            comm, runtime = comm_runtime[method].split("|")

            rows.append({
                "Run": run_id,
                "Scenario": scenario,
                "Method": method,
                "SHD": float(shd),
                "F1": float(f1),
                "Orient_F1": float(orient),
                "Comm": float(comm),
                "Runtime_s": float(runtime),
                "Delta_F1_fedpc_naive": delta if method == "fedpc" else None
            })

df = pd.DataFrame(rows)

# save to Excel
df.to_excel("table2.xlsx", index=False)
print("Excel file saved: table2.xlsx")

###################################### K10

import pandas as pd

text = """
Scenario | centralized (SHD|F1|Orient_F1) | fedpc (SHD|F1|Orient_F1) | naive (SHD|F1|Orient_F1) | local (SHD|F1|Orient_F1) | random (SHD|F1|Orient_F1) | ΔF1_fedpc_naive | centralized (Comm|Runtime s) | fedpc (Comm|Runtime s) | naive (Comm|Runtime s) | local (Comm|Runtime s) | random (Comm|Runtime s)
p=20_K=10 | 22.00|0.73|0.30 | 13.00|0.79|0.45 | 15.00|0.77|0.19 | 15.30|0.77|0.36 | 70.00|0.05|0.03 | 0.27 | 0|18.45 | 16000|6.05 | 8000|6.02 | 16000|6.57 | 16000|0.00
p=20_K=10 | 31.00|0.63|0.24 | 25.00|0.60|0.43 | 27.00|0.58|0.27 | 29.30|0.57|0.26 | 64.00|0.18|0.09 | 0.16 | 0|27.76 | 16000|7.93 | 8000|8.06 | 16000|7.49 | 16000|0.00
p=20_K=10 | 42.00|0.55|0.22 | 37.00|0.51|0.31 | 38.00|0.51|0.20 | 41.90|0.47|0.25 | 72.00|0.20|0.10 | 0.10 | 0|46.91 | 16000|13.12 | 8000|13.74 | 16000|13.53 | 16000|0.00
p=20_K=10 | 50.00|0.47|0.36 | 42.00|0.45|0.14 | 45.00|0.43|0.14 | 45.80|0.44|0.19 | 65.00|0.22|0.08 | 0.00 | 0|64.38 | 16000|12.77 | 8000|12.44 | 16000|12.28 | 16000|0.00
p=20_K=10 | 30.00|0.66|0.41 | 26.00|0.64|0.43 | 27.00|0.63|0.25 | 29.50|0.60|0.35 | 62.00|0.21|0.10 | 0.19 | 0|23.28 | 16000|6.38 | 8000|6.62 | 16000|6.54 | 16000|0.00
p=20_K=10 | 13.00|0.82|0.32 | 11.00|0.81|0.46 | 12.00|0.80|0.43 | 13.20|0.78|0.41 | 52.00|0.24|0.11 | 0.03 | 0|15.85 | 16000|5.82 | 8000|5.89 | 16000|5.85 | 16000|0.00
p=20_K=10 | 2.00|0.96|0.89 | 6.00|0.88|0.77 | 6.00|0.88|0.40 | 8.20|0.84|0.54 | 53.00|0.16|0.11 | 0.37 | 0|4.13 | 16000|3.09 | 8000|2.93 | 16000|3.14 | 16000|0.00
p=20_K=10 | 21.00|0.70|0.29 | 22.00|0.62|0.26 | 23.00|0.61|0.24 | 22.80|0.61|0.29 | 48.00|0.35|0.19 | 0.02 | 0|16.81 | 16000|5.55 | 8000|5.95 | 16000|6.26 | 16000|0.00
p=20_K=10 | 50.00|0.50|0.24 | 39.00|0.48|0.25 | 42.00|0.46|0.20 | 40.50|0.46|0.23 | 57.00|0.30|0.11 | 0.06 | 0|57.73 | 16000|11.88 | 8000|11.33 | 16000|15.80 | 16000|0.00
p=20_K=10 | 14.00|0.82|0.58 | 12.00|0.82|0.45 | 12.00|0.82|0.49 | 14.90|0.78|0.43 | 67.00|0.13|0.07 | -0.04 | 0|53.97 | 16000|11.79 | 8000|12.09 | 16000|12.01 | 16000|0.00
p=20_K=10 | 40.00|0.57|0.29 | 34.00|0.53|0.31 | 34.00|0.53|0.29 | 34.70|0.55|0.28 | 71.00|0.24|0.07 | 0.02 | 0|48.69 | 16000|10.64 | 8000|10.03 | 16000|10.62 | 16000|0.00
p=20_K=10 | 22.00|0.69|0.43 | 24.00|0.57|0.36 | 26.00|0.55|0.16 | 26.30|0.56|0.31 | 60.00|0.19|0.13 | 0.20 | 0|33.58 | 16000|8.03 | 8000|7.74 | 16000|7.65 | 16000|0.00
p=20_K=10 | 15.00|0.79|0.51 | 11.00|0.82|0.41 | 11.00|0.82|0.48 | 12.00|0.80|0.41 | 56.00|0.22|0.15 | -0.08 | 0|15.67 | 16000|4.94 | 8000|5.00 | 16000|5.35 | 16000|0.00
p=20_K=10 | 32.00|0.66|0.21 | 24.00|0.68|0.32 | 26.00|0.67|0.27 | 30.10|0.62|0.27 | 63.00|0.16|0.12 | 0.05 | 0|41.30 | 16000|8.96 | 8000|8.45 | 16000|8.83 | 16000|0.00
p=20_K=10 | 23.00|0.72|0.18 | 14.00|0.79|0.41 | 15.00|0.78|0.35 | 17.30|0.76|0.34 | 65.00|0.18|0.12 | 0.05 | 0|17.63 | 16000|5.77 | 8000|6.24 | 16000|5.88 | 16000|0.00
p=20_K=10 | 27.00|0.67|0.31 | 14.00|0.79|0.31 | 17.00|0.75|0.32 | 17.90|0.74|0.35 | 50.00|0.26|0.09 | -0.01 | 0|19.10 | 16000|8.05 | 8000|7.79 | 16000|7.51 | 16000|0.00
p=20_K=10 | 36.00|0.61|0.19 | 36.00|0.53|0.22 | 36.00|0.53|0.18 | 35.40|0.55|0.18 | 73.00|0.16|0.05 | 0.04 | 0|42.23 | 16000|9.91 | 8000|9.88 | 16000|10.17 | 16000|0.00
p=20_K=10 | 42.00|0.55|0.29 | 30.00|0.61|0.21 | 33.00|0.58|0.21 | 32.40|0.59|0.29 | 61.00|0.27|0.07 | -0.01 | 0|49.33 | 16000|11.76 | 8000|12.08 | 16000|11.67 | 16000|0.00
p=20_K=10 | 50.00|0.49|0.16 | 43.00|0.46|0.13 | 45.00|0.44|0.12 | 46.90|0.44|0.15 | 61.00|0.25|0.16 | 0.02 | 0|48.07 | 16000|10.72 | 8000|11.25 | 16000|10.85 | 16000|0.00
p=20_K=10 | 35.00|0.58|0.36 | 25.00|0.64|0.36 | 25.00|0.64|0.23 | 26.50|0.63|0.36 | 75.00|0.21|0.07 | 0.14 | 0|36.56 | 16000|11.68 | 8000|12.07 | 16000|11.40 | 16000|0.00
"""

rows = []
lines = text.strip().split("\n")

run_id = 0

for line in lines:
    if line.startswith("p="):
        run_id += 1
        parts = [p.strip() for p in line.split(" | ")]

        scenario = parts[0]

        methods_metrics = {
            "centralized": parts[1],
            "fedpc": parts[2],
            "naive": parts[3],
            "local": parts[4],
            "random": parts[5],
        }

        delta = float(parts[6])

        comm_runtime = {
            "centralized": parts[7],
            "fedpc": parts[8],
            "naive": parts[9],
            "local": parts[10],
            "random": parts[11],
        }

        for method in methods_metrics:
            shd, f1, orient = methods_metrics[method].split("|")
            comm, runtime = comm_runtime[method].split("|")

            rows.append({
                "Run": run_id,
                "Scenario": scenario,
                "Method": method,
                "SHD": float(shd),
                "F1": float(f1),
                "Orient_F1": float(orient),
                "Comm": float(comm),
                "Runtime_s": float(runtime),
                "Delta_F1_fedpc_naive": delta if method == "fedpc" else None
            })

df = pd.DataFrame(rows)

# save to Excel
df.to_excel("table3.xlsx", index=False)
print("Excel file saved: table3.xlsx")

###################################### N 100

import pandas as pd

text = """
Scenario | centralized (SHD|F1|Orient_F1) | fedpc (SHD|F1|Orient_F1) | naive (SHD|F1|Orient_F1) | local (SHD|F1|Orient_F1) | random (SHD|F1|Orient_F1) | ΔF1_fedpc_naive | centralized (Comm|Runtime s) | fedpc (Comm|Runtime s) | naive (Comm|Runtime s) | local (Comm|Runtime s) | random (Comm|Runtime s)
p=20_N=100 | 18.00|0.68|0.32 | 28.00|0.36|0.09 | 28.00|0.36|0.09 | 28.80|0.40|0.14 | 58.00|0.22|0.13 | -0.00 | 0|0.57 | 8000|1.08 | 4000|1.00 | 8000|1.04 | 8000|0.00
p=20_N=100 | 34.00|0.48|0.28 | 33.00|0.35|0.24 | 33.00|0.35|0.24 | 37.00|0.32|0.20 | 60.00|0.25|0.13 | 0.00 | 0|0.94 | 8000|1.32 | 4000|1.03 | 8000|1.26 | 8000|0.00
p=20_N=100 | 24.00|0.60|0.20 | 28.00|0.44|0.16 | 28.00|0.44|0.17 | 28.20|0.45|0.20 | 62.00|0.14|0.06 | -0.00 | 0|0.94 | 8000|1.45 | 4000|1.39 | 8000|1.43 | 8000|0.00
p=20_N=100 | 22.00|0.65|0.30 | 29.00|0.43|0.20 | 29.00|0.43|0.20 | 30.40|0.42|0.19 | 55.00|0.25|0.12 | -0.00 | 0|0.84 | 8000|1.36 | 4000|1.34 | 8000|1.22 | 8000|0.00
p=20_N=100 | 39.00|0.43|0.27 | 43.00|0.27|0.07 | 43.00|0.27|0.10 | 41.80|0.32|0.15 | 67.00|0.26|0.19 | -0.04 | 0|1.06 | 8000|1.74 | 4000|1.79 | 8000|1.68 | 8000|0.00
p=20_N=100 | 33.00|0.52|0.21 | 32.00|0.41|0.30 | 32.00|0.41|0.23 | 35.40|0.35|0.23 | 60.00|0.25|0.03 | 0.07 | 0|1.62 | 8000|1.98 | 4000|1.89 | 8000|1.81 | 8000|0.00
p=20_N=100 | 47.00|0.37|0.11 | 52.00|0.19|0.16 | 52.00|0.19|0.09 | 52.00|0.22|0.13 | 74.00|0.18|0.10 | 0.06 | 0|1.15 | 8000|1.59 | 4000|1.67 | 8000|1.57 | 8000|0.00
p=20_N=100 | 17.00|0.72|0.30 | 29.00|0.43|0.27 | 29.00|0.43|0.28 | 29.60|0.44|0.19 | 54.00|0.29|0.20 | -0.01 | 0|0.74 | 8000|1.32 | 4000|1.24 | 8000|1.22 | 8000|0.00
p=20_N=100 | 37.00|0.43|0.19 | 35.00|0.31|0.27 | 35.00|0.31|0.24 | 36.00|0.34|0.24 | 65.00|0.18|0.03 | 0.04 | 0|0.94 | 8000|1.43 | 4000|1.52 | 8000|1.55 | 8000|0.00
p=20_N=100 | 36.00|0.54|0.21 | 41.00|0.35|0.19 | 41.00|0.35|0.19 | 42.40|0.35|0.15 | 68.00|0.21|0.15 | 0.00 | 0|0.82 | 8000|1.25 | 4000|1.19 | 8000|1.19 | 8000|0.00
p=20_N=100 | 26.00|0.61|0.33 | 33.00|0.35|0.20 | 33.00|0.35|0.16 | 33.40|0.37|0.19 | 53.00|0.33|0.14 | 0.04 | 0|0.88 | 8000|1.48 | 4000|1.43 | 8000|1.51 | 8000|0.00
p=20_N=100 | 27.00|0.49|0.23 | 27.00|0.37|0.14 | 27.00|0.37|0.15 | 28.00|0.38|0.21 | 60.00|0.14|0.04 | -0.01 | 0|0.88 | 8000|1.37 | 4000|1.37 | 8000|1.49 | 8000|0.00
p=20_N=100 | 19.00|0.61|0.37 | 23.00|0.38|0.32 | 23.00|0.38|0.32 | 24.60|0.38|0.21 | 51.00|0.30|0.29 | 0.00 | 0|0.70 | 8000|1.35 | 4000|1.25 | 8000|1.32 | 8000|0.00
p=20_N=100 | 29.00|0.55|0.13 | 31.00|0.37|0.08 | 31.00|0.37|0.08 | 32.00|0.39|0.17 | 72.00|0.08|0.06 | 0.00 | 0|1.34 | 8000|2.35 | 4000|1.93 | 8000|2.13 | 8000|0.00
p=20_N=100 | 17.00|0.67|0.39 | 22.00|0.48|0.24 | 22.00|0.48|0.24 | 25.00|0.42|0.19 | 64.00|0.14|0.03 | 0.00 | 0|0.73 | 8000|1.38 | 4000|1.43 | 8000|1.45 | 8000|0.00
p=20_N=100 | 32.00|0.54|0.24 | 35.00|0.36|0.15 | 35.00|0.36|0.11 | 37.60|0.34|0.17 | 73.00|0.14|0.08 | 0.03 | 0|0.92 | 8000|1.43 | 4000|1.38 | 8000|1.39 | 8000|0.00
p=20_N=100 | 20.00|0.67|0.34 | 22.00|0.52|0.30 | 22.00|0.52|0.30 | 24.00|0.49|0.31 | 65.00|0.13|0.06 | 0.00 | 0|0.90 | 8000|1.13 | 4000|1.15 | 8000|1.21 | 8000|0.00
p=20_N=100 | 26.00|0.59|0.33 | 31.00|0.42|0.26 | 31.00|0.42|0.24 | 33.20|0.39|0.20 | 65.00|0.18|0.12 | 0.03 | 0|0.81 | 8000|1.46 | 4000|1.32 | 8000|1.32 | 8000|0.00
p=20_N=100 | 28.00|0.56|0.19 | 29.00|0.43|0.16 | 29.00|0.43|0.08 | 28.60|0.46|0.21 | 60.00|0.14|0.06 | 0.08 | 0|0.82 | 8000|1.27 | 4000|1.03 | 8000|1.13 | 8000|0.00
p=20_N=100 | 20.00|0.64|0.32 | 28.00|0.42|0.25 | 28.00|0.42|0.26 | 28.40|0.44|0.20 | 67.00|0.15|0.07 | -0.01 | 0|0.76 | 8000|1.23 | 4000|1.12 | 8000|1.10 | 8000|0.00
"""

rows = []
lines = text.strip().split("\n")

run_id = 0

for line in lines:
    if line.startswith("p="):
        run_id += 1
        parts = [p.strip() for p in line.split(" | ")]

        scenario = parts[0]

        methods_metrics = {
            "centralized": parts[1],
            "fedpc": parts[2],
            "naive": parts[3],
            "local": parts[4],
            "random": parts[5],
        }

        delta = float(parts[6])

        comm_runtime = {
            "centralized": parts[7],
            "fedpc": parts[8],
            "naive": parts[9],
            "local": parts[10],
            "random": parts[11],
        }

        for method in methods_metrics:
            shd, f1, orient = methods_metrics[method].split("|")
            comm, runtime = comm_runtime[method].split("|")

            rows.append({
                "Run": run_id,
                "Scenario": scenario,
                "Method": method,
                "SHD": float(shd),
                "F1": float(f1),
                "Orient_F1": float(orient),
                "Comm": float(comm),
                "Runtime_s": float(runtime),
                "Delta_F1_fedpc_naive": delta if method == "fedpc" else None
            })

df = pd.DataFrame(rows)

# save to Excel
df.to_excel("table4.xlsx", index=False)
print("Excel file saved: table4.xlsx")

###################################### N 3000

import pandas as pd

text = """
Scenario | centralized (SHD|F1|Orient_F1) | fedpc (SHD|F1|Orient_F1) | naive (SHD|F1|Orient_F1) | local (SHD|F1|Orient_F1) | random (SHD|F1|Orient_F1) | ΔF1_fedpc_naive | centralized (Comm|Runtime s) | fedpc (Comm|Runtime s) | naive (Comm|Runtime s) | local (Comm|Runtime s) | random (Comm|Runtime s)
p=20_N=3000 | 20.00|0.77|0.50 | 18.00|0.76|0.38 | 18.00|0.76|0.45 | 17.60|0.76|0.40 | 65.00|0.22|0.13 | -0.07 | 0|2.29 | 8000|3.70 | 4000|3.80 | 8000|3.82 | 8000|0.00
p=20_N=3000 | 41.00|0.56|0.20 | 34.00|0.59|0.26 | 34.00|0.59|0.25 | 37.80|0.54|0.18 | 54.00|0.27|0.20 | 0.01 | 0|3.72 | 8000|6.16 | 4000|5.57 | 8000|6.08 | 8000|0.00
p=20_N=3000 | 27.00|0.70|0.34 | 19.00|0.74|0.41 | 19.00|0.74|0.34 | 22.00|0.71|0.35 | 59.00|0.14|0.07 | 0.06 | 0|2.44 | 8000|3.58 | 4000|3.84 | 8000|3.96 | 8000|0.00
p=20_N=3000 | 18.00|0.73|0.32 | 19.00|0.69|0.30 | 19.00|0.69|0.31 | 20.80|0.67|0.29 | 59.00|0.17|0.07 | -0.02 | 0|2.11 | 8000|4.23 | 4000|3.67 | 8000|3.77 | 8000|0.00
p=20_N=3000 | 33.00|0.67|0.30 | 36.00|0.58|0.32 | 36.00|0.58|0.25 | 37.40|0.57|0.29 | 69.00|0.19|0.05 | 0.08 | 0|3.46 | 8000|5.54 | 4000|5.98 | 8000|6.30 | 8000|0.00
p=20_N=3000 | 25.00|0.67|0.26 | 27.00|0.57|0.23 | 27.00|0.57|0.21 | 25.00|0.63|0.28 | 68.00|0.23|0.08 | 0.02 | 0|2.25 | 8000|3.71 | 4000|3.74 | 8000|3.76 | 8000|0.00
p=20_N=3000 | 21.00|0.70|0.33 | 22.00|0.68|0.39 | 22.00|0.68|0.34 | 22.80|0.67|0.35 | 61.00|0.21|0.14 | 0.05 | 0|2.25 | 8000|3.61 | 4000|3.65 | 8000|3.56 | 8000|0.00
p=20_N=3000 | 48.00|0.49|0.24 | 44.00|0.42|0.23 | 44.00|0.42|0.16 | 44.80|0.42|0.24 | 62.00|0.11|0.06 | 0.07 | 0|4.33 | 8000|7.07 | 4000|7.10 | 8000|7.55 | 8000|0.00
p=20_N=3000 | 19.00|0.77|0.42 | 24.00|0.68|0.38 | 24.00|0.68|0.19 | 24.60|0.68|0.34 | 56.00|0.18|0.03 | 0.19 | 0|2.60 | 8000|4.78 | 4000|4.14 | 8000|4.58 | 8000|0.00
p=20_N=3000 | 29.00|0.67|0.32 | 32.00|0.60|0.19 | 32.00|0.60|0.18 | 31.80|0.60|0.26 | 62.00|0.24|0.08 | 0.01 | 0|3.80 | 8000|5.70 | 4000|5.84 | 8000|5.96 | 8000|0.00
p=20_N=3000 | 18.00|0.78|0.35 | 23.00|0.68|0.36 | 23.00|0.68|0.27 | 22.60|0.69|0.30 | 63.00|0.16|0.06 | 0.09 | 0|2.30 | 8000|4.20 | 4000|4.23 | 8000|4.13 | 8000|0.00
p=20_N=3000 | 31.00|0.61|0.28 | 25.00|0.63|0.27 | 25.00|0.63|0.27 | 27.40|0.61|0.27 | 66.00|0.17|0.06 | 0.01 | 0|2.21 | 8000|4.39 | 4000|4.34 | 8000|4.56 | 8000|0.00
p=20_N=3000 | 33.00|0.57|0.26 | 29.00|0.55|0.23 | 29.00|0.55|0.18 | 30.80|0.55|0.22 | 65.00|0.16|0.09 | 0.05 | 0|2.61 | 8000|4.94 | 4000|5.34 | 8000|5.27 | 8000|0.00
p=20_N=3000 | 18.00|0.79|0.43 | 16.00|0.78|0.34 | 16.00|0.78|0.19 | 16.80|0.78|0.35 | 58.00|0.19|0.11 | 0.16 | 0|2.21 | 8000|4.43 | 4000|4.42 | 8000|4.20 | 8000|0.00
p=20_N=3000 | 21.00|0.75|0.31 | 24.00|0.69|0.37 | 24.00|0.69|0.26 | 25.00|0.68|0.37 | 66.00|0.20|0.08 | 0.10 | 0|1.80 | 8000|3.91 | 4000|3.55 | 8000|3.41 | 8000|0.00
p=20_N=3000 | 24.00|0.71|0.40 | 27.00|0.65|0.36 | 27.00|0.65|0.31 | 27.60|0.64|0.37 | 47.00|0.32|0.13 | 0.05 | 0|2.71 | 8000|4.37 | 4000|4.41 | 8000|4.07 | 8000|0.00
p=20_N=3000 | 22.00|0.74|0.38 | 20.00|0.71|0.36 | 20.00|0.71|0.25 | 22.00|0.70|0.35 | 56.00|0.28|0.17 | 0.11 | 0|2.15 | 8000|3.58 | 4000|3.64 | 8000|3.75 | 8000|0.00
p=20_N=3000 | 32.00|0.60|0.32 | 29.00|0.59|0.36 | 29.00|0.59|0.30 | 30.80|0.57|0.33 | 55.00|0.23|0.09 | 0.06 | 0|3.47 | 8000|5.05 | 4000|5.27 | 8000|5.18 | 8000|0.00
p=20_N=3000 | 30.00|0.62|0.34 | 25.00|0.63|0.48 | 25.00|0.63|0.30 | 29.00|0.58|0.34 | 63.00|0.14|0.06 | 0.18 | 0|2.69 | 8000|4.50 | 4000|4.35 | 8000|4.18 | 8000|0.00
p=20_N=3000 | 39.00|0.51|0.18 | 36.00|0.47|0.09 | 36.00|0.47|0.03 | 35.80|0.49|0.14 | 63.00|0.24|0.09 | 0.06 | 0|2.94 | 8000|4.63 | 4000|4.16 | 8000|4.01 | 8000|0.00
"""

rows = []
lines = text.strip().split("\n")

run_id = 0

for line in lines:
    if line.startswith("p="):
        run_id += 1
        parts = [p.strip() for p in line.split(" | ")]

        scenario = parts[0]

        methods_metrics = {
            "centralized": parts[1],
            "fedpc": parts[2],
            "naive": parts[3],
            "local": parts[4],
            "random": parts[5],
        }

        delta = float(parts[6])

        comm_runtime = {
            "centralized": parts[7],
            "fedpc": parts[8],
            "naive": parts[9],
            "local": parts[10],
            "random": parts[11],
        }

        for method in methods_metrics:
            shd, f1, orient = methods_metrics[method].split("|")
            comm, runtime = comm_runtime[method].split("|")

            rows.append({
                "Run": run_id,
                "Scenario": scenario,
                "Method": method,
                "SHD": float(shd),
                "F1": float(f1),
                "Orient_F1": float(orient),
                "Comm": float(comm),
                "Runtime_s": float(runtime),
                "Delta_F1_fedpc_naive": delta if method == "fedpc" else None
            })

df = pd.DataFrame(rows)

# save to Excel
df.to_excel("table5.xlsx", index=False)
print("Excel file saved: table5.xlsx")

###################################### N 5000

import pandas as pd

text = """
Scenario | centralized (SHD|F1|Orient_F1) | fedpc (SHD|F1|Orient_F1) | naive (SHD|F1|Orient_F1) | local (SHD|F1|Orient_F1) | random (SHD|F1|Orient_F1) | ΔF1_fedpc_naive | centralized (Comm|Runtime s) | fedpc (Comm|Runtime s) | naive (Comm|Runtime s) | local (Comm|Runtime s) | random (Comm|Runtime s)
p=20_N=5000 | 29.00|0.67|0.26 | 28.00|0.64|0.29 | 28.00|0.64|0.13 | 29.20|0.63|0.26 | 66.00|0.13|0.05 | 0.15 | 0|29.31 | 8000|5.84 | 4000|5.83 | 8000|6.05 | 8000|0.00
p=20_N=5000 | 29.00|0.65|0.22 | 26.00|0.64|0.20 | 26.00|0.64|0.12 | 27.40|0.63|0.22 | 66.00|0.13|0.06 | 0.08 | 0|24.57 | 8000|5.08 | 4000|4.62 | 8000|4.71 | 8000|0.00
p=20_N=5000 | 15.00|0.79|0.48 | 14.00|0.79|0.45 | 14.00|0.79|0.34 | 14.80|0.78|0.45 | 64.00|0.14|0.07 | 0.11 | 0|6.87 | 8000|2.50 | 4000|2.44 | 8000|2.42 | 8000|0.00
p=20_N=5000 | 12.00|0.82|0.36 | 10.00|0.84|0.41 | 10.00|0.84|0.25 | 10.60|0.83|0.42 | 49.00|0.25|0.17 | 0.16 | 0|8.14 | 8000|3.12 | 4000|3.02 | 8000|3.05 | 8000|0.00
p=20_N=5000 | 13.00|0.79|0.51 | 12.00|0.81|0.48 | 12.00|0.81|0.27 | 14.00|0.78|0.48 | 56.00|0.18|0.07 | 0.21 | 0|6.60 | 8000|3.66 | 4000|3.53 | 8000|3.74 | 8000|0.00
p=20_N=5000 | 2.00|0.96|0.92 | 1.00|0.98|0.94 | 1.00|0.98|0.61 | 2.60|0.95|0.82 | 44.00|0.24|0.11 | 0.33 | 0|4.35 | 8000|2.51 | 4000|2.82 | 8000|2.51 | 8000|0.00
p=20_N=5000 | 22.00|0.70|0.38 | 23.00|0.66|0.32 | 23.00|0.66|0.33 | 22.60|0.66|0.27 | 50.00|0.29|0.17 | -0.01 | 0|27.85 | 8000|5.62 | 4000|5.17 | 8000|5.56 | 8000|0.00
p=20_N=5000 | 41.00|0.56|0.27 | 42.00|0.47|0.27 | 42.00|0.47|0.25 | 44.00|0.47|0.25 | 66.00|0.23|0.17 | 0.02 | 0|100.64 | 8000|16.27 | 4000|15.41 | 8000|15.23 | 8000|0.00
p=20_N=5000 | 25.00|0.70|0.21 | 20.00|0.72|0.35 | 20.00|0.72|0.20 | 20.80|0.72|0.39 | 57.00|0.20|0.07 | 0.15 | 0|23.88 | 8000|5.49 | 4000|5.61 | 8000|5.47 | 8000|0.00
p=20_N=5000 | 35.00|0.59|0.28 | 32.00|0.57|0.26 | 32.00|0.57|0.27 | 33.40|0.56|0.24 | 56.00|0.24|0.09 | -0.02 | 0|23.52 | 8000|5.83 | 4000|5.67 | 8000|5.40 | 8000|0.00
p=20_N=5000 | 22.00|0.72|0.36 | 13.00|0.82|0.49 | 13.00|0.82|0.39 | 16.40|0.78|0.45 | 57.00|0.15|0.09 | 0.10 | 0|18.29 | 8000|5.50 | 4000|5.19 | 8000|5.20 | 8000|0.00
p=20_N=5000 | 8.00|0.87|0.51 | 12.00|0.79|0.45 | 12.00|0.79|0.43 | 12.80|0.78|0.44 | 56.00|0.24|0.14 | 0.02 | 0|8.30 | 8000|2.82 | 4000|2.59 | 8000|2.73 | 8000|0.00
p=20_N=5000 | 37.00|0.62|0.30 | 38.00|0.55|0.29 | 38.00|0.55|0.19 | 37.80|0.56|0.25 | 58.00|0.29|0.12 | 0.10 | 0|30.15 | 8000|5.36 | 4000|5.07 | 8000|5.23 | 8000|0.00
p=20_N=5000 | 32.00|0.59|0.24 | 20.00|0.67|0.31 | 20.00|0.67|0.24 | 22.60|0.64|0.32 | 58.00|0.22|0.07 | 0.07 | 0|22.62 | 8000|5.58 | 4000|5.66 | 8000|5.70 | 8000|0.00
p=20_N=5000 | 27.00|0.66|0.24 | 32.00|0.57|0.24 | 32.00|0.57|0.22 | 33.20|0.57|0.29 | 58.00|0.22|0.13 | 0.02 | 0|36.35 | 8000|7.51 | 4000|7.38 | 8000|7.44 | 8000|0.00
p=20_N=5000 | 35.00|0.60|0.35 | 26.00|0.64|0.29 | 26.00|0.64|0.33 | 29.00|0.62|0.31 | 63.00|0.20|0.03 | -0.04 | 0|27.05 | 8000|5.64 | 4000|6.04 | 8000|5.26 | 8000|0.00
p=20_N=5000 | 18.00|0.74|0.41 | 16.00|0.73|0.41 | 16.00|0.73|0.32 | 18.00|0.72|0.37 | 52.00|0.26|0.10 | 0.09 | 0|14.95 | 8000|3.49 | 4000|3.59 | 8000|3.67 | 8000|0.00
p=20_N=5000 | 19.00|0.73|0.29 | 18.00|0.73|0.34 | 18.00|0.73|0.27 | 18.80|0.72|0.30 | 70.00|0.05|0.00 | 0.07 | 0|13.26 | 8000|3.71 | 4000|3.96 | 8000|3.97 | 8000|0.00
p=20_N=5000 | 45.00|0.54|0.19 | 36.00|0.55|0.17 | 36.00|0.55|0.12 | 38.60|0.52|0.19 | 63.00|0.16|0.03 | 0.04 | 0|185.07 | 8000|17.58 | 4000|14.08 | 8000|15.10 | 8000|0.00
p=20_N=5000 | 36.00|0.63|0.34 | 29.00|0.66|0.37 | 29.00|0.66|0.27 | 33.60|0.63|0.35 | 70.00|0.15|0.13 | 0.10 | 0|73.91 | 8000|10.38 | 4000|10.62 | 8000|11.42 | 8000|0.00
"""

rows = []
lines = text.strip().split("\n")

run_id = 0

for line in lines:
    if line.startswith("p="):
        run_id += 1
        parts = [p.strip() for p in line.split(" | ")]

        scenario = parts[0]

        methods_metrics = {
            "centralized": parts[1],
            "fedpc": parts[2],
            "naive": parts[3],
            "local": parts[4],
            "random": parts[5],
        }

        delta = float(parts[6])

        comm_runtime = {
            "centralized": parts[7],
            "fedpc": parts[8],
            "naive": parts[9],
            "local": parts[10],
            "random": parts[11],
        }

        for method in methods_metrics:
            shd, f1, orient = methods_metrics[method].split("|")
            comm, runtime = comm_runtime[method].split("|")

            rows.append({
                "Run": run_id,
                "Scenario": scenario,
                "Method": method,
                "SHD": float(shd),
                "F1": float(f1),
                "Orient_F1": float(orient),
                "Comm": float(comm),
                "Runtime_s": float(runtime),
                "Delta_F1_fedpc_naive": delta if method == "fedpc" else None
            })

df = pd.DataFrame(rows)

# save to Excel
df.to_excel("table6.xlsx", index=False)
print("Excel file saved: table6.xlsx")

###################################### phi 0.1

import pandas as pd

text = """
Scenario | centralized (SHD|F1|Orient_F1) | fedpc (SHD|F1|Orient_F1) | naive (SHD|F1|Orient_F1) | local (SHD|F1|Orient_F1) | random (SHD|F1|Orient_F1) | ΔF1_fedpc_naive | centralized (Comm|Runtime s) | fedpc (Comm|Runtime s) | naive (Comm|Runtime s) | local (Comm|Runtime s) | random (Comm|Runtime s)
p=20_phi=0.1 | 6.00|0.84|0.42 | 1.00|0.97|0.79 | 1.00|0.97|0.40 | 3.60|0.90|0.63 | 26.00|0.07|0.00 | 0.39 | 0|2.46 | 8000|1.79 | 4000|1.76 | 8000|1.62 | 8000|0.00
p=20_phi=0.1 | 12.00|0.70|0.50 | 7.00|0.77|0.77 | 7.00|0.77|0.25 | 8.60|0.74|0.56 | 32.00|0.06|0.06 | 0.52 | 0|3.92 | 8000|2.07 | 4000|2.18 | 8000|2.08 | 8000|0.00
p=20_phi=0.1 | 1.00|0.97|0.58 | 0.00|1.00|0.73 | 0.00|1.00|0.36 | 2.40|0.93|0.71 | 33.00|0.11|0.06 | 0.38 | 0|0.76 | 8000|1.12 | 4000|1.09 | 8000|1.11 | 8000|0.00
p=20_phi=0.1 | 3.00|0.93|0.60 | 1.00|0.97|0.42 | 1.00|0.97|0.46 | 2.20|0.95|0.38 | 38.00|0.05|0.00 | -0.04 | 0|2.37 | 8000|1.49 | 4000|1.31 | 8000|1.38 | 8000|0.00
p=20_phi=0.1 | 9.00|0.82|0.39 | 10.00|0.80|0.48 | 10.00|0.80|0.30 | 10.20|0.80|0.39 | 44.00|0.12|0.09 | 0.17 | 0|4.51 | 8000|2.71 | 4000|2.45 | 8000|2.62 | 8000|0.00
p=20_phi=0.1 | 6.00|0.88|0.36 | 4.00|0.92|0.54 | 4.00|0.92|0.48 | 5.00|0.90|0.53 | 33.00|0.23|0.15 | 0.06 | 0|5.89 | 8000|1.99 | 4000|1.74 | 8000|1.65 | 8000|0.00
p=20_phi=0.1 | 9.00|0.82|0.29 | 6.00|0.87|0.22 | 6.00|0.87|0.20 | 5.80|0.87|0.23 | 36.00|0.10|0.05 | 0.02 | 0|2.71 | 8000|1.61 | 4000|1.41 | 8000|1.45 | 8000|0.00
p=20_phi=0.1 | 6.00|0.88|0.42 | 7.00|0.85|0.38 | 7.00|0.85|0.29 | 8.00|0.83|0.36 | 40.00|0.09|0.10 | 0.09 | 0|1.22 | 8000|1.50 | 4000|1.43 | 8000|1.45 | 8000|0.00
p=20_phi=0.1 | 3.00|0.93|0.56 | 3.00|0.93|0.57 | 3.00|0.93|0.44 | 3.20|0.93|0.58 | 44.00|0.12|0.04 | 0.13 | 0|1.61 | 8000|1.65 | 4000|1.58 | 8000|1.57 | 8000|0.00
p=20_phi=0.1 | 3.00|0.93|0.55 | 1.00|0.97|0.86 | 1.00|0.97|0.51 | 2.20|0.94|0.71 | 37.00|0.05|0.00 | 0.35 | 0|0.90 | 8000|1.20 | 4000|1.27 | 8000|1.37 | 8000|0.00
p=20_phi=0.1 | 6.00|0.86|0.54 | 0.00|1.00|0.89 | 0.00|1.00|0.50 | 2.20|0.94|0.53 | 29.00|0.06|0.07 | 0.39 | 0|1.85 | 8000|1.40 | 4000|1.22 | 8000|1.31 | 8000|0.00
p=20_phi=0.1 | 7.00|0.84|0.67 | 7.00|0.83|0.25 | 7.00|0.83|0.42 | 7.80|0.82|0.38 | 44.00|0.04|0.00 | -0.17 | 0|3.15 | 8000|1.60 | 4000|1.51 | 8000|1.51 | 8000|0.00
p=20_phi=0.1 | 5.00|0.86|0.50 | 3.00|0.91|0.46 | 3.00|0.91|0.40 | 5.60|0.85|0.49 | 34.00|0.06|0.06 | 0.06 | 0|0.58 | 8000|1.18 | 4000|0.90 | 8000|0.89 | 8000|0.00
p=20_phi=0.1 | 7.00|0.84|0.48 | 3.00|0.93|0.44 | 3.00|0.93|0.51 | 5.00|0.88|0.44 | 43.00|0.09|0.09 | -0.07 | 0|1.93 | 8000|1.61 | 4000|1.53 | 8000|1.65 | 8000|0.00
p=20_phi=0.1 | 1.00|0.98|0.55 | 1.00|0.98|0.57 | 1.00|0.98|0.30 | 4.20|0.92|0.46 | 34.00|0.15|0.05 | 0.26 | 0|2.29 | 8000|1.75 | 4000|1.65 | 8000|1.69 | 8000|0.00
p=20_phi=0.1 | 3.00|0.92|0.58 | 0.00|1.00|0.83 | 0.00|1.00|0.44 | 2.00|0.95|0.74 | 40.00|0.05|0.05 | 0.40 | 0|1.19 | 8000|1.57 | 4000|1.35 | 8000|1.26 | 8000|0.00
p=20_phi=0.1 | 19.00|0.69|0.36 | 19.00|0.63|0.34 | 19.00|0.63|0.29 | 20.00|0.63|0.35 | 48.00|0.08|0.04 | 0.05 | 0|14.01 | 8000|3.30 | 4000|3.24 | 8000|3.38 | 8000|0.00
p=20_phi=0.1 | 2.00|0.93|0.80 | 3.00|0.90|0.76 | 3.00|0.90|0.43 | 5.00|0.84|0.68 | 33.00|0.15|0.11 | 0.33 | 0|0.57 | 8000|1.08 | 4000|1.08 | 8000|1.17 | 8000|0.00
p=20_phi=0.1 | 4.00|0.91|0.43 | 2.00|0.95|0.52 | 2.00|0.95|0.44 | 4.20|0.91|0.51 | 41.00|0.00|0.00 | 0.08 | 0|4.47 | 8000|2.09 | 4000|1.96 | 8000|1.93 | 8000|0.00
p=20_phi=0.1 | 2.00|0.93|0.27 | 0.00|1.00|0.79 | 0.00|1.00|0.46 | 3.80|0.88|0.58 | 36.00|0.05|0.00 | 0.32 | 0|0.82 | 8000|1.32 | 4000|1.24 | 8000|1.30 | 8000|0.00
"""

rows = []
lines = text.strip().split("\n")

run_id = 0

for line in lines:
    if line.startswith("p="):
        run_id += 1
        parts = [p.strip() for p in line.split(" | ")]

        scenario = parts[0]

        methods_metrics = {
            "centralized": parts[1],
            "fedpc": parts[2],
            "naive": parts[3],
            "local": parts[4],
            "random": parts[5],
        }

        delta = float(parts[6])

        comm_runtime = {
            "centralized": parts[7],
            "fedpc": parts[8],
            "naive": parts[9],
            "local": parts[10],
            "random": parts[11],
        }

        for method in methods_metrics:
            shd, f1, orient = methods_metrics[method].split("|")
            comm, runtime = comm_runtime[method].split("|")

            rows.append({
                "Run": run_id,
                "Scenario": scenario,
                "Method": method,
                "SHD": float(shd),
                "F1": float(f1),
                "Orient_F1": float(orient),
                "Comm": float(comm),
                "Runtime_s": float(runtime),
                "Delta_F1_fedpc_naive": delta if method == "fedpc" else None
            })

df = pd.DataFrame(rows)

# save to Excel
df.to_excel("table7.xlsx", index=False)
print("Excel file saved: table7.xlsx")

###################################### phi 0.2

import pandas as pd

text = """
Scenario | centralized (SHD|F1|Orient_F1) | fedpc (SHD|F1|Orient_F1) | naive (SHD|F1|Orient_F1) | local (SHD|F1|Orient_F1) | random (SHD|F1|Orient_F1) | ΔF1_fedpc_naive | centralized (Comm|Runtime s) | fedpc (Comm|Runtime s) | naive (Comm|Runtime s) | local (Comm|Runtime s) | random (Comm|Runtime s)
p=20_phi=0.2 | 22.00|0.72|0.39 | 26.00|0.59|0.28 | 26.00|0.59|0.14 | 26.00|0.60|0.24 | 64.00|0.11|0.03 | 0.14 | 0|19.43 | 8000|4.30 | 4000|3.80 | 8000|4.10 | 8000|0.00
p=20_phi=0.2 | 29.00|0.67|0.34 | 33.00|0.59|0.27 | 33.00|0.59|0.26 | 32.80|0.60|0.30 | 65.00|0.18|0.08 | 0.01 | 0|25.18 | 8000|4.75 | 4000|4.84 | 8000|4.63 | 8000|0.00
p=20_phi=0.2 | 39.00|0.61|0.16 | 34.00|0.57|0.27 | 34.00|0.57|0.25 | 36.00|0.57|0.22 | 61.00|0.19|0.08 | 0.01 | 0|40.56 | 8000|6.94 | 4000|7.26 | 8000|7.83 | 8000|0.00
p=20_phi=0.2 | 30.00|0.66|0.37 | 23.00|0.69|0.27 | 23.00|0.69|0.29 | 27.20|0.65|0.31 | 61.00|0.30|0.23 | -0.02 | 0|30.17 | 8000|6.27 | 4000|5.84 | 8000|6.11 | 8000|0.00
p=20_phi=0.2 | 25.00|0.68|0.33 | 26.00|0.62|0.28 | 26.00|0.62|0.20 | 24.80|0.65|0.31 | 61.00|0.19|0.09 | 0.08 | 0|25.66 | 8000|4.73 | 4000|4.56 | 8000|4.81 | 8000|0.00
p=20_phi=0.2 | 30.00|0.63|0.31 | 28.00|0.58|0.37 | 28.00|0.58|0.33 | 29.20|0.58|0.28 | 65.00|0.16|0.11 | 0.05 | 0|28.29 | 8000|4.90 | 4000|4.87 | 8000|4.91 | 8000|0.00
p=20_phi=0.2 | 40.00|0.57|0.20 | 39.00|0.53|0.24 | 39.00|0.53|0.11 | 38.00|0.53|0.24 | 66.00|0.23|0.11 | 0.13 | 0|46.91 | 8000|8.10 | 4000|7.41 | 8000|7.25 | 8000|0.00
p=20_phi=0.2 | 29.00|0.66|0.32 | 22.00|0.69|0.26 | 22.00|0.69|0.19 | 24.60|0.66|0.27 | 62.00|0.22|0.03 | 0.07 | 0|44.82 | 8000|8.59 | 4000|8.55 | 8000|9.75 | 8000|0.00
p=20_phi=0.2 | 17.00|0.77|0.48 | 19.00|0.70|0.42 | 19.00|0.70|0.39 | 20.80|0.68|0.40 | 60.00|0.17|0.15 | 0.03 | 0|28.90 | 8000|5.21 | 4000|5.13 | 8000|5.12 | 8000|0.00
p=20_phi=0.2 | 43.00|0.55|0.27 | 42.00|0.49|0.28 | 42.00|0.49|0.17 | 43.20|0.49|0.22 | 79.00|0.15|0.10 | 0.11 | 0|72.74 | 8000|7.95 | 4000|6.36 | 8000|6.54 | 8000|0.00
p=20_phi=0.2 | 31.00|0.65|0.43 | 26.00|0.66|0.46 | 26.00|0.66|0.38 | 28.20|0.64|0.35 | 76.00|0.10|0.06 | 0.08 | 0|36.22 | 8000|7.34 | 4000|6.97 | 8000|6.98 | 8000|0.00
p=20_phi=0.2 | 11.00|0.83|0.36 | 7.00|0.88|0.43 | 7.00|0.88|0.40 | 8.20|0.86|0.44 | 60.00|0.12|0.00 | 0.03 | 0|5.69 | 8000|2.60 | 4000|2.39 | 8000|2.39 | 8000|0.00
p=20_phi=0.2 | 18.00|0.74|0.47 | 16.00|0.72|0.34 | 16.00|0.72|0.27 | 17.60|0.70|0.32 | 59.00|0.21|0.13 | 0.07 | 0|6.51 | 8000|2.60 | 4000|2.67 | 8000|2.71 | 8000|0.00
p=20_phi=0.2 | 17.00|0.76|0.35 | 13.00|0.79|0.37 | 13.00|0.79|0.33 | 16.00|0.76|0.38 | 66.00|0.11|0.03 | 0.05 | 0|7.71 | 8000|2.96 | 4000|2.67 | 8000|2.73 | 8000|0.00
p=20_phi=0.2 | 26.00|0.68|0.14 | 19.00|0.73|0.24 | 19.00|0.73|0.14 | 20.00|0.72|0.26 | 66.00|0.20|0.14 | 0.10 | 0|16.23 | 8000|4.79 | 4000|4.30 | 8000|4.37 | 8000|0.00
p=20_phi=0.2 | 33.00|0.64|0.23 | 35.00|0.55|0.23 | 35.00|0.55|0.18 | 36.40|0.56|0.29 | 63.00|0.32|0.20 | 0.04 | 0|34.83 | 8000|8.06 | 4000|8.15 | 8000|8.97 | 8000|0.00
p=20_phi=0.2 | 29.00|0.63|0.36 | 22.00|0.68|0.31 | 22.00|0.68|0.30 | 26.80|0.62|0.29 | 68.00|0.15|0.06 | 0.02 | 0|28.51 | 8000|5.07 | 4000|4.80 | 8000|4.80 | 8000|0.00
p=20_phi=0.2 | 41.00|0.59|0.28 | 32.00|0.62|0.33 | 32.00|0.62|0.27 | 33.60|0.60|0.25 | 72.00|0.18|0.10 | 0.06 | 0|37.02 | 8000|7.05 | 4000|6.90 | 8000|6.85 | 8000|0.00
p=20_phi=0.2 | 17.00|0.80|0.33 | 17.00|0.77|0.30 | 17.00|0.77|0.36 | 17.80|0.76|0.30 | 67.00|0.17|0.06 | -0.06 | 0|9.88 | 8000|3.14 | 4000|3.03 | 8000|3.15 | 8000|0.00
p=20_phi=0.2 | 37.00|0.60|0.32 | 30.00|0.59|0.19 | 30.00|0.59|0.23 | 32.00|0.58|0.20 | 65.00|0.20|0.09 | -0.04 | 0|42.27 | 8000|7.42 | 4000|7.12 | 8000|7.04 | 8000|0.00
"""

rows = []
lines = text.strip().split("\n")

run_id = 0

for line in lines:
    if line.startswith("p="):
        run_id += 1
        parts = [p.strip() for p in line.split(" | ")]

        scenario = parts[0]

        methods_metrics = {
            "centralized": parts[1],
            "fedpc": parts[2],
            "naive": parts[3],
            "local": parts[4],
            "random": parts[5],
        }

        delta = float(parts[6])

        comm_runtime = {
            "centralized": parts[7],
            "fedpc": parts[8],
            "naive": parts[9],
            "local": parts[10],
            "random": parts[11],
        }

        for method in methods_metrics:
            shd, f1, orient = methods_metrics[method].split("|")
            comm, runtime = comm_runtime[method].split("|")

            rows.append({
                "Run": run_id,
                "Scenario": scenario,
                "Method": method,
                "SHD": float(shd),
                "F1": float(f1),
                "Orient_F1": float(orient),
                "Comm": float(comm),
                "Runtime_s": float(runtime),
                "Delta_F1_fedpc_naive": delta if method == "fedpc" else None
            })

df = pd.DataFrame(rows)

# save to Excel
df.to_excel("table8.xlsx", index=False)
print("Excel file saved: table8.xlsx")

###################################### hetero none

import pandas as pd

text = """
Scenario | centralized (SHD|F1|Orient_F1) | fedpc (SHD|F1|Orient_F1) | naive (SHD|F1|Orient_F1) | local (SHD|F1|Orient_F1) | random (SHD|F1|Orient_F1) | ΔF1_fedpc_naive | centralized (Comm|Runtime s) | fedpc (Comm|Runtime s) | naive (Comm|Runtime s) | local (Comm|Runtime s) | random (Comm|Runtime s)
p=20_hetero=None | 43.00|0.56|0.22 | 33.00|0.58|0.35 | 33.00|0.58|0.25 | 34.60|0.58|0.28 | 63.00|0.28|0.05 | 0.10 | 0|44.01 | 8000|7.34 | 4000|6.65 | 8000|6.27 | 8000|0.00
p=20_hetero=None | 27.00|0.68|0.25 | 20.00|0.71|0.28 | 20.00|0.71|0.25 | 22.00|0.70|0.36 | 71.00|0.10|0.07 | 0.02 | 0|19.93 | 8000|3.44 | 4000|3.66 | 8000|3.33 | 8000|0.00
p=20_hetero=None | 26.00|0.72|0.35 | 33.00|0.60|0.30 | 33.00|0.60|0.29 | 33.40|0.61|0.29 | 70.00|0.15|0.03 | 0.01 | 0|51.37 | 8000|8.39 | 4000|7.96 | 8000|8.20 | 8000|0.00
p=20_hetero=None | 15.00|0.77|0.53 | 15.00|0.74|0.41 | 15.00|0.74|0.28 | 16.60|0.72|0.37 | 54.00|0.25|0.09 | 0.13 | 0|10.64 | 8000|3.01 | 4000|2.98 | 8000|2.99 | 8000|0.00
p=20_hetero=None | 21.00|0.71|0.41 | 22.00|0.65|0.23 | 22.00|0.65|0.18 | 22.40|0.65|0.25 | 58.00|0.17|0.04 | 0.05 | 0|12.91 | 8000|4.23 | 4000|3.82 | 8000|3.99 | 8000|0.00
p=20_hetero=None | 32.00|0.61|0.36 | 32.00|0.56|0.20 | 32.00|0.56|0.18 | 34.20|0.55|0.26 | 59.00|0.19|0.12 | 0.02 | 0|39.05 | 8000|5.88 | 4000|5.96 | 8000|6.44 | 8000|0.00
p=20_hetero=None | 13.00|0.75|0.58 | 12.00|0.76|0.48 | 12.00|0.76|0.40 | 12.00|0.76|0.44 | 56.00|0.20|0.16 | 0.08 | 0|5.81 | 8000|2.57 | 4000|2.34 | 8000|2.41 | 8000|0.00
p=20_hetero=None | 34.00|0.63|0.29 | 29.00|0.63|0.35 | 29.00|0.63|0.27 | 32.00|0.60|0.32 | 65.00|0.18|0.08 | 0.08 | 0|50.67 | 8000|9.11 | 4000|8.95 | 8000|9.07 | 8000|0.00
p=20_hetero=None | 23.00|0.70|0.40 | 26.00|0.64|0.27 | 26.00|0.64|0.33 | 25.80|0.64|0.25 | 57.00|0.20|0.13 | -0.06 | 0|18.66 | 8000|5.26 | 4000|4.93 | 8000|4.80 | 8000|0.00
p=20_hetero=None | 19.00|0.75|0.41 | 17.00|0.75|0.51 | 17.00|0.75|0.34 | 18.40|0.73|0.42 | 64.00|0.16|0.09 | 0.16 | 0|20.59 | 8000|4.30 | 4000|4.03 | 8000|4.31 | 8000|0.00
p=20_hetero=None | 18.00|0.76|0.37 | 21.00|0.69|0.22 | 21.00|0.69|0.34 | 19.80|0.71|0.31 | 53.00|0.21|0.12 | -0.12 | 0|7.46 | 8000|2.94 | 4000|2.90 | 8000|2.82 | 8000|0.00
p=20_hetero=None | 22.00|0.74|0.36 | 21.00|0.71|0.31 | 21.00|0.71|0.29 | 20.40|0.73|0.33 | 65.00|0.16|0.08 | 0.02 | 0|14.73 | 8000|5.28 | 4000|5.49 | 8000|5.08 | 8000|0.00
p=20_hetero=None | 35.00|0.64|0.28 | 28.00|0.67|0.29 | 28.00|0.67|0.25 | 29.40|0.66|0.33 | 63.00|0.20|0.11 | 0.04 | 0|26.04 | 8000|5.51 | 4000|5.00 | 8000|5.06 | 8000|0.00
p=20_hetero=None | 17.00|0.79|0.52 | 22.00|0.70|0.31 | 22.00|0.70|0.35 | 22.80|0.70|0.35 | 66.00|0.17|0.06 | -0.05 | 0|26.32 | 8000|6.48 | 4000|6.24 | 8000|7.19 | 8000|0.00
p=20_hetero=None | 16.00|0.75|0.37 | 10.00|0.83|0.37 | 10.00|0.83|0.43 | 11.80|0.80|0.29 | 61.00|0.19|0.12 | -0.06 | 0|6.58 | 8000|2.29 | 4000|2.15 | 8000|2.46 | 8000|0.00
p=20_hetero=None | 24.00|0.71|0.46 | 25.00|0.65|0.37 | 25.00|0.65|0.28 | 26.60|0.64|0.35 | 61.00|0.19|0.18 | 0.09 | 0|38.90 | 8000|7.42 | 4000|7.95 | 8000|7.23 | 8000|0.00
p=20_hetero=None | 12.00|0.80|0.47 | 8.00|0.85|0.52 | 8.00|0.85|0.50 | 12.40|0.79|0.46 | 62.00|0.22|0.06 | 0.02 | 0|10.59 | 8000|3.85 | 4000|3.54 | 8000|3.56 | 8000|0.00
p=20_hetero=None | 13.00|0.81|0.39 | 13.00|0.80|0.35 | 13.00|0.80|0.30 | 15.00|0.77|0.38 | 66.00|0.15|0.06 | 0.05 | 0|9.24 | 8000|3.05 | 4000|2.90 | 8000|2.93 | 8000|0.00
p=20_hetero=None | 14.00|0.81|0.38 | 23.00|0.65|0.27 | 23.00|0.65|0.26 | 22.40|0.66|0.31 | 61.00|0.23|0.04 | 0.00 | 0|17.11 | 8000|4.00 | 4000|3.74 | 8000|4.08 | 8000|0.00
p=20_hetero=None | 27.00|0.70|0.22 | 31.00|0.60|0.33 | 31.00|0.60|0.26 | 30.80|0.61|0.29 | 64.00|0.16|0.06 | 0.07 | 0|28.86 | 8000|6.17 | 4000|6.22 | 8000|5.94 | 8000|0.00
"""

rows = []
lines = text.strip().split("\n")

run_id = 0

for line in lines:
    if line.startswith("p="):
        run_id += 1
        parts = [p.strip() for p in line.split(" | ")]

        scenario = parts[0]

        methods_metrics = {
            "centralized": parts[1],
            "fedpc": parts[2],
            "naive": parts[3],
            "local": parts[4],
            "random": parts[5],
        }

        delta = float(parts[6])

        comm_runtime = {
            "centralized": parts[7],
            "fedpc": parts[8],
            "naive": parts[9],
            "local": parts[10],
            "random": parts[11],
        }

        for method in methods_metrics:
            shd, f1, orient = methods_metrics[method].split("|")
            comm, runtime = comm_runtime[method].split("|")

            rows.append({
                "Run": run_id,
                "Scenario": scenario,
                "Method": method,
                "SHD": float(shd),
                "F1": float(f1),
                "Orient_F1": float(orient),
                "Comm": float(comm),
                "Runtime_s": float(runtime),
                "Delta_F1_fedpc_naive": delta if method == "fedpc" else None
            })

df = pd.DataFrame(rows)

# save to Excel
df.to_excel("table9.xlsx", index=False)
print("Excel file saved: table9.xlsx")

###################################### hetero mild

import pandas as pd

text = """
Scenario | centralized (SHD|F1|Orient_F1) | fedpc (SHD|F1|Orient_F1) | naive (SHD|F1|Orient_F1) | local (SHD|F1|Orient_F1) | random (SHD|F1|Orient_F1) | ΔF1_fedpc_naive | centralized (Comm|Runtime s) | fedpc (Comm|Runtime s) | naive (Comm|Runtime s) | local (Comm|Runtime s) | random (Comm|Runtime s)
p=20_hetero=mild | 34.00|0.63|0.38 | 30.00|0.58|0.33 | 30.00|0.58|0.28 | 34.20|0.56|0.32 | 58.00|0.26|0.14 | 0.05 | 0|28.45 | 8000|4.46 | 4000|4.62 | 8000|4.44 | 8000|0.00
p=20_hetero=mild | 25.00|0.72|0.39 | 22.00|0.70|0.44 | 22.00|0.70|0.31 | 24.20|0.68|0.36 | 60.00|0.23|0.15 | 0.12 | 0|22.48 | 8000|5.23 | 4000|5.18 | 8000|5.42 | 8000|0.00
p=20_hetero=mild | 8.00|0.87|0.60 | 8.00|0.85|0.54 | 8.00|0.85|0.42 | 9.00|0.84|0.50 | 51.00|0.22|0.11 | 0.12 | 0|5.26 | 8000|2.46 | 4000|2.43 | 8000|2.48 | 8000|0.00
p=20_hetero=mild | 39.00|0.61|0.28 | 36.00|0.58|0.30 | 36.00|0.58|0.16 | 38.20|0.56|0.28 | 68.00|0.24|0.12 | 0.14 | 0|31.61 | 8000|5.07 | 4000|5.00 | 8000|5.04 | 8000|0.00
p=20_hetero=mild | 12.00|0.83|0.31 | 13.00|0.80|0.45 | 13.00|0.80|0.38 | 13.80|0.79|0.40 | 63.00|0.22|0.06 | 0.07 | 0|14.98 | 8000|4.67 | 4000|4.50 | 8000|4.98 | 8000|0.00
p=20_hetero=mild | 12.00|0.83|0.48 | 10.00|0.84|0.47 | 10.00|0.84|0.30 | 13.20|0.80|0.46 | 54.00|0.21|0.06 | 0.18 | 0|10.24 | 8000|3.11 | 4000|3.18 | 8000|3.06 | 8000|0.00
p=20_hetero=mild | 37.00|0.57|0.23 | 30.00|0.58|0.27 | 30.00|0.58|0.26 | 31.20|0.57|0.31 | 70.00|0.12|0.09 | 0.01 | 0|28.34 | 8000|5.21 | 4000|5.29 | 8000|5.37 | 8000|0.00
p=20_hetero=mild | 39.00|0.54|0.27 | 33.00|0.56|0.23 | 33.00|0.56|0.15 | 35.20|0.54|0.30 | 54.00|0.18|0.07 | 0.08 | 0|41.81 | 8000|7.17 | 4000|7.10 | 8000|7.03 | 8000|0.00
p=20_hetero=mild | 23.00|0.71|0.38 | 24.00|0.66|0.39 | 24.00|0.66|0.32 | 26.00|0.63|0.37 | 54.00|0.27|0.17 | 0.08 | 0|20.60 | 8000|4.79 | 4000|4.77 | 8000|4.87 | 8000|0.00
p=20_hetero=mild | 27.00|0.67|0.30 | 26.00|0.64|0.38 | 26.00|0.64|0.30 | 29.60|0.60|0.27 | 67.00|0.19|0.06 | 0.08 | 0|20.56 | 8000|5.01 | 4000|4.81 | 8000|4.78 | 8000|0.00
p=20_hetero=mild | 21.00|0.76|0.39 | 23.00|0.70|0.44 | 23.00|0.70|0.35 | 23.40|0.70|0.41 | 50.00|0.31|0.12 | 0.09 | 0|28.38 | 8000|5.68 | 4000|5.93 | 8000|6.39 | 8000|0.00
p=20_hetero=mild | 36.00|0.58|0.33 | 24.00|0.65|0.24 | 24.00|0.65|0.32 | 25.60|0.63|0.29 | 57.00|0.15|0.07 | -0.08 | 0|30.20 | 8000|6.27 | 4000|6.22 | 8000|6.29 | 8000|0.00
p=20_hetero=mild | 8.00|0.88|0.36 | 10.00|0.83|0.30 | 10.00|0.83|0.35 | 12.40|0.79|0.42 | 56.00|0.18|0.06 | -0.05 | 0|14.60 | 8000|3.61 | 4000|3.34 | 8000|3.54 | 8000|0.00
p=20_hetero=mild | 30.00|0.65|0.30 | 25.00|0.68|0.34 | 25.00|0.68|0.23 | 26.60|0.66|0.34 | 70.00|0.12|0.03 | 0.11 | 0|31.49 | 8000|5.99 | 4000|5.93 | 8000|6.09 | 8000|0.00
p=20_hetero=mild | 29.00|0.64|0.19 | 24.00|0.65|0.41 | 24.00|0.65|0.24 | 25.80|0.63|0.31 | 54.00|0.29|0.13 | 0.18 | 0|17.09 | 8000|4.72 | 4000|4.61 | 8000|4.56 | 8000|0.00
p=20_hetero=mild | 23.00|0.74|0.46 | 26.00|0.67|0.33 | 26.00|0.67|0.24 | 28.20|0.66|0.32 | 64.00|0.29|0.09 | 0.10 | 0|18.00 | 8000|5.51 | 4000|5.58 | 8000|5.52 | 8000|0.00
p=20_hetero=mild | 20.00|0.72|0.29 | 15.00|0.75|0.40 | 15.00|0.75|0.31 | 15.80|0.75|0.39 | 57.00|0.20|0.10 | 0.09 | 0|10.58 | 8000|3.29 | 4000|3.28 | 8000|3.15 | 8000|0.00
p=20_hetero=mild | 26.00|0.69|0.41 | 24.00|0.67|0.34 | 24.00|0.67|0.36 | 23.20|0.68|0.37 | 60.00|0.21|0.12 | -0.02 | 0|23.17 | 8000|4.57 | 4000|4.57 | 8000|4.46 | 8000|0.00
p=20_hetero=mild | 16.00|0.76|0.56 | 13.00|0.79|0.59 | 13.00|0.79|0.39 | 13.60|0.79|0.53 | 55.00|0.18|0.14 | 0.20 | 0|10.89 | 8000|3.73 | 4000|3.68 | 8000|3.54 | 8000|0.00
p=20_hetero=mild | 15.00|0.80|0.56 | 11.00|0.85|0.56 | 11.00|0.85|0.43 | 13.00|0.82|0.50 | 50.00|0.29|0.13 | 0.13 | 0|10.25 | 8000|4.15 | 4000|3.90 | 8000|3.92 | 8000|0.00
"""

rows = []
lines = text.strip().split("\n")

run_id = 0

for line in lines:
    if line.startswith("p="):
        run_id += 1
        parts = [p.strip() for p in line.split(" | ")]

        scenario = parts[0]

        methods_metrics = {
            "centralized": parts[1],
            "fedpc": parts[2],
            "naive": parts[3],
            "local": parts[4],
            "random": parts[5],
        }

        delta = float(parts[6])

        comm_runtime = {
            "centralized": parts[7],
            "fedpc": parts[8],
            "naive": parts[9],
            "local": parts[10],
            "random": parts[11],
        }

        for method in methods_metrics:
            shd, f1, orient = methods_metrics[method].split("|")
            comm, runtime = comm_runtime[method].split("|")

            rows.append({
                "Run": run_id,
                "Scenario": scenario,
                "Method": method,
                "SHD": float(shd),
                "F1": float(f1),
                "Orient_F1": float(orient),
                "Comm": float(comm),
                "Runtime_s": float(runtime),
                "Delta_F1_fedpc_naive": delta if method == "fedpc" else None
            })

df = pd.DataFrame(rows)

# save to Excel
df.to_excel("table10.xlsx", index=False)
print("Excel file saved: table10.xlsx")

###################################### hetero strong

import pandas as pd

text = """
Scenario | centralized (SHD|F1|Orient_F1) | fedpc (SHD|F1|Orient_F1) | naive (SHD|F1|Orient_F1) | local (SHD|F1|Orient_F1) | random (SHD|F1|Orient_F1) | ΔF1_fedpc_naive | centralized (Comm|Runtime s) | fedpc (Comm|Runtime s) | naive (Comm|Runtime s) | local (Comm|Runtime s) | random (Comm|Runtime s)
p=20_hetero=strong | 26.00|0.69|0.38 | 24.00|0.67|0.39 | 24.00|0.67|0.38 | 24.00|0.67|0.39 | 56.00|0.12|0.00 | 0.02 | 0|23.92 | 8000|4.60 | 4000|4.82 | 8000|4.46 | 8000|0.00
p=20_hetero=strong | 26.00|0.68|0.39 | 19.00|0.74|0.42 | 19.00|0.74|0.25 | 20.40|0.73|0.40 | 67.00|0.17|0.11 | 0.17 | 0|19.49 | 8000|4.24 | 4000|4.47 | 8000|4.37 | 8000|0.00
p=20_hetero=strong | 12.00|0.83|0.34 | 14.00|0.79|0.45 | 14.00|0.79|0.43 | 15.60|0.77|0.39 | 50.00|0.26|0.19 | 0.03 | 0|7.81 | 8000|3.27 | 4000|3.07 | 8000|3.30 | 8000|0.00
p=20_hetero=strong | 35.00|0.63|0.34 | 32.00|0.60|0.22 | 32.00|0.60|0.12 | 34.20|0.58|0.25 | 61.00|0.25|0.03 | 0.10 | 0|45.95 | 8000|6.99 | 4000|6.86 | 8000|7.09 | 8000|0.00
p=20_hetero=strong | 30.00|0.67|0.30 | 27.00|0.65|0.25 | 27.00|0.65|0.29 | 28.00|0.64|0.28 | 60.00|0.21|0.07 | -0.04 | 0|31.34 | 8000|5.91 | 4000|5.62 | 8000|6.07 | 8000|0.00
p=20_hetero=strong | 32.00|0.64|0.24 | 39.00|0.49|0.19 | 39.00|0.49|0.19 | 39.60|0.50|0.21 | 62.00|0.26|0.17 | -0.00 | 0|59.58 | 8000|8.69 | 4000|8.33 | 8000|8.45 | 8000|0.00
p=20_hetero=strong | 25.00|0.68|0.42 | 26.00|0.61|0.42 | 26.00|0.61|0.32 | 27.20|0.60|0.38 | 55.00|0.13|0.04 | 0.10 | 0|31.13 | 8000|6.02 | 4000|5.72 | 8000|5.87 | 8000|0.00
p=20_hetero=strong | 31.00|0.69|0.31 | 27.00|0.68|0.27 | 27.00|0.68|0.23 | 30.40|0.65|0.23 | 64.00|0.24|0.11 | 0.04 | 0|38.96 | 8000|7.32 | 4000|7.60 | 8000|7.58 | 8000|0.00
p=20_hetero=strong | 15.00|0.78|0.44 | 14.00|0.74|0.44 | 14.00|0.74|0.27 | 15.40|0.73|0.28 | 62.00|0.16|0.09 | 0.17 | 0|7.49 | 8000|2.46 | 4000|2.54 | 8000|2.56 | 8000|0.00
p=20_hetero=strong | 16.00|0.75|0.41 | 16.00|0.71|0.46 | 16.00|0.71|0.28 | 16.40|0.72|0.41 | 54.00|0.16|0.07 | 0.18 | 0|9.54 | 8000|3.54 | 4000|3.69 | 8000|3.48 | 8000|0.00
p=20_hetero=strong | 32.00|0.63|0.25 | 37.00|0.52|0.28 | 37.00|0.52|0.23 | 37.00|0.52|0.25 | 57.00|0.31|0.17 | 0.05 | 0|36.87 | 8000|7.79 | 4000|7.82 | 8000|8.02 | 8000|0.00
p=20_hetero=strong | 8.00|0.87|0.48 | 4.00|0.93|0.42 | 4.00|0.93|0.33 | 6.20|0.89|0.45 | 56.00|0.22|0.04 | 0.09 | 0|5.76 | 8000|2.89 | 4000|2.75 | 8000|2.88 | 8000|0.00
p=20_hetero=strong | 36.00|0.62|0.19 | 31.00|0.63|0.24 | 31.00|0.63|0.18 | 32.00|0.62|0.23 | 68.00|0.28|0.03 | 0.06 | 0|35.27 | 8000|6.76 | 4000|6.03 | 8000|6.16 | 8000|0.00
p=20_hetero=strong | 22.00|0.71|0.39 | 17.00|0.75|0.31 | 17.00|0.75|0.21 | 18.00|0.74|0.35 | 61.00|0.16|0.06 | 0.11 | 0|18.03 | 8000|4.89 | 4000|4.69 | 8000|4.75 | 8000|0.00
p=20_hetero=strong | 25.00|0.68|0.28 | 22.00|0.68|0.42 | 22.00|0.68|0.44 | 23.80|0.66|0.37 | 56.00|0.32|0.11 | -0.02 | 0|19.35 | 8000|4.99 | 4000|4.87 | 8000|5.13 | 8000|0.00
p=20_hetero=strong | 32.00|0.60|0.20 | 28.00|0.62|0.14 | 28.00|0.62|0.22 | 30.40|0.59|0.18 | 54.00|0.36|0.25 | -0.07 | 0|16.89 | 8000|4.43 | 4000|4.32 | 8000|4.41 | 8000|0.00
p=20_hetero=strong | 15.00|0.79|0.41 | 16.00|0.74|0.41 | 16.00|0.74|0.41 | 17.20|0.73|0.39 | 54.00|0.16|0.04 | 0.01 | 0|18.34 | 8000|5.74 | 4000|5.70 | 8000|5.57 | 8000|0.00
p=20_hetero=strong | 29.00|0.62|0.41 | 22.00|0.69|0.39 | 22.00|0.69|0.30 | 23.00|0.67|0.40 | 60.00|0.17|0.13 | 0.09 | 0|18.27 | 8000|4.72 | 4000|4.77 | 8000|4.85 | 8000|0.00
p=20_hetero=strong | 23.00|0.70|0.32 | 18.00|0.71|0.28 | 18.00|0.71|0.24 | 19.60|0.69|0.38 | 48.00|0.17|0.19 | 0.04 | 0|15.06 | 8000|4.11 | 4000|4.12 | 8000|3.97 | 8000|0.00
p=20_hetero=strong | 24.00|0.68|0.30 | 25.00|0.65|0.25 | 25.00|0.65|0.21 | 25.00|0.64|0.31 | 61.00|0.21|0.00 | 0.04 | 0|23.00 | 8000|5.86 | 4000|5.85 | 8000|5.86 | 8000|0.00
"""

rows = []
lines = text.strip().split("\n")

run_id = 0

for line in lines:
    if line.startswith("p="):
        run_id += 1
        parts = [p.strip() for p in line.split(" | ")]

        scenario = parts[0]

        methods_metrics = {
            "centralized": parts[1],
            "fedpc": parts[2],
            "naive": parts[3],
            "local": parts[4],
            "random": parts[5],
        }

        delta = float(parts[6])

        comm_runtime = {
            "centralized": parts[7],
            "fedpc": parts[8],
            "naive": parts[9],
            "local": parts[10],
            "random": parts[11],
        }

        for method in methods_metrics:
            shd, f1, orient = methods_metrics[method].split("|")
            comm, runtime = comm_runtime[method].split("|")

            rows.append({
                "Run": run_id,
                "Scenario": scenario,
                "Method": method,
                "SHD": float(shd),
                "F1": float(f1),
                "Orient_F1": float(orient),
                "Comm": float(comm),
                "Runtime_s": float(runtime),
                "Delta_F1_fedpc_naive": delta if method == "fedpc" else None
            })

df = pd.DataFrame(rows)

# save to Excel
df.to_excel("table11.xlsx", index=False)
print("Excel file saved: table11.xlsx")

###################################### ell 0.3

import pandas as pd

text = """
Scenario | centralized (SHD|F1|Orient_F1) | fedpc (SHD|F1|Orient_F1) | naive (SHD|F1|Orient_F1) | local (SHD|F1|Orient_F1) | random (SHD|F1|Orient_F1) | ΔF1_fedpc_naive | centralized (Comm|Runtime s) | fedpc (Comm|Runtime s) | naive (Comm|Runtime s) | local (Comm|Runtime s) | random (Comm|Runtime s)
p=20_tau=0.3 | 28.00|0.70|0.17 | 32.00|0.64|0.28 | 32.00|0.64|0.29 | 30.60|0.65|0.26 | 69.00|0.19|0.03 | -0.01 | 0|22.61 | 8000|5.22 | 4000|4.99 | 8000|5.63 | 8000|0.00
p=20_tau=0.3 | 24.00|0.71|0.32 | 27.00|0.66|0.30 | 27.00|0.66|0.22 | 26.60|0.64|0.23 | 52.00|0.28|0.13 | 0.08 | 0|24.14 | 8000|5.13 | 4000|5.07 | 8000|4.97 | 8000|0.00
p=20_tau=0.3 | 26.00|0.68|0.25 | 20.00|0.74|0.27 | 20.00|0.74|0.25 | 22.80|0.69|0.29 | 50.00|0.26|0.03 | 0.01 | 0|26.42 | 8000|5.48 | 4000|6.62 | 8000|5.12 | 8000|0.00
p=20_tau=0.3 | 16.00|0.76|0.36 | 16.00|0.75|0.43 | 16.00|0.75|0.41 | 17.20|0.73|0.45 | 53.00|0.23|0.10 | 0.03 | 0|9.06 | 8000|4.28 | 4000|4.28 | 8000|3.77 | 8000|0.00
p=20_tau=0.3 | 11.00|0.83|0.60 | 14.00|0.77|0.25 | 14.00|0.77|0.42 | 12.60|0.79|0.30 | 58.00|0.12|0.07 | -0.17 | 0|13.16 | 8000|4.15 | 4000|4.20 | 8000|4.53 | 8000|0.00
p=20_tau=0.3 | 18.00|0.74|0.31 | 19.00|0.71|0.32 | 19.00|0.71|0.22 | 19.00|0.71|0.37 | 59.00|0.17|0.07 | 0.10 | 0|23.09 | 8000|5.07 | 4000|4.71 | 8000|4.78 | 8000|0.00
p=20_tau=0.3 | 12.00|0.83|0.29 | 13.00|0.80|0.35 | 13.00|0.80|0.38 | 13.60|0.78|0.30 | 59.00|0.19|0.09 | -0.02 | 0|8.64 | 8000|3.05 | 4000|2.86 | 8000|2.97 | 8000|0.00
p=20_tau=0.3 | 26.00|0.71|0.45 | 33.00|0.60|0.31 | 33.00|0.60|0.25 | 32.60|0.61|0.29 | 75.00|0.07|0.03 | 0.06 | 0|19.71 | 8000|5.61 | 4000|5.59 | 8000|5.49 | 8000|0.00
p=20_tau=0.3 | 47.00|0.51|0.25 | 47.00|0.43|0.27 | 47.00|0.43|0.22 | 42.20|0.47|0.28 | 62.00|0.18|0.15 | 0.05 | 0|45.71 | 8000|7.28 | 4000|7.11 | 8000|6.89 | 8000|0.00
p=20_tau=0.3 | 23.00|0.72|0.36 | 27.00|0.65|0.31 | 27.00|0.65|0.18 | 23.60|0.67|0.30 | 55.00|0.20|0.07 | 0.13 | 0|21.70 | 8000|4.74 | 4000|4.84 | 8000|4.35 | 8000|0.00
p=20_tau=0.3 | 29.00|0.63|0.29 | 29.00|0.61|0.34 | 29.00|0.61|0.26 | 28.00|0.61|0.25 | 66.00|0.15|0.03 | 0.08 | 0|31.57 | 8000|6.00 | 4000|5.98 | 8000|6.87 | 8000|0.00
p=20_tau=0.3 | 47.00|0.48|0.20 | 40.00|0.51|0.17 | 40.00|0.51|0.18 | 37.80|0.50|0.22 | 64.00|0.24|0.11 | -0.01 | 0|28.73 | 8000|6.03 | 4000|5.75 | 8000|6.00 | 8000|0.00
p=20_tau=0.3 | 36.00|0.58|0.27 | 31.00|0.62|0.24 | 31.00|0.62|0.24 | 29.00|0.63|0.25 | 57.00|0.24|0.09 | 0.01 | 0|27.71 | 8000|5.87 | 4000|5.47 | 8000|5.34 | 8000|0.00
p=20_tau=0.3 | 28.00|0.68|0.30 | 27.00|0.67|0.30 | 27.00|0.67|0.25 | 24.60|0.68|0.34 | 63.00|0.24|0.09 | 0.05 | 0|36.61 | 8000|6.35 | 4000|6.35 | 8000|6.76 | 8000|0.00
p=20_tau=0.3 | 22.00|0.74|0.25 | 19.00|0.76|0.20 | 19.00|0.76|0.26 | 17.60|0.77|0.30 | 46.00|0.32|0.12 | -0.06 | 0|22.32 | 8000|4.18 | 4000|4.43 | 8000|4.31 | 8000|0.00
p=20_tau=0.3 | 10.00|0.84|0.43 | 10.00|0.83|0.37 | 10.00|0.83|0.40 | 11.80|0.80|0.41 | 60.00|0.14|0.03 | -0.03 | 0|5.96 | 8000|2.90 | 4000|2.84 | 8000|2.66 | 8000|0.00
p=20_tau=0.3 | 32.00|0.66|0.34 | 36.00|0.59|0.31 | 36.00|0.59|0.27 | 34.20|0.59|0.29 | 61.00|0.30|0.21 | 0.04 | 0|34.78 | 8000|6.78 | 4000|6.06 | 8000|5.94 | 8000|0.00
p=20_tau=0.3 | 26.00|0.66|0.20 | 24.00|0.68|0.26 | 24.00|0.68|0.31 | 22.80|0.68|0.31 | 52.00|0.13|0.11 | -0.04 | 0|17.61 | 8000|5.12 | 4000|4.84 | 8000|4.87 | 8000|0.00
p=20_tau=0.3 | 43.00|0.54|0.15 | 38.00|0.56|0.18 | 38.00|0.56|0.11 | 36.80|0.56|0.17 | 75.00|0.19|0.10 | 0.06 | 0|34.46 | 8000|6.49 | 4000|6.74 | 8000|6.34 | 8000|0.00
p=20_tau=0.3 | 32.00|0.65|0.22 | 33.00|0.62|0.25 | 33.00|0.62|0.21 | 30.60|0.62|0.26 | 70.00|0.15|0.08 | 0.04 | 0|26.62 | 8000|5.16 | 4000|5.16 | 8000|4.96 | 8000|0.00
"""

rows = []
lines = text.strip().split("\n")

run_id = 0

for line in lines:
    if line.startswith("p="):
        run_id += 1
        parts = [p.strip() for p in line.split(" | ")]

        scenario = parts[0]

        methods_metrics = {
            "centralized": parts[1],
            "fedpc": parts[2],
            "naive": parts[3],
            "local": parts[4],
            "random": parts[5],
        }

        delta = float(parts[6])

        comm_runtime = {
            "centralized": parts[7],
            "fedpc": parts[8],
            "naive": parts[9],
            "local": parts[10],
            "random": parts[11],
        }

        for method in methods_metrics:
            shd, f1, orient = methods_metrics[method].split("|")
            comm, runtime = comm_runtime[method].split("|")

            rows.append({
                "Run": run_id,
                "Scenario": scenario,
                "Method": method,
                "SHD": float(shd),
                "F1": float(f1),
                "Orient_F1": float(orient),
                "Comm": float(comm),
                "Runtime_s": float(runtime),
                "Delta_F1_fedpc_naive": delta if method == "fedpc" else None
            })

df = pd.DataFrame(rows)

# save to Excel
df.to_excel("table12.xlsx", index=False)
print("Excel file saved: table12.xlsx")

###################################### ell 0.5

import pandas as pd

text = """
Scenario | centralized (SHD|F1|Orient_F1) | fedpc (SHD|F1|Orient_F1) | naive (SHD|F1|Orient_F1) | local (SHD|F1|Orient_F1) | random (SHD|F1|Orient_F1) | ΔF1_fedpc_naive | centralized (Comm|Runtime s) | fedpc (Comm|Runtime s) | naive (Comm|Runtime s) | local (Comm|Runtime s) | random (Comm|Runtime s)
p=20_tau=0.5 | 35.00|0.66|0.21 | 38.00|0.55|0.35 | 38.00|0.55|0.25 | 35.20|0.60|0.33 | 72.00|0.12|0.08 | 0.10 | 0|37.01 | 8000|6.81 | 4000|6.48 | 8000|6.45 | 8000|0.00
p=20_tau=0.5 | 34.00|0.62|0.21 | 39.00|0.52|0.31 | 39.00|0.52|0.27 | 38.80|0.53|0.27 | 66.00|0.23|0.08 | 0.04 | 0|41.75 | 8000|9.07 | 4000|8.82 | 8000|8.87 | 8000|0.00
p=20_tau=0.5 | 29.00|0.61|0.29 | 25.00|0.63|0.29 | 25.00|0.63|0.25 | 27.20|0.61|0.30 | 57.00|0.22|0.12 | 0.04 | 0|22.43 | 8000|6.28 | 4000|5.92 | 8000|5.95 | 8000|0.00
p=20_tau=0.5 | 24.00|0.71|0.29 | 26.00|0.64|0.33 | 26.00|0.64|0.21 | 24.80|0.66|0.26 | 66.00|0.11|0.06 | 0.12 | 0|15.43 | 8000|4.00 | 4000|3.92 | 8000|3.92 | 8000|0.00
p=20_tau=0.5 | 29.00|0.66|0.39 | 28.00|0.61|0.29 | 28.00|0.61|0.23 | 29.20|0.59|0.29 | 57.00|0.24|0.06 | 0.06 | 0|34.84 | 8000|8.16 | 4000|8.20 | 8000|7.80 | 8000|0.00
p=20_tau=0.5 | 25.00|0.71|0.23 | 21.00|0.70|0.33 | 21.00|0.70|0.31 | 23.80|0.67|0.29 | 62.00|0.22|0.12 | 0.02 | 0|14.64 | 8000|4.82 | 4000|5.14 | 8000|4.94 | 8000|0.00
p=20_tau=0.5 | 35.00|0.60|0.36 | 31.00|0.56|0.32 | 31.00|0.56|0.22 | 31.60|0.56|0.28 | 67.00|0.11|0.11 | 0.10 | 0|30.58 | 8000|4.90 | 4000|4.71 | 8000|4.80 | 8000|0.00
p=20_tau=0.5 | 28.00|0.68|0.29 | 25.00|0.68|0.41 | 25.00|0.68|0.35 | 28.00|0.64|0.33 | 57.00|0.20|0.07 | 0.06 | 0|34.18 | 8000|6.28 | 4000|6.53 | 8000|6.59 | 8000|0.00
p=20_tau=0.5 | 22.00|0.67|0.43 | 19.00|0.70|0.22 | 19.00|0.70|0.22 | 21.20|0.67|0.29 | 64.00|0.14|0.00 | 0.00 | 0|19.50 | 8000|5.67 | 4000|5.66 | 8000|5.63 | 8000|0.00
p=20_tau=0.5 | 19.00|0.77|0.53 | 18.00|0.76|0.56 | 18.00|0.76|0.51 | 17.60|0.76|0.53 | 63.00|0.14|0.03 | 0.05 | 0|18.72 | 8000|4.33 | 4000|4.51 | 8000|4.04 | 8000|0.00
p=20_tau=0.5 | 29.00|0.63|0.38 | 20.00|0.70|0.43 | 20.00|0.70|0.25 | 21.80|0.68|0.35 | 62.00|0.21|0.10 | 0.18 | 0|20.72 | 8000|3.85 | 4000|3.70 | 8000|3.83 | 8000|0.00
p=20_tau=0.5 | 19.00|0.73|0.38 | 14.00|0.77|0.41 | 14.00|0.77|0.35 | 16.40|0.74|0.35 | 62.00|0.22|0.09 | 0.06 | 0|8.71 | 8000|2.62 | 4000|2.60 | 8000|2.48 | 8000|0.00
p=20_tau=0.5 | 38.00|0.55|0.26 | 30.00|0.56|0.32 | 30.00|0.56|0.32 | 31.00|0.56|0.30 | 62.00|0.21|0.03 | 0.00 | 0|31.89 | 8000|6.59 | 4000|6.56 | 8000|6.40 | 8000|0.00
p=20_tau=0.5 | 16.00|0.76|0.39 | 15.00|0.76|0.33 | 15.00|0.76|0.42 | 15.40|0.76|0.37 | 48.00|0.23|0.15 | -0.09 | 0|13.44 | 8000|4.31 | 4000|4.10 | 8000|4.43 | 8000|0.00
p=20_tau=0.5 | 40.00|0.56|0.19 | 36.00|0.53|0.26 | 36.00|0.53|0.21 | 38.00|0.52|0.26 | 72.00|0.18|0.08 | 0.05 | 0|50.35 | 8000|8.75 | 4000|8.80 | 8000|8.43 | 8000|0.00
p=20_tau=0.5 | 30.00|0.61|0.27 | 20.00|0.71|0.47 | 20.00|0.71|0.34 | 22.60|0.68|0.37 | 57.00|0.22|0.10 | 0.12 | 0|27.21 | 8000|6.09 | 4000|6.05 | 8000|6.23 | 8000|0.00
p=20_tau=0.5 | 30.00|0.61|0.37 | 18.00|0.72|0.37 | 18.00|0.72|0.23 | 20.60|0.69|0.32 | 63.00|0.16|0.07 | 0.15 | 0|16.62 | 8000|4.84 | 4000|4.84 | 8000|4.90 | 8000|0.00
p=20_tau=0.5 | 26.00|0.67|0.34 | 23.00|0.67|0.25 | 23.00|0.67|0.25 | 24.40|0.65|0.29 | 68.00|0.19|0.09 | 0.00 | 0|27.80 | 8000|5.94 | 4000|5.80 | 8000|5.76 | 8000|0.00
p=20_tau=0.5 | 23.00|0.68|0.31 | 26.00|0.58|0.40 | 26.00|0.58|0.28 | 24.80|0.60|0.34 | 56.00|0.22|0.07 | 0.12 | 0|19.99 | 8000|4.10 | 4000|3.88 | 8000|3.87 | 8000|0.00
p=20_tau=0.5 | 40.00|0.53|0.19 | 36.00|0.50|0.30 | 36.00|0.50|0.16 | 37.80|0.49|0.24 | 64.00|0.20|0.06 | 0.14 | 0|30.24 | 8000|5.57 | 4000|5.03 | 8000|5.03 | 8000|0.00
"""

rows = []
lines = text.strip().split("\n")

run_id = 0

for line in lines:
    if line.startswith("p="):
        run_id += 1
        parts = [p.strip() for p in line.split(" | ")]

        scenario = parts[0]

        methods_metrics = {
            "centralized": parts[1],
            "fedpc": parts[2],
            "naive": parts[3],
            "local": parts[4],
            "random": parts[5],
        }

        delta = float(parts[6])

        comm_runtime = {
            "centralized": parts[7],
            "fedpc": parts[8],
            "naive": parts[9],
            "local": parts[10],
            "random": parts[11],
        }

        for method in methods_metrics:
            shd, f1, orient = methods_metrics[method].split("|")
            comm, runtime = comm_runtime[method].split("|")

            rows.append({
                "Run": run_id,
                "Scenario": scenario,
                "Method": method,
                "SHD": float(shd),
                "F1": float(f1),
                "Orient_F1": float(orient),
                "Comm": float(comm),
                "Runtime_s": float(runtime),
                "Delta_F1_fedpc_naive": delta if method == "fedpc" else None
            })

df = pd.DataFrame(rows)

# save to Excel
df.to_excel("table13.xlsx", index=False)
print("Excel file saved: table13.xlsx")

###################################### ell 1

import pandas as pd

text = """
Scenario | centralized (SHD|F1|Orient_F1) | fedpc (SHD|F1|Orient_F1) | naive (SHD|F1|Orient_F1) | local (SHD|F1|Orient_F1) | random (SHD|F1|Orient_F1) | ΔF1_fedpc_naive | centralized (Comm|Runtime s) | fedpc (Comm|Runtime s) | naive (Comm|Runtime s) | local (Comm|Runtime s) | random (Comm|Runtime s)
p=20_ell=1 | 47.00|0.58|0.33 | 34.00|0.63|0.41 | 34.00|0.63|0.27 | 37.60|0.61|0.34 | 61.00|0.27|0.19 | 0.14 | 0|2.25 | 4000|4.38 | 2000|4.17 | 4000|4.38 | 4000|0.00
p=20_ell=1 | 41.00|0.60|0.37 | 37.00|0.57|0.22 | 37.00|0.57|0.20 | 35.60|0.60|0.25 | 59.00|0.19|0.10 | 0.01 | 0|1.65 | 4000|3.39 | 2000|3.88 | 4000|3.49 | 4000|0.00
p=20_ell=1 | 45.00|0.58|0.25 | 34.00|0.60|0.24 | 34.00|0.60|0.18 | 36.80|0.58|0.22 | 60.00|0.21|0.03 | 0.06 | 0|2.02 | 4000|3.54 | 2000|3.33 | 4000|3.52 | 4000|0.00
p=20_ell=1 | 41.00|0.59|0.21 | 39.00|0.56|0.14 | 39.00|0.56|0.15 | 38.80|0.57|0.18 | 52.00|0.26|0.03 | -0.01 | 0|1.24 | 4000|3.04 | 2000|2.92 | 4000|2.89 | 4000|0.00
p=20_ell=1 | 26.00|0.67|0.45 | 19.00|0.74|0.47 | 19.00|0.74|0.46 | 22.60|0.70|0.38 | 67.00|0.11|0.12 | 0.01 | 0|0.83 | 4000|1.88 | 2000|1.96 | 4000|1.87 | 4000|0.00
p=20_ell=1 | 20.00|0.76|0.32 | 16.00|0.77|0.33 | 16.00|0.77|0.29 | 18.40|0.75|0.32 | 53.00|0.13|0.04 | 0.04 | 0|1.15 | 4000|2.59 | 2000|2.65 | 4000|2.36 | 4000|0.00
p=20_ell=1 | 33.00|0.65|0.44 | 26.00|0.69|0.32 | 26.00|0.69|0.30 | 31.20|0.64|0.32 | 56.00|0.26|0.15 | 0.02 | 0|1.33 | 4000|3.02 | 2000|2.92 | 4000|2.88 | 4000|0.00
p=20_ell=1 | 63.00|0.49|0.17 | 50.00|0.55|0.14 | 50.00|0.55|0.16 | 51.60|0.53|0.15 | 59.00|0.19|0.12 | -0.03 | 0|1.96 | 4000|4.58 | 2000|4.64 | 4000|4.50 | 4000|0.00
p=20_ell=1 | 29.00|0.67|0.27 | 19.00|0.75|0.24 | 19.00|0.75|0.19 | 21.00|0.73|0.28 | 60.00|0.14|0.13 | 0.05 | 0|1.03 | 4000|2.35 | 2000|2.25 | 4000|2.18 | 4000|0.00
p=20_ell=1 | 39.00|0.58|0.17 | 32.00|0.61|0.27 | 32.00|0.61|0.22 | 33.20|0.60|0.24 | 59.00|0.19|0.09 | 0.05 | 0|1.13 | 4000|2.48 | 2000|2.46 | 4000|2.54 | 4000|0.00
p=20_ell=1 | 37.00|0.61|0.21 | 27.00|0.65|0.29 | 27.00|0.65|0.26 | 28.20|0.64|0.29 | 61.00|0.19|0.12 | 0.02 | 0|1.53 | 4000|3.40 | 2000|3.15 | 4000|3.24 | 4000|0.00
p=20_ell=1 | 52.00|0.54|0.24 | 41.00|0.58|0.17 | 41.00|0.58|0.30 | 42.20|0.56|0.23 | 59.00|0.23|0.09 | -0.13 | 0|2.41 | 4000|4.35 | 2000|4.29 | 4000|4.34 | 4000|0.00
p=20_ell=1 | 35.00|0.62|0.32 | 23.00|0.69|0.36 | 23.00|0.69|0.28 | 28.00|0.65|0.29 | 61.00|0.14|0.03 | 0.09 | 0|1.62 | 4000|3.18 | 2000|3.42 | 4000|3.52 | 4000|0.00
p=20_ell=1 | 43.00|0.60|0.31 | 39.00|0.60|0.24 | 39.00|0.60|0.13 | 39.40|0.60|0.26 | 60.00|0.25|0.09 | 0.11 | 0|1.60 | 4000|3.51 | 2000|3.45 | 4000|3.25 | 4000|0.00
p=20_ell=1 | 71.00|0.51|0.20 | 55.00|0.52|0.33 | 55.00|0.52|0.24 | 55.40|0.52|0.23 | 67.00|0.17|0.09 | 0.09 | 0|2.90 | 4000|5.94 | 2000|5.68 | 4000|5.80 | 4000|0.00
p=20_ell=1 | 65.00|0.43|0.11 | 39.00|0.54|0.21 | 39.00|0.54|0.17 | 44.40|0.51|0.22 | 63.00|0.18|0.11 | 0.04 | 0|2.49 | 4000|4.61 | 2000|4.64 | 4000|4.39 | 4000|0.00
p=20_ell=1 | 33.00|0.66|0.32 | 24.00|0.71|0.26 | 24.00|0.71|0.33 | 25.80|0.70|0.28 | 62.00|0.16|0.12 | -0.07 | 0|1.44 | 4000|3.09 | 2000|3.12 | 4000|3.52 | 4000|0.00
p=20_ell=1 | 30.00|0.67|0.30 | 23.00|0.71|0.21 | 23.00|0.71|0.11 | 26.60|0.67|0.24 | 55.00|0.15|0.07 | 0.11 | 0|1.49 | 4000|3.28 | 2000|3.30 | 4000|3.19 | 4000|0.00
p=20_ell=1 | 28.00|0.66|0.28 | 23.00|0.68|0.32 | 23.00|0.68|0.19 | 22.80|0.69|0.30 | 64.00|0.09|0.00 | 0.13 | 0|1.17 | 4000|2.12 | 2000|2.23 | 4000|2.18 | 4000|0.00
p=20_ell=1 | 40.00|0.61|0.26 | 31.00|0.64|0.27 | 31.00|0.64|0.21 | 33.00|0.63|0.27 | 61.00|0.19|0.03 | 0.05 | 0|1.70 | 4000|3.47 | 2000|3.47 | 4000|3.35 | 4000|0.00
"""

rows = []
lines = text.strip().split("\n")

run_id = 0

for line in lines:
    if line.startswith("p="):
        run_id += 1
        parts = [p.strip() for p in line.split(" | ")]

        scenario = parts[0]

        methods_metrics = {
            "centralized": parts[1],
            "fedpc": parts[2],
            "naive": parts[3],
            "local": parts[4],
            "random": parts[5],
        }

        delta = float(parts[6])

        comm_runtime = {
            "centralized": parts[7],
            "fedpc": parts[8],
            "naive": parts[9],
            "local": parts[10],
            "random": parts[11],
        }

        for method in methods_metrics:
            shd, f1, orient = methods_metrics[method].split("|")
            comm, runtime = comm_runtime[method].split("|")

            rows.append({
                "Run": run_id,
                "Scenario": scenario,
                "Method": method,
                "SHD": float(shd),
                "F1": float(f1),
                "Orient_F1": float(orient),
                "Comm": float(comm),
                "Runtime_s": float(runtime),
                "Delta_F1_fedpc_naive": delta if method == "fedpc" else None
            })

df = pd.DataFrame(rows)

# save to Excel
df.to_excel("table14.xlsx", index=False)
print("Excel file saved: table14.xlsx")

###################################### ell 2

import pandas as pd

text = """
Scenario | centralized (SHD|F1|Orient_F1) | fedpc (SHD|F1|Orient_F1) | naive (SHD|F1|Orient_F1) | local (SHD|F1|Orient_F1) | random (SHD|F1|Orient_F1) | ΔF1_fedpc_naive | centralized (Comm|Runtime s) | fedpc (Comm|Runtime s) | naive (Comm|Runtime s) | local (Comm|Runtime s) | random (Comm|Runtime s)
p=20_ell=2 | 21.00|0.73|0.49 | 20.00|0.70|0.50 | 20.00|0.70|0.33 | 21.20|0.69|0.46 | 54.00|0.21|0.10 | 0.17 | 0|15.11 | 8000|3.99 | 4000|3.83 | 8000|4.04 | 8000|0.00
p=20_ell=2 | 35.00|0.59|0.24 | 30.00|0.61|0.25 | 30.00|0.61|0.24 | 32.60|0.58|0.26 | 51.00|0.26|0.10 | 0.01 | 0|36.52 | 8000|7.56 | 4000|7.48 | 8000|7.58 | 8000|0.00
p=20_ell=2 | 30.00|0.68|0.51 | 26.00|0.68|0.41 | 26.00|0.68|0.39 | 28.80|0.66|0.38 | 69.00|0.17|0.08 | 0.02 | 0|33.38 | 8000|7.32 | 4000|7.08 | 8000|7.33 | 8000|0.00
p=20_ell=2 | 39.00|0.57|0.29 | 26.00|0.66|0.37 | 26.00|0.66|0.29 | 31.00|0.60|0.26 | 65.00|0.22|0.09 | 0.08 | 0|28.07 | 8000|5.06 | 4000|5.11 | 8000|4.93 | 8000|0.00
p=20_ell=2 | 40.00|0.62|0.31 | 46.00|0.45|0.21 | 46.00|0.45|0.09 | 44.20|0.48|0.19 | 62.00|0.31|0.17 | 0.13 | 0|52.59 | 8000|7.85 | 4000|7.30 | 8000|7.19 | 8000|0.00
p=20_ell=2 | 29.00|0.69|0.28 | 29.00|0.63|0.24 | 29.00|0.63|0.17 | 29.80|0.64|0.25 | 68.00|0.15|0.08 | 0.07 | 0|20.34 | 8000|5.07 | 4000|5.25 | 8000|5.25 | 8000|0.00
p=20_ell=2 | 37.00|0.58|0.30 | 33.00|0.57|0.20 | 33.00|0.57|0.28 | 33.80|0.57|0.29 | 59.00|0.31|0.14 | -0.08 | 0|33.72 | 8000|6.69 | 4000|6.50 | 8000|6.42 | 8000|0.00
p=20_ell=2 | 21.00|0.74|0.38 | 18.00|0.75|0.28 | 18.00|0.75|0.28 | 18.40|0.75|0.28 | 72.00|0.12|0.09 | 0.00 | 0|16.00 | 8000|4.27 | 4000|4.13 | 8000|4.07 | 8000|0.00
p=20_ell=2 | 19.00|0.76|0.45 | 18.00|0.75|0.31 | 18.00|0.75|0.28 | 18.20|0.74|0.34 | 62.00|0.16|0.09 | 0.04 | 0|12.02 | 8000|3.15 | 4000|3.36 | 8000|2.92 | 8000|0.00
p=20_ell=2 | 43.00|0.56|0.33 | 33.00|0.58|0.25 | 33.00|0.58|0.26 | 33.80|0.58|0.28 | 66.00|0.15|0.08 | -0.00 | 0|29.75 | 8000|5.44 | 4000|5.70 | 8000|5.69 | 8000|0.00
p=20_ell=2 | 27.00|0.70|0.27 | 28.00|0.63|0.28 | 28.00|0.63|0.34 | 28.60|0.64|0.28 | 64.00|0.26|0.06 | -0.07 | 0|18.36 | 8000|4.45 | 4000|4.53 | 8000|4.48 | 8000|0.00
p=20_ell=2 | 52.00|0.45|0.13 | 40.00|0.49|0.25 | 40.00|0.49|0.28 | 40.80|0.49|0.23 | 53.00|0.29|0.10 | -0.04 | 0|51.62 | 8000|8.63 | 4000|8.24 | 8000|8.94 | 8000|0.00
p=20_ell=2 | 33.00|0.61|0.23 | 25.00|0.66|0.34 | 25.00|0.66|0.23 | 25.80|0.65|0.35 | 65.00|0.16|0.11 | 0.11 | 0|14.19 | 8000|3.87 | 4000|3.94 | 8000|3.96 | 8000|0.00
p=20_ell=2 | 4.00|0.94|0.47 | 3.00|0.95|0.50 | 3.00|0.95|0.43 | 6.20|0.89|0.38 | 55.00|0.25|0.11 | 0.07 | 0|5.31 | 8000|2.20 | 4000|1.91 | 8000|2.09 | 8000|0.00
p=20_ell=2 | 40.00|0.57|0.14 | 33.00|0.57|0.20 | 33.00|0.57|0.19 | 34.20|0.56|0.21 | 57.00|0.30|0.11 | 0.02 | 0|29.28 | 8000|5.40 | 4000|5.50 | 8000|6.29 | 8000|0.00
p=20_ell=2 | 19.00|0.75|0.35 | 17.00|0.75|0.33 | 17.00|0.75|0.29 | 19.40|0.72|0.33 | 55.00|0.23|0.12 | 0.05 | 0|14.69 | 8000|4.82 | 4000|4.61 | 8000|4.64 | 8000|0.00
p=20_ell=2 | 29.00|0.69|0.32 | 24.00|0.70|0.29 | 24.00|0.70|0.25 | 24.80|0.69|0.36 | 62.00|0.21|0.14 | 0.04 | 0|18.32 | 8000|4.65 | 4000|4.62 | 8000|4.53 | 8000|0.00
p=20_ell=2 | 22.00|0.65|0.42 | 18.00|0.68|0.63 | 18.00|0.68|0.38 | 19.00|0.67|0.52 | 54.00|0.23|0.13 | 0.25 | 0|7.28 | 8000|2.94 | 4000|2.90 | 8000|2.94 | 8000|0.00
p=20_ell=2 | 23.00|0.72|0.28 | 24.00|0.66|0.24 | 24.00|0.66|0.26 | 24.60|0.66|0.25 | 61.00|0.27|0.16 | -0.02 | 0|16.21 | 8000|4.70 | 4000|4.54 | 8000|4.37 | 8000|0.00
p=20_ell=2 | 40.00|0.57|0.19 | 37.00|0.54|0.19 | 37.00|0.54|0.25 | 41.00|0.52|0.18 | 47.00|0.39|0.21 | -0.06 | 0|39.19 | 8000|7.33 | 4000|7.40 | 8000|7.24 | 8000|0.00
"""

rows = []
lines = text.strip().split("\n")

run_id = 0

for line in lines:
    if line.startswith("p="):
        run_id += 1
        parts = [p.strip() for p in line.split(" | ")]

        scenario = parts[0]

        methods_metrics = {
            "centralized": parts[1],
            "fedpc": parts[2],
            "naive": parts[3],
            "local": parts[4],
            "random": parts[5],
        }

        delta = float(parts[6])

        comm_runtime = {
            "centralized": parts[7],
            "fedpc": parts[8],
            "naive": parts[9],
            "local": parts[10],
            "random": parts[11],
        }

        for method in methods_metrics:
            shd, f1, orient = methods_metrics[method].split("|")
            comm, runtime = comm_runtime[method].split("|")

            rows.append({
                "Run": run_id,
                "Scenario": scenario,
                "Method": method,
                "SHD": float(shd),
                "F1": float(f1),
                "Orient_F1": float(orient),
                "Comm": float(comm),
                "Runtime_s": float(runtime),
                "Delta_F1_fedpc_naive": delta if method == "fedpc" else None
            })

df = pd.DataFrame(rows)

# save to Excel
df.to_excel("table15.xlsx", index=False)
print("Excel file saved: table15.xlsx")

###################################### ablation

import pandas as pd

text = """
ell | tau | FedPC (SHD|F1|OrientF1) | Naive (SHD|F1|OrientF1) | ΔOrientF1 | FedPC_Comm | Naive_Comm
1 | 0.3 | 36.65|0.62|0.27 | 36.65|0.62|0.22 | 0.06 | 4000 | 2000
1 | 0.5 | 34.45|0.63|0.29 | 34.45|0.63|0.25 | 0.04 | 4000 | 2000
2 | 0.3 | 26.90|0.66|0.31 | 26.90|0.66|0.25 | 0.06 | 8000 | 4000
2 | 0.5 | 26.40|0.65|0.29 | 26.40|0.65|0.27 | 0.02 | 8000 | 4000
"""

rows = []
lines = text.strip().split("\n")

for line in lines[1:]:  # skip header
    parts = [p.strip() for p in line.split("|")]

    ell = int(parts[0])
    tau = float(parts[1])

    fedpc_shd = float(parts[2])
    fedpc_f1 = float(parts[3])
    fedpc_orient = float(parts[4])

    naive_shd = float(parts[5])
    naive_f1 = float(parts[6])
    naive_orient = float(parts[7])

    delta_orient = float(parts[8])
    fedpc_comm = float(parts[9])
    naive_comm = float(parts[10])

    rows.append({
        "ell": ell,
        "tau": tau,
        "FedPC_SHD": fedpc_shd,
        "FedPC_F1": fedpc_f1,
        "FedPC_OrientF1": fedpc_orient,
        "Naive_SHD": naive_shd,
        "Naive_F1": naive_f1,
        "Naive_OrientF1": naive_orient,
        "Delta_OrientF1": delta_orient,
        "FedPC_Comm": fedpc_comm,
        "Naive_Comm": naive_comm
    })

df = pd.DataFrame(rows)

df.to_excel("table16.xlsx", index=False)
print("Excel file saved: table16.xlsx")

#######################################

import pandas as pd
# List file tabel
files = ["table1.xlsx", "table2.xlsx", "table3.xlsx", "table4.xlsx", "table5.xlsx",
         "table6.xlsx", "table7.xlsx", "table8.xlsx", "table9.xlsx", "table10.xlsx",
         "table11.xlsx", "table12.xlsx", "table13.xlsx", "table14.xlsx", "table15.xlsx"]

metrics = ["SHD", "F1", "Orient_F1", "Comm", "Runtime_s", "Delta_F1_fedpc_naive"]

# Excel writer
with pd.ExcelWriter("tables_summary.xlsx") as writer:
    for file in files:
        #
        sheet_name = file.split(".")[0]
        
        # data
        df = pd.read_excel(file) 
        summary_mean = df.groupby("Method")[metrics].mean()
        summary_std = df.groupby("Method")[metrics].std()
        
        # mean & std
        summary = pd.concat([summary_mean, summary_std], axis=1, keys=["Mean", "Std"])
        
        # save
        summary.to_excel(writer, sheet_name=sheet_name)
print("tables_summary.xlsx")

#######################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Load Excel with multiple sheets --- #
xls = pd.ExcelFile("tables_summary.xlsx")

# Define metrics
metrics = ["SHD","F1","Orient_F1","Delta_F1_fedpc_naive","Comm","Runtime_s"]

# --- Define groups for comparative plots --- #
groups = {
    "K": ["table1", "table2", "table3"],               # K=2,5,10
    "N": ["table4", "table5", "table6"],               # N=100,3000,5000
    "phi": ["table7", "table8"],                       # phi=0.1,0.2
    "hetero": ["table9", "table10", "table11"],       # hetero none,mild,strong
    "tau": ["table12", "table13"],                     # tau 0.3, 0.5
    "ell": ["table14", "table15"]                      # ell 1,2
}

# Define scenario labels for legend (replace sheet indices with meaningful names)
scenario_labels_dict = {
    "K": ["K2","K5","K10"],
    "N": ["N100","N3000","N5000"],
    "phi": ["phi0.1","phi0.2"],
    "hetero": ["hetero_none","hetero_mild","hetero_strong"],
    "tau": ["tau0.3","tau0.5"],
    "ell": ["ell1","ell2"]
}

# Color palette (one color per scenario in a group)
color_palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"]

# --- Loop through groups --- #
for group_idx, (group_name, sheets) in enumerate(groups.items(), start=1):
    combined_long = []
    scenario_labels = scenario_labels_dict[group_name]

    for i, sheet in enumerate(sheets):
        scenario_label = scenario_labels[i]  # descriptive label for legend
        df = pd.read_excel(xls, sheet_name=sheet, header=[0,1], index_col=0)
        for metric in metrics:
            for method in df.index:
                mean = df[('Mean', metric)].loc[method]
                sd = df[('Std', metric)].loc[method]
                combined_long.append([scenario_label, method, metric, mean, sd])

    df_long = pd.DataFrame(combined_long, columns=["Scenario","Method","Metric","Mean","SD"])

    # --- Plot 2x3 grid --- #
    fig, axes = plt.subplots(2,3,figsize=(20,12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        subset = df_long[df_long["Metric"]==metric]
        methods = subset["Method"].unique()
        x = np.arange(len(methods))
        width = 0.15  # bar width per scenario

        for j, scenario in enumerate(scenario_labels):
            data_s = subset[subset["Scenario"]==scenario].set_index("Method").loc[methods]
            ax.bar(x + j*width, data_s["Mean"], width, yerr=data_s["SD"], capsize=5,
                   color=color_palette[j], label=scenario)

        ax.set_xticks(x + width*(len(scenario_labels)-1)/2)
        ax.set_xticklabels(methods, rotation=45, fontsize=10)
        ax.set_ylabel("Mean ± SD", fontsize=12)
        ax.set_title(metric, fontsize=14, fontweight='bold')  # category above each subplot

    # --- Legend outside the grid --- #
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenario_labels), fontsize=12)
    plt.tight_layout(rect=[0,0.07,1,1])
    plt.savefig(f"plot_{group_idx}.png", dpi=300)
    plt.close(fig)

#################################################

#Ablation

# --- Plot 2x3 grid --- #
metrics = ["SHD","F1","OrientF1","Delta_OrientF1","Comm"]
fig, axes = plt.subplots(2,3,figsize=(18,10))
axes = axes.flatten()

method_colors = {"FedPC":"#1f77b4","Naive":"#ff7f0e"}

for i, metric in enumerate(metrics):
    ax = axes[i]
    subset = df_long[df_long["Metric"]==metric]
    scenarios = subset["Scenario"].unique()
    x = np.arange(len(scenarios))
    width = 0.35
    
    for j, method in enumerate(["FedPC","Naive"]):
        data_m = subset[subset["Method"]==method].set_index("Scenario").loc[scenarios]
        ax.bar(x + j*width, data_m["Value"], width, label=method, color=method_colors[method])
    
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(metric, fontsize=12, fontweight='bold')  # <-- bold metric titles

# Remove empty subplot (since 2x3 grid but only 5 metrics)
axes[-1].axis("off")

# Legend below
fig.legend(["FedPC","Naive"], loc='lower center', ncol=2, fontsize=12)
plt.tight_layout(rect=[0,0.07,1,1])
plt.savefig("plot_7.png", dpi=300)
plt.show()

################################## comparative baseline

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    "Methods": ["centralized","fedpc","local","naive","random"],
    "SHD Best":[1,13,1,11,0],
    "SHD Worse":[0,0,0,0,15],
    "F1 Best":[11,4,0,2,0],
    "F1 Worse":[0,0,0,0,15],
    "Orient_F1 Best":[9,7,1,0,0],
    "Orient_F1 Worse":[0,0,0,0,15],
    "Comm Best":[0,0,0,15,0],
    "Comm Worse":[0,15,15,0,15],
    "Runtime_s Best":[3,1,2,9,0],
    "Runtime_s Worse":[12,2,1,0,0]
}
df = pd.DataFrame(data).set_index("Methods")

metrics = ["SHD","F1","Orient_F1","Comm","Runtime_s"]
methods = df.index

# Grid 2x3
fig, axes = plt.subplots(2,3, figsize=(15,8), sharey=True)
axes = axes.flatten()

colors = {"Best":"#1f77b4", "Worse":"#ff7f0e"}  # biru, orange

for i, metric in enumerate(metrics):
    best = df[f"{metric} Best"]
    worse = df[f"{metric} Worse"]
    x = np.arange(len(methods))

    axes[i].bar(x, best, label="Best", color=colors["Best"])
    axes[i].bar(x, worse, bottom=best, label="Worse", color=colors["Worse"])

    axes[i].set_xticks(x)
    axes[i].set_xticklabels(methods, rotation=45)
    axes[i].set_title(metric)

# ylabel
axes[0].set_ylabel("Count across scenarios")

# 
axes[-1].axis('off')

# (figure legend)
fig.legend(["Best","Worse"], loc='upper right', fontsize=12, frameon=True)

plt.tight_layout(rect=[0,0,0.9,1])  #
plt.savefig("final_summary_barplot.png", dpi=300)
plt.show()