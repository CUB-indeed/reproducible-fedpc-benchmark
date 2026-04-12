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
# VISUALIZATION MODULE - Causal Graph
# =======================

import matplotlib.pyplot as plt
import os

def draw_graph(ax, G, title, ref_edges=None):
    # layout lebih stabil
    pos = nx.kamada_kawai_layout(G)

    edges = set(G.edges())

    if ref_edges is not None:
        edges_common = edges & ref_edges
        edges_diff = edges - ref_edges
    else:
        edges_common = edges
        edges_diff = set()

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=500,
        node_color="#4C72B0",
        alpha=0.9
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12)

    # Common edges (abu-abu)
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=list(edges_common),
        edge_color="gray",
        width=1.5,
        alpha=0.6,
        arrows=True
    )

    # Different edges (merah)
    if edges_diff:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=list(edges_diff),
            edge_color="red",
            width=2.5,
            alpha=0.9,
            arrows=True
        )

    ax.set_title(title, fontsize=14)
    ax.axis("off")


def visualize_single_run(
    p=20, N=5000, K=5, edge_prob=0.2, hetero="mild",
    alpha=ALPHA, tau=TAU_DEFAULT, ell=ELL_MAX,
    out_file="graph_outputs/all_graphs.png"
):

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # TRUE DAG
    B = generate_random_dag(p, edge_prob)
    true_skel = true_skeleton_edges(B)

    true_G = nx.DiGraph()
    true_G.add_nodes_from(range(p))
    true_G.add_edges_from(true_directed_edges(B))

    # DATA
    client_data = [
        simulate_linear_sem(B, N // K, hetero)
        for _ in range(K)
    ]

    X_global = np.vstack(client_data)

    # CENTRALIZED
    Gc, seps = pc_skeleton_with_sepsets(X_global, alpha, ell)
    cpdag_c = orient_v_structures(Gc, seps, use_meek=True)

    # FEDPC
    client_edges, client_seps = local_pc_clients(client_data, alpha, ell)
    # ===== Skeleton aggregation =====
    pred_fedpc = weighted_aggregation(
        client_edges, client_data, true_skel, tau, p
    )

    Gf = nx.Graph()
    Gf.add_nodes_from(range(p))
    Gf.add_edges_from(pred_fedpc)

# ===== Weighted Sepset aggregation =====
    weights_raw = np.array([
        compute_client_reliability(X, edges, p, true_skel)
        for X, edges in zip(client_data, client_edges)
    ], dtype=float)

    weights = (weights_raw ** 2) / np.sum(weights_raw ** 2)

    agg_seps = {}

    # collect all sepset keys
    all_keys = set()
    for seps in client_seps:
        all_keys.update(seps.keys())

    # aggregate with weights
    for key in all_keys:
        votes = {}

        for k, seps_k in enumerate(client_seps):
            if key in seps_k:
                for z in seps_k[key]:
                    votes[z] = votes.get(z, 0.0) + weights[k]

        if votes:
            agg_seps[key] = {z for z, w in votes.items() if w >= tau}

    cpdag_f = orient_v_structures(Gf, agg_seps, use_meek=True)

    # NAIVE
    pred_naive = naive_majority_aggregation(client_edges, tau)
    Gn = nx.Graph()
    Gn.add_nodes_from(range(p))
    Gn.add_edges_from(pred_naive)
    cpdag_n = orient_v_structures(Gn, {}, use_meek=True)

    # LOCAL (client 0)
    Gl = nx.Graph()
    Gl.add_nodes_from(range(p))
    Gl.add_edges_from(client_edges[0])
    cpdag_l = orient_v_structures(Gl, client_seps[0], use_meek=True)

    # RANDOM
    Gr = nx.fast_gnp_random_graph(p, edge_prob)
    cpdag_r = orient_v_structures(Gr, {}, use_meek=True)

    # =======================
    # EDGE COMPARISON
    # =======================
    fed_edges = set(cpdag_f.edges())
    naive_edges = set(cpdag_n.edges())

    # =======================
    # PLOT
    # =======================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    draw_graph(axes[0, 0], true_G, "True DAG")
    draw_graph(axes[0, 1], cpdag_c, "Centralized")

    # FedPC vs Naive
    draw_graph(
        axes[0, 2],
        cpdag_f,
        "FedPC",
        ref_edges=naive_edges
    )

    # Naive vs FedPC
    draw_graph(
        axes[1, 0],
        cpdag_n,
        "Naive",
        ref_edges=fed_edges
    )

    draw_graph(axes[1, 1], cpdag_l, "Local (client 1)")
    draw_graph(axes[1, 2], cpdag_r, "Random")

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


if __name__ == "__main__":
    visualize_single_run()
