import os
import time
import random
import hashlib
from dataclasses import dataclass, asdict, replace
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import pearsonr, norm, t as t_dist
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MASTER_SEED = 42
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

PALETTE = {
    "centralized": "#2c3e50", "local": "#7f8c8d",
    "no_agg_equal": "#e67e22", "naive": "#e67e22",
    "agg_equal": "#9b59b6",
    "agg_consensus": "#27ae60", "fedpc_consensus": "#27ae60",
    "agg_oracle": "#3498db", "fedpc_oracle": "#3498db",
}
METHOD_LABELS = {
    "centralized": "Centralized", "local": "Local PC (mean, diff. estimand)",
    "no_agg_equal": "No Sepset-agg (equal)", "naive": "No Sepset-agg (equal)",
    "agg_equal": "Sepset-agg, equal weight",
    "agg_consensus": "Sepset-agg, consensus weight", "fedpc_consensus": "FedPC-Consensus",
    "agg_oracle": "Sepset-agg, oracle weight (ablation)", "fedpc_oracle": "FedPC-Oracle (ablation)",
}

# ==============================================================================
# SECTION 1 -- CORE FedPC ALGORITHM (skeleton / Sepsets / collider / Meek)
# Verbatim from the validated production core. Not modified.
# ==============================================================================

def safe_pearsonr(x, y):
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0, 1.0
    r, p = pearsonr(x, y)
    if not np.isfinite(r):
        return 0.0, 1.0
    return float(np.clip(r, -0.999999, 0.999999)), p


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
    sepsets = {}
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
                cand_pools = [
                    [v for v in adj_snapshot[i] if v != j],
                    [v for v in adj_snapshot[j] if v != i],
                ]
                removed = False
                for pool in cand_pools:
                    if removed:
                        break
                    if len(pool) < l:
                        continue
                    for cond in combinations(pool, l):
                        if ci_test(X, i, j, cond, alpha):
                            if G.has_edge(i, j):
                                G.remove_edge(i, j)
                                sepsets[(i, j)] = set(cond)
                                sepsets[(j, i)] = set(cond)
                            removed = True
                            break
        l += 1
        if l > ell_max:
            break
    return G, sepsets


def get_sepset(sepsets, i, j):
    # Distinguishes "no separating-set evidence available" (key absent,
    # returns None) from "separated by the empty set" (key present with
    # an empty set) -- see module docstring point 13.
    if (i, j) in sepsets:
        return sepsets[(i, j)]
    if (j, i) in sepsets:
        return sepsets[(j, i)]
    return None


def make_cpdag_from_skeleton(G):
    cpdag = nx.DiGraph()
    cpdag.add_nodes_from(G.nodes())
    for u, v in G.edges():
        cpdag.add_edge(u, v)
        cpdag.add_edge(v, u)
    return cpdag


def is_undirected(cpdag, u, v):
    return cpdag.has_edge(u, v) and cpdag.has_edge(v, u)


def is_fully_directed(cpdag, u, v):
    return cpdag.has_edge(u, v) and not cpdag.has_edge(v, u)


def orient_edge(cpdag, u, v):
    if cpdag.has_edge(v, u):
        cpdag.remove_edge(v, u)


def apply_meek_rules(cpdag):
    changed = True
    while changed:
        changed = False
        nodes = list(cpdag.nodes())
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
        for c in nodes:
            parents_c = [x for x in cpdag.predecessors(c) if not is_undirected(cpdag, x, c)]
            for b, d in combinations(parents_c, 2):
                if cpdag.has_edge(b, d) or cpdag.has_edge(d, b):
                    continue
                for a in nodes:
                    if a in (b, c, d):
                        continue
                    if (is_undirected(cpdag, a, b) and is_undirected(cpdag, a, d)
                            and is_undirected(cpdag, a, c)):
                        orient_edge(cpdag, a, c)
                        changed = True
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
                        if (is_undirected(cpdag, a, b) and is_undirected(cpdag, a, d)
                                and not cpdag.has_edge(a, c) and not cpdag.has_edge(c, a)):
                            orient_edge(cpdag, a, d)
                            changed = True
    return cpdag


def _all_unshielded_triples(G):
    triples = set()
    for b in G.nodes():
        neighbors_b = list(G.neighbors(b))
        for a, c in combinations(neighbors_b, 2):
            if G.has_edge(a, c):
                continue
            lo, hi = (a, c) if a < c else (c, a)
            triples.add((lo, hi, b))
    return sorted(triples)


def orient_v_structures(G, sepsets, use_meek=True):
    cpdag = make_cpdag_from_skeleton(G)
    triples = _all_unshielded_triples(G)
    for (a, c, b) in triples:
        sep_ac = get_sepset(sepsets, a, c)
        sep_ca = get_sepset(sepsets, c, a)
        if sep_ac is None and sep_ca is None:
            continue  # no separating-set evidence available -> do not orient
        combined_sep = set()
        if sep_ac is not None:
            combined_sep |= sep_ac
        if sep_ca is not None:
            combined_sep |= sep_ca
        if b not in combined_sep:  # includes the case combined_sep == set()
            opposes_a = is_fully_directed(cpdag, b, a)
            opposes_c = is_fully_directed(cpdag, b, c)
            if opposes_a or opposes_c:
                continue
            orient_edge(cpdag, a, b)
            orient_edge(cpdag, c, b)
    if use_meek:
        cpdag = apply_meek_rules(cpdag)
    return cpdag


SEP_TAU_FACTOR_DEFAULT = 0.5


def aggregate_sepsets(client_seps, weights, tau=0.5, sep_tau_factor=SEP_TAU_FACTOR_DEFAULT):
    """
    Aggregates per-client Sepsets under weighted voting.

    Two distinct outcomes per candidate pair (i, j), by design:
      (a) evidence_weight/total_weight < sep_tau  -> pair is OMITTED from
          the returned dict entirely. get_sepset() will return None for
          it later, and orient_v_structures will not attempt to orient
          any triple through it: "no separating-set evidence available".
      (b) evidence_weight/total_weight >= sep_tau -> pair IS included,
          with `included` possibly EMPTY if no single conditioning
          variable individually clears sep_tau. An empty included set is
          a legitimate outcome ("separated by nothing", i.e. marginal
          independence) and correctly triggers the standard PC collider
          rule for any unshielded triple through that pair.
    """
    total_weight = sum(weights)
    if total_weight < 1e-12:
        return {}
    sep_tau = tau * sep_tau_factor
    all_keys = set()
    for seps in client_seps:
        all_keys.update(seps.keys())
    agg_seps = {}
    for key in all_keys:
        evidence_weight = sum(weights[k] for k, seps_k in enumerate(client_seps) if key in seps_k)
        if evidence_weight / total_weight < sep_tau:
            continue
        votes = {}
        for k, seps_k in enumerate(client_seps):
            if key in seps_k:
                for z in seps_k[key]:
                    votes[z] = votes.get(z, 0.0) + weights[k]
        included = {z for z, w in votes.items() if (w / total_weight) >= sep_tau}
        agg_seps[key] = included
    return agg_seps


def skeleton_edges_from_graph(G):
    return set(tuple(sorted(e)) for e in G.edges())


def directed_edges_from_cpdag(cpdag):
    return {(i, j) for i, j in cpdag.edges() if not cpdag.has_edge(j, i)}


def true_skeleton_edges(B):
    return set(tuple(sorted((i, j))) for i, j in zip(*np.where(B != 0)))


def true_directed_edges(B):
    return set(zip(*np.where(B != 0)))


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


# ==============================================================================
# SECTION 2 -- FEDERATED HELPERS (data sim, splitting, weighting, aggregation)
# Verbatim from the validated production core. Not modified.
# ==============================================================================

def generate_random_dag(p, edge_prob, rng):
    B = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            if rng.random() < edge_prob:
                B[i, j] = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
    return B


def simulate_linear_sem(B, n_samples, rng, noise_regime=None, B_perturbed=None):
    """noise_regime in {None, 'mild', 'strong'} scales noise PER-OBSERVATION
    (sample-wise eta_t), i.e. within-pool heteroskedasticity -- NOT
    client-level heterogeneity. See module docstring point 7."""
    Buse = B_perturbed if B_perturbed is not None else B
    p = Buse.shape[0]
    X = np.zeros((n_samples, p))
    G_temp = nx.DiGraph(); G_temp.add_nodes_from(range(p))
    for i in range(p):
        for j in range(p):
            if Buse[i, j] != 0:
                G_temp.add_edge(i, j)
    order = list(nx.topological_sort(G_temp))
    for i in order:
        parents = np.where(Buse[:, i] != 0)[0]
        noise = rng.normal(0, 1, n_samples)
        if noise_regime == "mild":
            eta = rng.uniform(0.8, 1.2, n_samples)
            noise = noise * eta
        elif noise_regime == "strong":
            eta = rng.uniform(0.5, 1.5, n_samples)
            noise = noise * eta
        if len(parents):
            X[:, i] = X[:, parents] @ Buse[parents, i] + noise
        else:
            X[:, i] = noise
    return X


def perturb_B(B, rng, drop_frac=0.07, coeff_noise=0.2):
    """Genuine CLIENT-LEVEL heterogeneity: each client gets its own
    perturbed structural DAG. See module docstring point 7."""
    Bp = B.copy()
    edges = list(zip(*np.where(B != 0)))
    if not edges:
        return Bp
    n_drop = max(1, int(len(edges) * drop_frac))
    idx = rng.choice(len(edges), size=min(n_drop, len(edges)), replace=False)
    for k in idx:
        i, j = edges[k]
        Bp[i, j] = 0.0
    for i, j in zip(*np.where(Bp != 0)):
        Bp[i, j] *= rng.uniform(1 - coeff_noise, 1 + coeff_noise)
    return Bp


def federated_split(X, K, rng):
    idx = rng.permutation(X.shape[0])
    return [X[idx[i::K]] for i in range(K)]


def local_pc_clients_timed(client_data, alpha, ell):
    def run_pc(Xc):
        t0 = time.time()
        Gc, sepsc = pc_skeleton_with_sepsets(Xc, alpha, ell)
        dt = time.time() - t0
        return skeleton_edges_from_graph(Gc), sepsc, dt
    results = Parallel(n_jobs=-1)(delayed(run_pc)(Xc) for Xc in client_data)
    edges = [r[0] for r in results]
    seps = [r[1] for r in results]
    runtimes = [r[2] for r in results]
    return edges, seps, runtimes


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
    weights = np.array(weights, dtype=float) ** 2
    total = weights.sum()
    return weights / total if total > 1e-12 else np.ones(K) / K


def compute_oracle_reliability(client_edge_sets, true_skel):
    weights = []
    for edges in client_edge_sets:
        _, _, _, f1 = compute_metrics(true_skel, edges)
        weights.append(np.clip(f1, 0.05, 1.0))
    weights = np.array(weights, dtype=float) ** 2
    total = weights.sum()
    return weights / total if total > 1e-12 else np.ones(len(client_edge_sets)) / len(client_edge_sets)


def weighted_aggregation_with_weights(client_edge_sets, weights, tau=0.5):
    total_weight = sum(weights)
    if total_weight < 1e-12:
        return set()
    counter = {}
    for k, edges in enumerate(client_edge_sets):
        for e in edges:
            counter[e] = counter.get(e, 0.0) + weights[k]
    return {e for e, w in counter.items() if (w / total_weight) >= tau}


def communication_cost_fixed(client_edges, client_seps, p, ell_max, per_client=False):
    """Corrected formula (Reviewer point 13): payload scales with the
    Sepsets recorded for SEPARATED pairs, never with retained edges
    (retained edges have no Sepset). Denser graphs -> fewer separated
    pairs -> smaller payload."""
    n_pairs = p * (p - 1) / 2
    bits_per_index = np.ceil(np.log2(p)) if p > 1 else 1.0
    per_client_bytes = []
    for edges, seps in zip(client_edges, client_seps):
        skeleton_bits = n_pairs
        seen_pairs, total_cond_elements = set(), 0
        for (i, j), cond in seps.items():
            key = tuple(sorted((i, j)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            total_cond_elements += len(cond)
        sepset_bits = total_cond_elements * bits_per_index
        per_client_bytes.append(skeleton_bits / 8.0 + sepset_bits / 8.0)
    total_bytes = sum(per_client_bytes)
    if per_client:
        return total_bytes, per_client_bytes
    return total_bytes


@dataclass
class RuntimeRecord:
    client_runtimes: list
    server_runtime: float

    @property
    def runtime_sum(self) -> float:
        return sum(self.client_runtimes) + self.server_runtime

    @property
    def runtime_parallel_est(self) -> float:
        return (max(self.client_runtimes) if self.client_runtimes else 0.0) + self.server_runtime

    def as_dict(self):
        return {"runtime_sum": self.runtime_sum, "runtime_parallel_est": self.runtime_parallel_est,
                "server_runtime": self.server_runtime, "n_clients_timed": len(self.client_runtimes)}


# ==============================================================================
# SECTION 3 -- DETERMINISTIC SEED MANAGER (Reviewer point 4)
# ==============================================================================

class SeedManager:
    def __init__(self, master_seed: int = MASTER_SEED):
        self.master_seed = master_seed
        self._root = np.random.SeedSequence(master_seed)

    def _scenario_seq(self, scenario_id: str):
        scenario_key = int.from_bytes(scenario_id.encode("utf-8")[:16].ljust(16, b"\0"), "little") % (2**31 - 1)
        return np.random.SeedSequence([self.master_seed, scenario_key])

    def replicate_seed(self, scenario_id: str, replicate_id: int) -> int:
        scen_seq = self._scenario_seq(scenario_id)
        child = scen_seq.spawn(replicate_id + 1)[replicate_id]
        return int(child.generate_state(1, dtype=np.uint32)[0])

    def replicate_rng(self, scenario_id: str, replicate_id: int) -> np.random.Generator:
        return np.random.default_rng(self.replicate_seed(scenario_id, replicate_id))

    def client_seeds(self, scenario_id: str, replicate_id: int, n_clients: int):
        rep_seq = np.random.SeedSequence([self.master_seed, self.replicate_seed(scenario_id, replicate_id)])
        children = rep_seq.spawn(n_clients)
        return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]

    def record(self, scenario_id: str, replicate_id: int, n_clients: int = None) -> dict:
        out = {"master_seed": self.master_seed, "scenario_id": scenario_id,
               "replicate_id": replicate_id, "replicate_seed": self.replicate_seed(scenario_id, replicate_id)}
        if n_clients is not None:
            out["client_seeds"] = self.client_seeds(scenario_id, replicate_id, n_clients)
        return out


# ==============================================================================
# SECTION 4 -- CONFIG / SCENARIO REGISTRY (Reviewer point 5: no duplicate
# configs silently double-counted in the multiplicity correction)
# ==============================================================================

@dataclass(frozen=True)
class ExperimentConfig:
    p: int = 20
    N: int = 5000
    K: int = 5
    tau: float = 0.5
    edge_prob: float = 0.2
    ell: int = 2
    noise_regime: str = "mild"
    mechanism_shift: bool = False
    deletion_probability: float = 0.07
    alpha: float = 0.05
    sep_tau_factor: float = 0.5

    def as_dict(self):
        return asdict(self)


CANONICAL_DEFAULT = ExperimentConfig()


def config_id(cfg: ExperimentConfig) -> str:
    key = tuple(sorted(cfg.as_dict().items()))
    return hashlib.sha1(repr(key).encode()).hexdigest()[:10]


def make_scenario_id(cfg: ExperimentConfig, tag: str) -> str:
    return f"{tag}_{config_id(cfg)}"


def build_main_scenario_registry():
    """Sensitivity-sweep scenarios (K, N, phi, noise-scaling, tau, ell) +
    the canonical default. Mechanism-shift and corruption are handled by
    their own dedicated sweeps below (they need extra machinery), so
    they are NOT part of this registry -- avoids the exact
    same-config-counted-twice failure mode Reviewer #3 flagged."""
    rows = []

    def add(cfg, tag, family):
        sid = make_scenario_id(cfg, tag)
        rows.append({"scenario_id": sid, "family": family, "config_id": config_id(cfg), **cfg.as_dict()})

    add(CANONICAL_DEFAULT, "default", "default")
    for K in [2, 5, 10]:
        if K == CANONICAL_DEFAULT.K:
            continue
        add(replace(CANONICAL_DEFAULT, K=K), f"K{K}", "federation_scale")
    for N in [100, 3000, 5000]:
        if N == CANONICAL_DEFAULT.N:
            continue
        add(replace(CANONICAL_DEFAULT, N=N), f"N{N}", "sample_size")
    for phi in [0.1, 0.2]:
        if phi == CANONICAL_DEFAULT.edge_prob:
            continue
        add(replace(CANONICAL_DEFAULT, edge_prob=phi), f"phi{phi}", "graph_density")
    for noise in [None, "mild", "strong"]:
        label = noise if noise is not None else "none"
        if noise == CANONICAL_DEFAULT.noise_regime:
            continue
        add(replace(CANONICAL_DEFAULT, noise_regime=noise), f"noise_{label}", "noise_scaling")
    for tau in [0.3, 0.5]:
        if tau == CANONICAL_DEFAULT.tau:
            continue
        add(replace(CANONICAL_DEFAULT, tau=tau), f"tau{tau}", "tau_threshold")
    for ell in [1, 2]:
        if ell == CANONICAL_DEFAULT.ell:
            continue
        add(replace(CANONICAL_DEFAULT, ell=ell), f"ell{ell}", "conditioning_depth")

    df = pd.DataFrame(rows)
    dup = df[df.duplicated(subset=["config_id"], keep=False)]
    if not dup.empty:
        raise ValueError(f"Duplicate config_id in main registry:\n{dup[['scenario_id','config_id']]}")
    return df



# ==============================================================================
# SECTION 5 -- MAIN REPLICATE (centralized / local / naive / consensus / oracle)
# Used for the sensitivity-sweep registry above AND the mechanism-shift sweep.
# ==============================================================================

def run_replicate(cfg: ExperimentConfig, scenario_id: str, replicate_id: int,
                   seed_manager: SeedManager, mechshift_targets=False) -> list:
    rep_rng = seed_manager.replicate_rng(scenario_id, replicate_id)
    client_seeds = seed_manager.client_seeds(scenario_id, replicate_id, cfg.K)
    seed_record = seed_manager.record(scenario_id, replicate_id, cfg.K)

    p, N, K = cfg.p, cfg.N, cfg.K
    B = generate_random_dag(p, cfg.edge_prob, rep_rng)
    true_skel = true_skeleton_edges(B)
    true_dir = true_directed_edges(B)

    extra_targets = {}
    if cfg.mechanism_shift:
        client_data, B_clients = [], []
        for k in range(K):
            crng = np.random.default_rng(client_seeds[k])
            Bc = perturb_B(B, crng, drop_frac=cfg.deletion_probability)
            n_k = N // K
            Xk = simulate_linear_sem(B, n_k, crng, noise_regime=cfg.noise_regime, B_perturbed=Bc)
            client_data.append(Xk)
            B_clients.append(Bc)
        if mechshift_targets:
            # Cheap: reuses the already-simulated per-client DAGs, no extra PC runs.
            edge_counts = {}
            for Bc in B_clients:
                for e in true_skeleton_edges(Bc):
                    edge_counts[e] = edge_counts.get(e, 0) + 1
            extra_targets = {
                "target_base_dag": true_skel,
                "target_union": set(edge_counts.keys()),
                "target_intersection": {e for e, c in edge_counts.items() if c == K},
                "target_prevalence50": {e for e, c in edge_counts.items() if c / K >= 0.5},
            }
    else:
        X_pool = simulate_linear_sem(B, N, rep_rng, noise_regime=cfg.noise_regime)
        client_data = federated_split(X_pool, K, rep_rng)

    X_global = np.vstack(client_data)
    rows = []

    def base_row(method, evaluation_unit):
        return {"scenario_id": scenario_id, "replicate_id": replicate_id, "method": method,
                "evaluation_unit": evaluation_unit, "evaluation_target": "true_dag",
                **cfg.as_dict(), **seed_record}

    def add_targets(row, pred_skel):
        # Only meaningful for the aggregated federated skeleton; cheap since
        # pred_skel is already computed.
        for tname, tskel in extra_targets.items():
            _, _, _, tf1 = compute_metrics(tskel, pred_skel)
            row[f"F1_{tname}"] = tf1

    # ---- Centralized ----
    t0 = time.time()
    Gc, seps_c = pc_skeleton_with_sepsets(X_global, cfg.alpha, cfg.ell)
    pred_skel_c = skeleton_edges_from_graph(Gc)
    cpdag_c = orient_v_structures(Gc, seps_c)
    server_time_c = time.time() - t0
    shd, prec, rec, f1 = compute_metrics(true_skel, pred_skel_c)
    n_or, dp, dr, df1 = orientation_metrics_cpdag(true_dir, cpdag_c)
    row = base_row("centralized", "global_graph")
    row.update({"SHD": shd, "F1": f1, "Precision": prec, "Recall": rec, "n_oriented": n_or,
                "Dir_Prec": dp, "Dir_Rec": dr, "Dir_F1": df1, "communication_bytes": 0.0,
                **RuntimeRecord([server_time_c], 0.0).as_dict()})
    rows.append(row)

    # ---- Local PC per client (shared across naive/consensus/oracle) ----
    client_edges, client_seps, client_runtimes = local_pc_clients_timed(client_data, cfg.alpha, cfg.ell)

    # ---- Local baseline: DIFFERENT ESTIMAND (mean over per-client graphs,
    # not a single global graph) -- flagged explicitly (Reviewer point 9) ----
    t0 = time.time()
    local_shds, local_f1s, local_df1s, local_precs, local_recs = [], [], [], [], []
    for edges, seps in zip(client_edges, client_seps):
        Gcl = nx.Graph(); Gcl.add_nodes_from(range(p)); Gcl.add_edges_from(edges)
        cpdag_l = orient_v_structures(Gcl, seps)
        s, pr, rc, f = compute_metrics(true_skel, edges)
        _, _, _, df = orientation_metrics_cpdag(true_dir, cpdag_l)
        local_shds.append(s); local_f1s.append(f); local_df1s.append(df); local_precs.append(pr); local_recs.append(rc)
    server_time_local = time.time() - t0
    comm_local = communication_cost_fixed(client_edges, client_seps, p, cfg.ell)
    row = base_row("local", "mean_local_graph")
    row.update({"SHD": float(np.mean(local_shds)), "F1": float(np.mean(local_f1s)),
                "Precision": float(np.mean(local_precs)), "Recall": float(np.mean(local_recs)),
                "n_oriented": 0, "Dir_Prec": 0.0, "Dir_Rec": 0.0, "Dir_F1": float(np.mean(local_df1s)),
                "communication_bytes": comm_local,
                **RuntimeRecord(client_runtimes, server_time_local).as_dict()})
    rows.append(row)

    # ---- FedPC-Naive: no Sepset aggregation, equal weighting (structural
    # zero for orientation BY CONSTRUCTION -- see Reviewer point 2) ----
    t0 = time.time()
    pred_naive = naive_majority_aggregation(client_edges, cfg.tau)
    Gn = nx.Graph(); Gn.add_nodes_from(range(p)); Gn.add_edges_from(pred_naive)
    cpdag_n = orient_v_structures(Gn, {})
    server_time_n = time.time() - t0
    shd, prec, rec, f1 = compute_metrics(true_skel, pred_naive)
    n_or, dp, dr, df1 = orientation_metrics_cpdag(true_dir, cpdag_n)
    comm_naive = communication_cost_fixed(client_edges, client_seps, p, cfg.ell)
    row = base_row("naive", "global_graph")
    row.update({"SHD": shd, "F1": f1, "Precision": prec, "Recall": rec, "n_oriented": n_or,
                "Dir_Prec": dp, "Dir_Rec": dr, "Dir_F1": df1, "communication_bytes": comm_naive,
                "client_weights": (np.ones(K) / K).tolist(),
                "note": "n_oriented=0 / Dir_F1=0 BY CONSTRUCTION (no Sepsets passed to "
                        "orientation) -- definitional, not independent evidence of a "
                        "weighting effect; see factorial ablation for that comparison.",
                **RuntimeRecord(client_runtimes, server_time_n).as_dict()})
    add_targets(row, pred_naive)
    rows.append(row)

    # ---- FedPC-Consensus (Sepset aggregation ON, reliability weighting) ----
    t0 = time.time()
    w_con = compute_consensus_reliability(client_edges)
    pred_con = weighted_aggregation_with_weights(client_edges, w_con, cfg.tau)
    agg_seps_con = aggregate_sepsets(client_seps, w_con, cfg.tau, cfg.sep_tau_factor)
    Gcon = nx.Graph(); Gcon.add_nodes_from(range(p)); Gcon.add_edges_from(pred_con)
    cpdag_con = orient_v_structures(Gcon, agg_seps_con)
    server_time_con = time.time() - t0
    shd, prec, rec, f1 = compute_metrics(true_skel, pred_con)
    n_or, dp, dr, df1 = orientation_metrics_cpdag(true_dir, cpdag_con)
    comm_con = communication_cost_fixed(client_edges, client_seps, p, cfg.ell)
    row = base_row("fedpc_consensus", "global_graph")
    row.update({"SHD": shd, "F1": f1, "Precision": prec, "Recall": rec, "n_oriented": n_or,
                "Dir_Prec": dp, "Dir_Rec": dr, "Dir_F1": df1, "communication_bytes": comm_con,
                "client_weights": w_con.tolist(), "n_sepset_pairs_retained": len(agg_seps_con),
                **RuntimeRecord(client_runtimes, server_time_con).as_dict()})
    add_targets(row, pred_con)
    rows.append(row)

    # ---- FedPC-Oracle (ablation upper bound; requires ground truth) ----
    t0 = time.time()
    w_ora = compute_oracle_reliability(client_edges, true_skel)
    pred_ora = weighted_aggregation_with_weights(client_edges, w_ora, cfg.tau)
    agg_seps_ora = aggregate_sepsets(client_seps, w_ora, cfg.tau, cfg.sep_tau_factor)
    Gora = nx.Graph(); Gora.add_nodes_from(range(p)); Gora.add_edges_from(pred_ora)
    cpdag_ora = orient_v_structures(Gora, agg_seps_ora)
    server_time_ora = time.time() - t0
    shd, prec, rec, f1 = compute_metrics(true_skel, pred_ora)
    n_or, dp, dr, df1 = orientation_metrics_cpdag(true_dir, cpdag_ora)
    comm_ora = communication_cost_fixed(client_edges, client_seps, p, cfg.ell)
    row = base_row("fedpc_oracle", "global_graph")
    row.update({"SHD": shd, "F1": f1, "Precision": prec, "Recall": rec, "n_oriented": n_or,
                "Dir_Prec": dp, "Dir_Rec": dr, "Dir_F1": df1, "communication_bytes": comm_ora,
                "client_weights": w_ora.tolist(), "n_sepset_pairs_retained": len(agg_seps_ora),
                "note": "ablation only -- requires ground truth, not deployable",
                **RuntimeRecord(client_runtimes, server_time_ora).as_dict()})
    add_targets(row, pred_ora)
    rows.append(row)

    return rows



# ==============================================================================
# SECTION 6 -- FACTORIAL ABLATION ARMS (Sepset-aggregation x weighting),
# reused for BOTH the plain factorial ablation AND the corruption sweep
# (Reviewer points 1 and 3).
#   no_agg_equal   -- Sepset agg OFF, equal weight   (= 'naive' from above)
#   agg_equal      -- Sepset agg ON,  equal weight   (the missing control!)
#   agg_consensus  -- Sepset agg ON,  consensus weight
#   agg_oracle     -- Sepset agg ON,  oracle weight (ablation upper bound)
# ==============================================================================

ABLATION_ARMS = ["no_agg_equal", "agg_equal", "agg_consensus", "agg_oracle"]
STRUCTURAL_ZERO_METHODS = {"naive", "no_agg_equal"}  # Dir_F1 excluded from paired tests


def _corrupt_client_edges(edges, p, rng, drop_frac=0.6):
    all_possible = [tuple(sorted((i, j))) for i in range(p) for j in range(i + 1, p)]
    n_target = min(max(1, int(len(edges) * (1 + drop_frac))), len(all_possible))
    idx = rng.choice(len(all_possible), size=n_target, replace=False)
    return {all_possible[i] for i in idx}


def run_ablation_replicate(cfg: ExperimentConfig, scenario_id: str, replicate_id: int,
                            seed_manager: SeedManager, corruption_fraction: float = 0.0) -> list:
    rep_rng = seed_manager.replicate_rng(scenario_id, replicate_id)
    client_seeds = seed_manager.client_seeds(scenario_id, replicate_id, cfg.K)
    seed_record = seed_manager.record(scenario_id, replicate_id, cfg.K)

    p, N, K = cfg.p, cfg.N, cfg.K
    B = generate_random_dag(p, cfg.edge_prob, rep_rng)
    true_skel = true_skeleton_edges(B)
    true_dir = true_directed_edges(B)
    X_pool = simulate_linear_sem(B, N, rep_rng, noise_regime=cfg.noise_regime)
    client_data = federated_split(X_pool, K, rep_rng)
    client_edges, client_seps, client_runtimes = local_pc_clients_timed(client_data, cfg.alpha, cfg.ell)

    corrupted_ids = []
    if corruption_fraction > 0:
        n_corrupt = max(1, int(round(corruption_fraction * K)))
        for k in range(n_corrupt):
            crng = np.random.default_rng(client_seeds[k] + 999_983)
            client_edges[k] = _corrupt_client_edges(client_edges[k], p, crng)
            client_seps[k] = {}
            corrupted_ids.append(k)

    equal_weights = np.ones(K) / K

    def base_row(arm, sepset_agg, weighting):
        return {"experiment_family": "factorial_ablation" if corruption_fraction == 0 else "corrupted_client_sweep",
                "scenario_id": scenario_id, "replicate_id": replicate_id, "method": arm,
                "sepset_aggregation": sepset_agg, "weighting": weighting,
                "evaluation_unit": "global_graph", "evaluation_target": "true_dag",
                "corruption_fraction": corruption_fraction, "corrupted_client_ids": corrupted_ids,
                **cfg.as_dict(), **seed_record}

    def eval_arm(arm_name, sepset_agg, weighting, weights, use_sepsets):
        t0 = time.time()
        pred = weighted_aggregation_with_weights(client_edges, weights, cfg.tau)
        agg_seps = aggregate_sepsets(client_seps, weights, cfg.tau, cfg.sep_tau_factor) if use_sepsets else {}
        G = nx.Graph(); G.add_nodes_from(range(p)); G.add_edges_from(pred)
        cpdag = orient_v_structures(G, agg_seps)
        server_time = time.time() - t0
        shd, prec, rec, f1 = compute_metrics(true_skel, pred)
        n_or, dp, dr, df1 = orientation_metrics_cpdag(true_dir, cpdag)
        comm = communication_cost_fixed(client_edges, client_seps, p, cfg.ell)
        row = base_row(arm_name, sepset_agg, weighting)
        row.update({"SHD": shd, "F1": f1, "Precision": prec, "Recall": rec, "n_oriented": n_or,
                    "Dir_Prec": dp, "Dir_Rec": dr, "Dir_F1": df1, "communication_bytes": comm,
                    "client_weights": np.asarray(weights).tolist(), "n_sepset_pairs_retained": len(agg_seps),
                    **RuntimeRecord(client_runtimes, server_time).as_dict()})
        if not use_sepsets:
            row["note"] = ("Dir_F1=0 BY CONSTRUCTION -- no Sepsets aggregated/passed to "
                            "orientation; definitional, not empirical evidence about weighting.")
        return row

    rows = [
        eval_arm("no_agg_equal", "off", "equal", equal_weights, use_sepsets=False),
        eval_arm("agg_equal", "on", "equal", equal_weights, use_sepsets=True),
    ]
    w_con = compute_consensus_reliability(client_edges)
    rows.append(eval_arm("agg_consensus", "on", "consensus", w_con, use_sepsets=True))
    w_ora = compute_oracle_reliability(client_edges, true_skel)
    r = eval_arm("agg_oracle", "on", "oracle", w_ora, use_sepsets=True)
    r["note"] = "ablation only -- requires ground truth, not deployable"
    rows.append(r)
    return rows



# ==============================================================================
# SECTION 7 -- STATISTICS (Reviewer point 5, second half): Cohen's d for
# paired differences everywhere, consistently; BH correction within a
# family; structural-zero methods excluded from directional paired tests.
# ==============================================================================

def paired_cohens_d(diff):
    diff = np.asarray(diff, dtype=float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else np.nan


def ci95_mean(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    tcrit = t_dist.ppf(0.975, df=n - 1)
    return (m - tcrit * se, m + tcrit * se)


DIRECTIONAL_METRICS = {"Dir_F1", "Dir_Prec", "Dir_Rec", "n_oriented"}


def paired_test(a, b, metric, method_a, method_b):
    if metric in DIRECTIONAL_METRICS and (method_a in STRUCTURAL_ZERO_METHODS or method_b in STRUCTURAL_ZERO_METHODS):
        zero_method = method_a if method_a in STRUCTURAL_ZERO_METHODS else method_b
        return {"excluded": True, "reason": f"{metric} involves structural-zero method {zero_method}"}
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = len(diff)
    lo, hi = ci95_mean(diff)
    result = {"excluded": False, "n": n, "mean_diff": float(diff.mean()),
              "sd_diff": float(diff.std(ddof=1)) if n > 1 else np.nan,
              "ci95_lo": lo, "ci95_hi": hi, "cohens_d": paired_cohens_d(diff)}
    if n > 1 and diff.std(ddof=1) > 0:
        from scipy import stats as _stats
        _, pv = _stats.ttest_1samp(diff, 0.0)
        result["p_value"] = float(pv)
    else:
        result["p_value"] = np.nan
    return result


def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    valid = ~np.isnan(pvals)
    out = np.full_like(pvals, np.nan)
    if valid.sum() == 0:
        return out.tolist()
    p_valid = pvals[valid]
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    m = len(ranked)
    adjusted = ranked * m / (np.arange(m) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    result = np.empty(m)
    result[order] = adjusted
    out[valid] = result
    return out.tolist()


def build_condition_table(df, condition_col, method_a, method_b, metrics, key_cols=("scenario_id", "replicate_id")):
    a = df[df.method == method_a].set_index(list(key_cols))
    b = df[df.method == method_b].set_index(list(key_cols))
    common = a.index.intersection(b.index)
    # condition_col may itself be one of key_cols (e.g. "scenario_id"), in
    # which case it lives in the index rather than as a column.
    if condition_col in key_cols:
        cond_series = pd.Series(common.get_level_values(condition_col), index=common)
    else:
        cond_series = pd.Series(a.loc[common, condition_col].values, index=common)
    rows = []
    for cond, g_idx in cond_series.groupby(cond_series).groups.items():
        row = {condition_col: cond, "method_a": method_a, "method_b": method_b, "n_replicates": len(g_idx)}
        for m in metrics:
            res = paired_test(a.loc[g_idx, m].values, b.loc[g_idx, m].values, m, method_a, method_b)
            if res.get("excluded"):
                row[f"{m}_excluded_reason"] = res["reason"]
            else:
                row[f"{m}_mean_diff"] = res["mean_diff"]; row[f"{m}_ci95_lo"] = res["ci95_lo"]
                row[f"{m}_ci95_hi"] = res["ci95_hi"]; row[f"{m}_cohens_d"] = res["cohens_d"]
                row[f"{m}_p_value"] = res["p_value"]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(condition_col).reset_index(drop=True)
    for m in metrics:
        pcol = f"{m}_p_value"
        if pcol in out.columns:
            out[f"{m}_p_value_bh"] = benjamini_hochberg(out[pcol].tolist())
    return out


# ==============================================================================
# SECTION 8 -- SACHS REAL-DATA EXPERIMENT (Reviewer points 10, 11):
# observational-only and pooled-all-conditions are separate, explicitly
# named runs (never silently merged); a local-vs-federated orientation
# diagnostic is reported so a poor federated result is not automatically
# blamed on sample fragmentation.
# ==============================================================================

SACHS_GT_EDGES = [
    ("praf", "pmek"), ("pmek", "p44.42"), ("pmek", "pakts473"), ("PIP3", "PIP2"),
    ("PIP3", "pakts473"), ("plcg", "PIP2"), ("plcg", "PIP3"), ("PKA", "p44.42"),
    ("PKA", "pakts473"), ("PKA", "pjnk"), ("PKA", "P38"), ("PKA", "praf"),
    ("PKC", "praf"), ("PKC", "pmek"), ("PKC", "pjnk"), ("PKC", "P38"), ("PKC", "PKA"),
]
SACHS_CANONICAL = ["praf", "pmek", "p44.42", "pakts473", "PKA", "PKC", "P38", "pjnk", "plcg", "PIP2", "PIP3"]
SACHS_COL_MAP = {"raf": "praf", "mek": "pmek", "erk": "p44.42", "akt": "pakts473", "pka": "PKA",
                  "pkc": "PKC", "p38": "P38", "jnk": "pjnk", "plcg": "plcg", "pip2": "PIP2", "pip3": "PIP3"}
for _c in SACHS_CANONICAL:
    SACHS_COL_MAP[_c.lower()] = _c


def _discover_sachs_conditions(sachs_dir):
    if not os.path.isdir(sachs_dir):
        return {"files": [], "observational": [], "interventional": []}
    all_csvs = [f for f in os.listdir(sachs_dir) if f.endswith(".csv") and f.lower() != "groundtruth.csv"]
    obs, interv = [], []
    for f in sorted(all_csvs):
        (obs if "cd3cd28.csv" == f.lower() else interv).append(f)
    return {"files": sorted(all_csvs), "observational": obs, "interventional": interv}


def _load_sachs_files(sachs_dir, filenames):
    dfs = []
    for fn in filenames:
        fpath = os.path.join(sachs_dir, fn)
        for sep in [",", "\t", ";"]:
            try:
                df = pd.read_csv(fpath, sep=sep)
                if df.shape[1] > 1:
                    rename = {c: SACHS_COL_MAP[c.strip().lower()] for c in df.columns if c.strip().lower() in SACHS_COL_MAP}
                    dfs.append(df.rename(columns=rename))
                    break
            except Exception:
                continue
    if not dfs:
        return None, None
    combined = pd.concat(dfs, ignore_index=True)
    available = [c for c in SACHS_CANONICAL if c in combined.columns]
    if len(available) < 5:
        available = combined.select_dtypes(include=[np.number]).columns.tolist()
    X = combined[available].dropna().values.astype(float)
    return X, available


def _sachs_ground_truth(col_names):
    gt_skel, gt_dir = set(), set()
    for u, v in SACHS_GT_EDGES:
        if u in col_names and v in col_names:
            ui, vi = col_names.index(u), col_names.index(v)
            gt_skel.add(tuple(sorted((ui, vi)))); gt_dir.add((ui, vi))
    return gt_skel, gt_dir


def run_sachs_condition_set(sachs_dir, filenames, experiment_name, K, n_replicates,
                             cfg, seed_manager):
    X, col_names = _load_sachs_files(sachs_dir, filenames)
    if X is None or len(col_names) < 2:
        return None
    true_skel, true_dir = _sachs_ground_truth(col_names)
    p = X.shape[1]
    rows, diags = [], []
    for rep in range(n_replicates):
        rep_rng = seed_manager.replicate_rng(experiment_name, rep)
        client_data = federated_split(X, K, rep_rng)
        client_edges, client_seps, client_runtimes = local_pc_clients_timed(client_data, cfg.alpha, cfg.ell)

        def base_row(method, unit):
            return {"experiment_name": experiment_name, "scenario_id": experiment_name,
                     "replicate_id": rep, "method": method, "evaluation_unit": unit,
                     "evaluation_target": "sachs_ground_truth", "n_vars": p, "n_samples": X.shape[0],
                     "K": K, "tau": cfg.tau, "ell": cfg.ell}

        local_f1s, local_df1s = [], []
        t0 = time.time()
        for edges, seps in zip(client_edges, client_seps):
            Gcl = nx.Graph(); Gcl.add_nodes_from(range(p)); Gcl.add_edges_from(edges)
            cpdag_l = orient_v_structures(Gcl, seps)
            _, _, _, f = compute_metrics(true_skel, edges)
            _, _, _, df = orientation_metrics_cpdag(true_dir, cpdag_l)
            local_f1s.append(f); local_df1s.append(df)
        srv_local = time.time() - t0
        row = base_row("local", "mean_local_graph")
        row.update({"F1": float(np.mean(local_f1s)), "Dir_F1": float(np.mean(local_df1s)),
                    "communication_bytes": communication_cost_fixed(client_edges, client_seps, p, cfg.ell),
                    **RuntimeRecord(client_runtimes, srv_local).as_dict()})
        rows.append(row)

        t0 = time.time()
        pred_n = naive_majority_aggregation(client_edges, cfg.tau)
        Gn = nx.Graph(); Gn.add_nodes_from(range(p)); Gn.add_edges_from(pred_n)
        cpdag_n = orient_v_structures(Gn, {})
        srv_n = time.time() - t0
        shd_n, _, _, f1_n = compute_metrics(true_skel, pred_n)
        _, _, _, df1_n = orientation_metrics_cpdag(true_dir, cpdag_n)
        row = base_row("naive", "global_graph")
        row.update({"SHD": shd_n, "F1": f1_n, "Dir_F1": df1_n,
                    "communication_bytes": communication_cost_fixed(client_edges, client_seps, p, cfg.ell),
                    "note": "Dir_F1=0 by construction", **RuntimeRecord(client_runtimes, srv_n).as_dict()})
        rows.append(row)

        t0 = time.time()
        w_con = compute_consensus_reliability(client_edges)
        pred_c = weighted_aggregation_with_weights(client_edges, w_con, cfg.tau)
        agg_seps = aggregate_sepsets(client_seps, w_con, cfg.tau, cfg.sep_tau_factor)
        Gc = nx.Graph(); Gc.add_nodes_from(range(p)); Gc.add_edges_from(pred_c)
        cpdag_c = orient_v_structures(Gc, agg_seps)
        srv_c = time.time() - t0
        shd_c, _, _, f1_c = compute_metrics(true_skel, pred_c)
        _, _, _, df1_c = orientation_metrics_cpdag(true_dir, cpdag_c)
        row = base_row("fedpc_consensus", "global_graph")
        row.update({"SHD": shd_c, "F1": f1_c, "Dir_F1": df1_c, "n_sepset_pairs_retained": len(agg_seps),
                    "communication_bytes": communication_cost_fixed(client_edges, client_seps, p, cfg.ell),
                    **RuntimeRecord(client_runtimes, srv_c).as_dict()})
        rows.append(row)

        diags.append({
            "replicate_id": rep, "local_mean_dir_f1": float(np.mean(local_df1s)),
            "federated_consensus_dir_f1": df1_c, "federated_naive_dir_f1": df1_n,
            "n_sepset_pairs_retained": len(agg_seps),
            "note": ("If local_mean_dir_f1 > federated_consensus_dir_f1, the orientation "
                     "gap is NOT attributable to sample fragmentation alone -- the "
                     "cross-client Sepset-aggregation retention threshold is a candidate "
                     "mechanism and must be checked before attributing the gap to small "
                     "per-client sample size."),
        })
    return {"results": pd.DataFrame(rows), "diagnostics": pd.DataFrame(diags),
            "n_vars_used": p, "n_samples": X.shape[0], "var_names": col_names,
            "n_files_used": len(filenames), "filenames_used": filenames}



# ==============================================================================
# SECTION 9 -- MECHANISM-SHIFT SCENARIO REGISTRY (genuine client-level
# heterogeneity; Reviewer points 7, 8: swept, and evaluated against
# multiple candidate population-level targets)
# ==============================================================================

def build_mechshift_registry():
    rows = []
    for dp in [0.0, 0.03, 0.07, 0.15, 0.25]:
        cfg = replace(CANONICAL_DEFAULT, mechanism_shift=True, deletion_probability=dp)
        sid = make_scenario_id(cfg, f"mechshift_delprob{dp}")
        rows.append({"scenario_id": sid, "family": "mechanism_shift_sweep",
                     "config_id": config_id(cfg), **cfg.as_dict()})
    df = pd.DataFrame(rows)
    dup = df[df.duplicated(subset=["config_id"], keep=False)]
    if not dup.empty:
        raise ValueError(f"Duplicate config_id in mechshift registry:\n{dup}")
    return df


def build_corruption_registry():
    rows = []
    for cf in [0.0, 0.2, 0.4, 0.6]:
        cfg = CANONICAL_DEFAULT
        sid = f"corruption{cf}_{config_id(cfg)}"
        rows.append({"scenario_id": sid, "corruption_fraction": cf, "config_id": config_id(cfg), **cfg.as_dict()})
    return pd.DataFrame(rows)


# ==============================================================================
# SECTION 10 -- FIGURE HELPERS
# ==============================================================================

def _bar_group(ax, methods, means, stds, ylabel, title, ylim=None):
    colors = [PALETTE.get(m, "#95a5a6") for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    x = np.arange(len(methods))
    bars = ax.bar(x, means, color=colors, alpha=0.88, edgecolor="white", linewidth=1.1,
                  yerr=stds, capsize=4, error_kw={"elinewidth": 1.3, "ecolor": "#444"})
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5, rotation=25, ha="right")
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title, fontsize=10, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    for bar, val, std in zip(bars, means, stds):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (std if np.isfinite(std) else 0) + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")


def _line_group(ax, x_vals, data_dict, ylabel, title, xlabel=""):
    for m, vals in data_dict.items():
        means = [v[0] for v in vals]; stds = [v[1] for v in vals]
        ax.plot(x_vals, means, marker="o", color=PALETTE.get(m, "#333"), label=METHOD_LABELS.get(m, m),
                linewidth=1.8, markersize=5)
        ax.fill_between(x_vals, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                        color=PALETTE.get(m, "#333"), alpha=0.12)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold"); ax.legend(fontsize=7, framealpha=0.7)


def _get(df, scenario, method, metric):
    row = df[(df["scenario_id"] == scenario) & (df["method"] == method)]
    if row.empty or metric not in row.columns:
        return np.nan, np.nan
    vals = pd.to_numeric(row[metric], errors="coerce").dropna().values
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))



# ==============================================================================
# SECTION 11 -- ORCHESTRATION + FIGURES
# ==============================================================================

def run_main_sweep(reps, out_dir):
    registry = build_main_scenario_registry()
    sm = SeedManager(MASTER_SEED)
    all_rows = []
    print(f"[main sweep] {len(registry)} scenarios x {reps} reps ...")
    for _, scen in registry.iterrows():
        cfg = ExperimentConfig(**{k: scen[k] for k in CANONICAL_DEFAULT.as_dict().keys()})
        t0 = time.time()
        for rep in range(reps):
            all_rows.extend(run_replicate(cfg, scen["scenario_id"], rep, sm))
        print(f"  {scen['scenario_id']} ({scen['family']}) done in {time.time()-t0:.1f}s")
    df = pd.DataFrame(all_rows)
    dup = df[df.duplicated(subset=["scenario_id", "replicate_id", "method"], keep=False)]
    assert dup.empty, "duplicate rows in main sweep"
    df.to_csv(f"{out_dir}/raw/main_sweep_raw.csv", index=False)
    return df, registry


def run_main_sweep_partial(reps, out_dir, row_start, row_end):
    """Same logic as run_main_sweep, but only runs registry rows
    [row_start:row_end] and APPENDS to main_sweep_raw.csv instead of
    overwriting it -- lets the sweep be split across multiple runs
    while producing the exact same combined CSV as running it all at
    once (SeedManager is deterministic per scenario_id/replicate_id)."""
    registry = build_main_scenario_registry()
    subset = registry.iloc[row_start:row_end]
    sm = SeedManager(MASTER_SEED)
    all_rows = []
    print(f"[main sweep partial rows {row_start}:{row_end}] {len(subset)} scenarios x {reps} reps ...")
    for _, scen in subset.iterrows():
        cfg = ExperimentConfig(**{k: scen[k] for k in CANONICAL_DEFAULT.as_dict().keys()})
        t0 = time.time()
        for rep in range(reps):
            all_rows.extend(run_replicate(cfg, scen["scenario_id"], rep, sm))
        print(f"  {scen['scenario_id']} ({scen['family']}) done in {time.time()-t0:.1f}s")
    df = pd.DataFrame(all_rows)
    csv_path = f"{out_dir}/raw/main_sweep_raw.csv"
    if os.path.exists(csv_path):
        combined = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
    else:
        combined = df
    dup = combined[combined.duplicated(subset=["scenario_id", "replicate_id", "method"], keep=False)]
    assert dup.empty, "duplicate rows in main sweep"
    combined.to_csv(csv_path, index=False)
    return combined, registry


def run_mechshift_sweep_partial(reps, out_dir, row_start, row_end):
    """Same logic as run_mechshift_sweep, but only runs registry rows
    [row_start:row_end] and APPENDS to mechshift_sweep_raw.csv."""
    registry = build_mechshift_registry()
    subset = registry.iloc[row_start:row_end]
    sm = SeedManager(MASTER_SEED)
    all_rows = []
    print(f"[mechshift sweep partial rows {row_start}:{row_end}] {len(subset)} scenarios x {reps} reps ...")
    for _, scen in subset.iterrows():
        cfg = ExperimentConfig(**{k: scen[k] for k in CANONICAL_DEFAULT.as_dict().keys()})
        t0 = time.time()
        for rep in range(reps):
            all_rows.extend(run_replicate(cfg, scen["scenario_id"], rep, sm, mechshift_targets=True))
        print(f"  {scen['scenario_id']} (delprob={scen['deletion_probability']}) done in {time.time()-t0:.1f}s")
    df = pd.DataFrame(all_rows)
    csv_path = f"{out_dir}/raw/mechshift_sweep_raw.csv"
    if os.path.exists(csv_path):
        combined = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
    else:
        combined = df
    combined.to_csv(csv_path, index=False)
    return combined, registry


def run_mechshift_sweep(reps, out_dir):
    registry = build_mechshift_registry()
    sm = SeedManager(MASTER_SEED)
    all_rows = []
    print(f"[mechanism-shift sweep] {len(registry)} scenarios x {reps} reps ...")
    for _, scen in registry.iterrows():
        cfg = ExperimentConfig(**{k: scen[k] for k in CANONICAL_DEFAULT.as_dict().keys()})
        t0 = time.time()
        for rep in range(reps):
            all_rows.extend(run_replicate(cfg, scen["scenario_id"], rep, sm, mechshift_targets=True))
        print(f"  {scen['scenario_id']} (delprob={scen['deletion_probability']}) done in {time.time()-t0:.1f}s")
    df = pd.DataFrame(all_rows)
    df.to_csv(f"{out_dir}/raw/mechshift_sweep_raw.csv", index=False)
    return df, registry


def run_corruption_sweep(reps, out_dir):
    registry = build_corruption_registry()
    sm = SeedManager(MASTER_SEED)
    all_rows = []
    print(f"[corruption sweep] {len(registry)} scenarios x {reps} reps x 4 arms ...")
    for _, scen in registry.iterrows():
        cfg = CANONICAL_DEFAULT
        t0 = time.time()
        for rep in range(reps):
            all_rows.extend(run_ablation_replicate(cfg, scen["scenario_id"], rep, sm,
                                                     corruption_fraction=scen["corruption_fraction"]))
        print(f"  corruption_fraction={scen['corruption_fraction']} done in {time.time()-t0:.1f}s")
    df = pd.DataFrame(all_rows)
    df.to_csv(f"{out_dir}/raw/corruption_sweep_raw.csv", index=False)
    return df, registry


def run_factorial_ablation(reps, out_dir):
    sm = SeedManager(MASTER_SEED)
    scen_id = make_scenario_id(CANONICAL_DEFAULT, "factorial_default")
    print(f"[factorial ablation] canonical default x {reps} reps x 4 arms ...")
    all_rows = []
    t0 = time.time()
    for rep in range(reps):
        all_rows.extend(run_ablation_replicate(CANONICAL_DEFAULT, scen_id, rep, sm, corruption_fraction=0.0))
    print(f"  done in {time.time()-t0:.1f}s")
    df = pd.DataFrame(all_rows)
    df.to_csv(f"{out_dir}/raw/factorial_ablation_raw.csv", index=False)
    return df, scen_id


def run_scalability(reps, out_dir):
    sm = SeedManager(MASTER_SEED)
    rows = []
    print("[scalability] p in [20, 30, 50] ...")
    for p_val in [20, 30, 50]:
        cfg = replace(CANONICAL_DEFAULT, p=p_val)
        sid = make_scenario_id(cfg, f"scale_p{p_val}")
        t0 = time.time()
        for rep in range(reps):
            rows.extend(run_replicate(cfg, sid, rep, sm))
        print(f"  p={p_val} done in {time.time()-t0:.1f}s")
    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/raw/scalability_raw.csv", index=False)
    return df


def run_sachs(sachs_dir, reps, out_dir):
    if not os.path.isdir(sachs_dir):
        print(f"[Sachs] directory not found: {sachs_dir} -- skipping.")
        return None
    disc = _discover_sachs_conditions(sachs_dir)
    print(f"[Sachs] files={len(disc['files'])} observational={disc['observational']} "
          f"interventional_n={len(disc['interventional'])}")
    sm = SeedManager(MASTER_SEED)
    out = {}
    if disc["observational"]:
        obs = run_sachs_condition_set(sachs_dir, disc["observational"], "sachs_observational",
                                       K=5, n_replicates=reps, cfg=CANONICAL_DEFAULT, seed_manager=sm)
        if obs:
            obs["results"].to_csv(f"{out_dir}/raw/sachs_observational_raw.csv", index=False)
            obs["diagnostics"].to_csv(f"{out_dir}/raw/sachs_observational_diagnostics.csv", index=False)
            out["observational"] = obs
    if disc["files"]:
        pooled = run_sachs_condition_set(sachs_dir, disc["files"], "sachs_pooled_all_conditions",
                                          K=5, n_replicates=reps, cfg=CANONICAL_DEFAULT, seed_manager=sm)
        if pooled:
            pooled["results"].to_csv(f"{out_dir}/raw/sachs_pooled_raw.csv", index=False)
            pooled["diagnostics"].to_csv(f"{out_dir}/raw/sachs_pooled_diagnostics.csv", index=False)
            out["pooled"] = pooled
    return out



def generate_figures(df_main, df_mech, df_corr, df_fact, fact_scen_id, df_scale, sachs_out, out_dir):
    fig_dir = f"{out_dir}/figures"; os.makedirs(fig_dir, exist_ok=True)
    default_sid = make_scenario_id(CANONICAL_DEFAULT, "default")
    all_methods = ["centralized", "local", "naive", "fedpc_consensus", "fedpc_oracle"]

    # ---- FIG 1: overall structural + orientation recovery (baseline) ----
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Overall Structural & Orientation Recovery\n(baseline: p=20, N=5000, K=5, noise=mild, tau=0.5)",
                 fontsize=12, fontweight="bold")
    for ax, metric, ylabel, ylim in [
        (axes[0, 0], "SHD", "SHD (lower better)", None),
        (axes[0, 1], "Precision", "Skeleton Precision", (0, 1.15)),
        (axes[0, 2], "Recall", "Skeleton Recall", (0, 1.15)),
        (axes[0, 3], "F1", "Skeleton F1", (0, 1.15)),
        (axes[1, 0], "Dir_Prec", "Directional Precision", (0, 1.15)),
        (axes[1, 1], "Dir_Rec", "Directional Recall", (0, 1.15)),
        (axes[1, 2], "Dir_F1", "Directional F1", (0, 1.15)),
        (axes[1, 3], "n_oriented", "# Oriented edges", None),
    ]:
        means = [_get(df_main, default_sid, m, metric)[0] for m in all_methods]
        stds = [_get(df_main, default_sid, m, metric)[1] for m in all_methods]
        _bar_group(ax, all_methods, means, stds, ylabel, ylabel, ylim)
    plt.tight_layout()
    fig.savefig(f"{fig_dir}/fig1_overall_structural_orientation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1")

    # ---- FIG 2: sensitivity to K, N, phi, ell ----
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("Sensitivity: Federation Scale (K), Sample Size (N), Graph Density (phi), "
                 "Conditioning Depth (ell)", fontsize=12, fontweight="bold")
    sweeps = [
        ("K", [2, 5, 10], "federation_scale", "K (clients)"),
        ("N", [100, 3000, 5000], "sample_size", "N (total samples)"),
        ("edge_prob", [0.1, 0.2], "graph_density", "Edge probability phi"),
        ("ell", [1, 2], "conditioning_depth", "Max conditioning depth ell"),
    ]
    reg = build_main_scenario_registry()
    for col, (param, vals, family, xlabel) in enumerate(sweeps):
        scen_ids = []
        for v in vals:
            match = reg[(reg["family"] == family) & (reg[param] == v)]
            if match.empty:
                match = reg[reg["family"] == "default"]
            scen_ids.append(match.iloc[0]["scenario_id"])
        for row_i, (metric, ylabel) in enumerate([("SHD", "SHD"), ("F1", "Skeleton F1")]):
            data_d = {m: [_get(df_main, sid, m, metric) for sid in scen_ids] for m in all_methods}
            _line_group(axes[row_i, col], vals, data_d, ylabel, f"{ylabel} vs {xlabel}", xlabel=xlabel)
    plt.tight_layout()
    fig.savefig(f"{fig_dir}/fig2_sensitivity_K_N_phi_ell.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2")

    # ---- FIG 3: noise-scaling (renamed) vs genuine mechanism-shift heterogeneity ----
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle("Noise Scaling (sample-wise, NOT client heterogeneity) vs.\n"
                 "Mechanism-Shift (genuine client-level heterogeneity)", fontsize=12, fontweight="bold")
    noise_labels = ["none", "mild", "strong"]
    noise_scens = []
    for lbl in noise_labels:
        match = reg[(reg["family"] == "noise_scaling") & (reg["noise_regime"] == (None if lbl == "none" else lbl))]
        if match.empty:
            match = reg[reg["family"] == "default"]
        noise_scens.append(match.iloc[0]["scenario_id"])
    for ax, metric, ylabel in [(axes[0], "F1", "Skeleton F1"), (axes[1], "Dir_F1", "Directional F1")]:
        x = np.arange(len(noise_labels))
        for m in all_methods:
            means = [_get(df_main, sid, m, metric)[0] for sid in noise_scens]
            stds = [_get(df_main, sid, m, metric)[1] for sid in noise_scens]
            ax.plot(x, means, marker="o", color=PALETTE.get(m), label=METHOD_LABELS.get(m), linewidth=1.6)
        ax.set_xticks(x); ax.set_xticklabels(noise_labels, fontsize=8)
        ax.set_title(f"{ylabel} vs noise scaling", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.15); ax.legend(fontsize=6.5)

    mech_scens = df_mech.drop_duplicates("scenario_id")[["scenario_id", "deletion_probability"]] \
        .sort_values("deletion_probability")
    dps = mech_scens["deletion_probability"].tolist(); sids = mech_scens["scenario_id"].tolist()
    ax = axes[2]
    for m in ["naive", "fedpc_consensus", "fedpc_oracle"]:
        means = [_get(df_mech, sid, m, "F1")[0] for sid in sids]
        stds = [_get(df_mech, sid, m, "F1")[1] for sid in sids]
        ax.plot(dps, means, marker="o", color=PALETTE.get(m), label=METHOD_LABELS.get(m), linewidth=1.8)
    ax.set_xlabel("Deletion probability (client DAG perturbation)", fontsize=8)
    ax.set_title("Mechanism-shift: F1 vs perturbation strength", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.15); ax.legend(fontsize=7)

    ax = axes[3]
    target_cols = ["F1_target_base_dag", "F1_target_union", "F1_target_intersection", "F1_target_prevalence50"]
    target_labels = ["base DAG", "union", "intersection", "prevalence-50%"]
    for tcol, tlab in zip(target_cols, target_labels):
        means = [_get(df_mech, sid, "fedpc_consensus", tcol)[0] for sid in sids]
        ax.plot(dps, means, marker="o", linewidth=1.6, label=tlab)
    ax.set_xlabel("Deletion probability", fontsize=8)
    ax.set_title("FedPC-Consensus F1 vs candidate\npopulation-level targets", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.15); ax.legend(fontsize=6.5)
    plt.tight_layout()
    fig.savefig(f"{fig_dir}/fig3_noise_scaling_vs_mechanism_shift.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3")

    # ---- FIG 4: corruption sweep, proper control arm (agg_equal vs agg_consensus) ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Corrupted-Client Sweep: Sepset-aggregation ON for all arms shown --\n"
                 "isolates the WEIGHTING effect (equal vs consensus vs oracle)", fontsize=11, fontweight="bold")
    corr_fracs = sorted(df_corr["corruption_fraction"].unique())
    corr_sids = [df_corr[df_corr["corruption_fraction"] == cf]["scenario_id"].iloc[0] for cf in corr_fracs]
    arms_shown = ["no_agg_equal", "agg_equal", "agg_consensus", "agg_oracle"]
    for ax, metric, ylabel in [(axes[0], "SHD", "SHD"), (axes[1], "F1", "Skeleton F1"), (axes[2], "Dir_F1", "Directional F1")]:
        data_d = {m: [_get(df_corr, sid, m, metric) for sid in corr_sids] for m in arms_shown}
        _line_group(ax, corr_fracs, data_d, ylabel, f"{ylabel} vs corruption fraction", xlabel="Corruption fraction")
    plt.tight_layout()
    fig.savefig(f"{fig_dir}/fig4_corruption_control_arm.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4")

    # ---- FIG 5: factorial ablation (isolates Sepset-agg effect from weighting effect) ----
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle("Factorial Ablation: Sepset-aggregation vs Weighting Scheme\n"
                 "(canonical default config)", fontsize=11, fontweight="bold")
    for ax, metric, ylabel, ylim in [
        (axes[0], "SHD", "SHD", None), (axes[1], "F1", "Skeleton F1", (0, 1.15)),
        (axes[2], "Dir_F1", "Directional F1", (0, 1.15)), (axes[3], "n_oriented", "# Oriented edges", None),
    ]:
        means = [_get(df_fact, fact_scen_id, m, metric)[0] for m in ABLATION_ARMS]
        stds = [_get(df_fact, fact_scen_id, m, metric)[1] for m in ABLATION_ARMS]
        _bar_group(ax, ABLATION_ARMS, means, stds, ylabel, ylabel, ylim)
    plt.tight_layout()
    fig.savefig(f"{fig_dir}/fig5_factorial_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig5")

    # ---- FIG 6: runtime (dual reporting) + corrected communication cost ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    fig.suptitle("Runtime (sequential-sum vs parallel-hardware estimate) & "
                 "Communication Cost (corrected formula)", fontsize=11, fontweight="bold")
    ax = axes[0]
    x = np.arange(len(all_methods)); w = 0.35
    sum_means = [_get(df_main, default_sid, m, "runtime_sum")[0] for m in all_methods]
    par_means = [_get(df_main, default_sid, m, "runtime_parallel_est")[0] for m in all_methods]
    ax.bar(x - w / 2, sum_means, w, label="runtime_sum (sequential)", color="#7f8c8d")
    ax.bar(x + w / 2, par_means, w, label="runtime_parallel_est (max client + server)", color="#2c3e50")
    ax.set_xticks(x); ax.set_xticklabels([METHOD_LABELS.get(m) for m in all_methods], fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("Runtime (s)", fontsize=9); ax.set_title("Runtime: both reportings", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)

    ax = axes[1]
    K_scens = reg[reg["family"].isin(["federation_scale", "default"])].sort_values("K")
    for m in ["naive", "fedpc_consensus", "fedpc_oracle"]:
        vals = [_get(df_main, sid, m, "communication_bytes")[0] for sid in K_scens["scenario_id"]]
        ax.plot(K_scens["K"], vals, marker="o", color=PALETTE.get(m), label=METHOD_LABELS.get(m))
    ax.set_xlabel("K (clients)", fontsize=9); ax.set_ylabel("Communication cost (bytes)", fontsize=9)
    ax.set_title("Comm. cost vs K", fontsize=10, fontweight="bold"); ax.legend(fontsize=7)

    ax = axes[2]
    phi_scens = reg[reg["family"].isin(["graph_density", "default"])].sort_values("edge_prob")
    for m in ["naive", "fedpc_consensus", "fedpc_oracle"]:
        vals = [_get(df_main, sid, m, "communication_bytes")[0] for sid in phi_scens["scenario_id"]]
        ax.plot(phi_scens["edge_prob"], vals, marker="o", color=PALETTE.get(m), label=METHOD_LABELS.get(m))
    ax.set_xlabel("Edge probability phi (graph density)", fontsize=9)
    ax.set_title("Comm. cost vs density\n(corrected: denser -> lower cost)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(f"{fig_dir}/fig6_runtime_and_communication.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig6")

    # ---- FIG 7: scalability ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Scalability: p in {20, 30, 50}", fontsize=11, fontweight="bold")
    p_vals = [20, 30, 50]
    p_sids = [make_scenario_id(replace(CANONICAL_DEFAULT, p=pv), f"scale_p{pv}") for pv in p_vals]
    for ax, metric, ylabel in [(axes[0], "F1", "Skeleton F1"), (axes[1], "SHD", "SHD"), (axes[2], "runtime_parallel_est", "Runtime (parallel est., s)")]:
        data_d = {m: [_get(df_scale, sid, m, metric) for sid in p_sids] for m in all_methods}
        _line_group(ax, p_vals, data_d, ylabel, f"{ylabel} vs p", xlabel="p (variables)")
    plt.tight_layout()
    fig.savefig(f"{fig_dir}/fig7_scalability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig7")

    # ---- FIG 8: Sachs observational vs pooled ----
    if sachs_out:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        fig.suptitle("Sachs Real Data: Observational-only vs Pooled-all-conditions\n"
                     "(these are two DISTINCT, never-merged analyses)", fontsize=11, fontweight="bold")
        methods_s = ["local", "naive", "fedpc_consensus"]
        for ax, metric, ylabel in [(axes[0], "SHD", "SHD"), (axes[1], "F1", "Skeleton F1"), (axes[2], "Dir_F1", "Directional F1")]:
            x = np.arange(len(methods_s)); w = 0.35
            obs_means = [_get(sachs_out["observational"]["results"], "sachs_observational", m, metric)[0]
                         if "observational" in sachs_out else np.nan for m in methods_s]
            pool_means = [_get(sachs_out["pooled"]["results"], "sachs_pooled_all_conditions", m, metric)[0]
                          if "pooled" in sachs_out else np.nan for m in methods_s]
            ax.bar(x - w / 2, obs_means, w, label="Observational only", color="#3498db")
            ax.bar(x + w / 2, pool_means, w, label="Pooled all conditions", color="#e67e22")
            ax.set_xticks(x); ax.set_xticklabels([METHOD_LABELS.get(m) for m in methods_s], fontsize=7.5, rotation=15)
            ax.set_ylabel(ylabel, fontsize=9); ax.set_title(ylabel, fontsize=10, fontweight="bold")
            ax.legend(fontsize=7)
        plt.tight_layout()
        fig.savefig(f"{fig_dir}/fig8_sachs_observational_vs_pooled.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved fig8")
