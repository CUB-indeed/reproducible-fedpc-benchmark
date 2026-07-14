# ==========================
# overall structural recovery
# ==========================


import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Data
# ==========================

methods = [
    "Centralized",
    "Local PC",
    "FedPC-Naive",
    "FedPC-Consensus",
    "FedPC-Oracle"
]

colors = [
    "#4C5B6B",
    "#A8B2B3",
    "#F28E2B",
    "#3CB371",
    "#4C9BD4"
]

# Mean
shd_mean = np.array([25.05, 24.16, 23.15, 23.15, 23.15])
prec_mean = np.array([0.68, 0.73, 0.76, 0.76, 0.76])
rec_mean  = np.array([0.71, 0.61, 0.61, 0.61, 0.61])
f1_mean   = np.array([0.69, 0.66, 0.67, 0.67, 0.67])

# Standard deviation
shd_std = np.array([10.4, 8.0, 7.6, 7.6, 7.6])
prec_std = np.array([0.10, 0.08, 0.09, 0.09, 0.09])
rec_std  = np.array([0.11, 0.10, 0.09, 0.09, 0.09])
f1_std   = np.array([0.10, 0.08, 0.08, 0.08, 0.08])

metrics = [
    ("Structural Hamming Distance (↓)", shd_mean, shd_std),
    ("Skeleton Precision (↑)", prec_mean, prec_std),
    ("Skeleton Recall (↑)", rec_mean, rec_std),
    ("Skeleton F1 (↑)", f1_mean, f1_std)
]

# ==========================
# Plot
# ==========================

plt.style.use("seaborn-v0_8-whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
axes = axes.flatten()

for ax, (title, mean, std) in zip(axes, metrics):

    y = np.arange(len(methods))

    # Horizontal confidence interval
    for i in range(len(methods)):
        ax.hlines(
            y=i,
            xmin=mean[i]-std[i],
            xmax=mean[i]+std[i],
            color="lightgray",
            linewidth=4,
            zorder=1
        )

    # Error bar
    ax.errorbar(
        mean,
        y,
        xerr=std,
        fmt="none",
        ecolor="black",
        elinewidth=1.5,
        capsize=4,
        zorder=2
    )

    # Mean point
    ax.scatter(
        mean,
        y,
        s=120,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        zorder=3
    )

    # Value labels (slightly below the point)
    for i, v in enumerate(mean):
        ax.text(
            v,
            i + 0.38,
            f"{v:.2f}",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=10)

    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        pad=12
    )

    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.grid(axis="y", visible=False)

    # Clean look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if "Distance" in title:
        ax.set_xlim(10, 38)
    else:
        ax.set_xlim(0.45, 0.90)

    # Move x tick labels downward
    ax.tick_params(axis='x', pad=12)

    # Make x tick labels multiline
    ticks = ax.get_xticks()
    labels = [f"{t:.2f}".replace(".", ".\n") for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=9)

    ax.tick_params(axis='y', pad=8)

    ax.invert_yaxis()

# Overall title
plt.suptitle(
    "Overall Structural Recovery\n"
    "(baseline: p=20, N=5000, K=5, hetero=mild, τ=0.5)",
    fontsize=16,
    fontweight="bold",
    y=0.98
)

# Better spacing
plt.tight_layout(rect=[0, 0.04, 1, 0.95])

plt.savefig(
    "Fig5_2_overall_structural_recovery.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================
# Orientation Recovery
# ==========================

import matplotlib.pyplot as plt
import numpy as np

# ==========================
# Data
# ==========================
methods = [
    "Centralized",
    "Local PC",
    "FedPC-Naive",
    "FedPC-Consensus",
    "FedPC-Oracle"
]

colors = [
    "#4C5B6B",   # dark blue-gray
    "#A9B3B5",   # gray
    "#D8D8D8",   # light gray
    "#42B36D",   # green
    "#4C9ED9"    # blue
]

# Mean values
precision = [0.68, 0.00, 0.00, 0.50, 0.50]
recall    = [0.30, 0.00, 0.00, 0.26, 0.26]
f1        = [0.40, 0.35, 0.00, 0.33, 0.33]
edges     = [15.65, 0.00, 0.00, 19.10, 19.10]

# Standard deviation
precision_err = [0.12, 0.00, 0.00, 0.12, 0.12]
recall_err    = [0.12, 0.00, 0.00, 0.10, 0.10]
f1_err        = [0.12, 0.10, 0.00, 0.10, 0.10]
edges_err     = [4.00, 0.00, 0.00, 5.80, 5.80]

metrics = [
    (precision, precision_err, "Directional\nPrecision", "Dir-Precision (↑)", (0, 1.25)),
    (recall, recall_err, "Directional\nRecall", "Dir-Recall (↑)", (0, 1.25)),
    (f1, f1_err, "Directional F1", "Dir-F1 (↑)", (0, 1.25)),
    (edges, edges_err, "Oriented\nEdges", "Count", (0, 26))
]

# ==========================
# Plot
# ==========================
fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10),
    constrained_layout=True
)

axes = axes.flatten()

for ax, (vals, errs, title, ylabel, ylim) in zip(axes, metrics):

    x = np.arange(len(methods))

    bars = ax.bar(
        x,
        vals,
        yerr=errs,
        capsize=5,
        width=0.75,
        color=colors,
        edgecolor="white",
        linewidth=1.2
    )

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12)

    ax.set_ylim(*ylim)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=11)

    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -------------------------
    # Value labels ABOVE error bars
    # -------------------------
    offset = ylim[1] * 0.025

    for bar, val, err in zip(bars, vals, errs):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            val + err + offset,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

# ==========================
# Figure Title
# ==========================
fig.suptitle(
    "§5.3 Orientation Recovery (CPDAG directed edges)\n"
    "(baseline: p=20, N=5000, K=5, hetero=mild, τ=0.5)",
    fontsize=18,
    fontweight="bold"
)

# ==========================
# Save Figure
# ==========================
save_path = "fig5_3_orientation_recovery.png"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

print(f"Figure saved to: {save_path}")

import matplotlib.pyplot as plt

# =========================
# Data 
# =========================

centralized_runtime = 19.4
centralized_f1 = 0.729
centralized_runtime_err = 10.1
centralized_f1_err = 0.092

local_runtime = 5.2
local_f1 = 0.694
local_runtime_err = 1.7
local_f1_err = 0.091

# Ketiga FedPC berada pada titik yang sama
fedpc_runtime = 0.0
fedpc_f1 = 0.704
fedpc_runtime_err = 0.0
fedpc_f1_err = 0.099

# =========================
# Colors
# =========================

colors = {
    "Centralized": "#2C3E50",
    "Local PC": "#7F8C8D",
    "FedPC-Naive": "#E67E22",
    "FedPC-Consensus": "#27AE60",
    "FedPC-Oracle": "#3498DB",
}

# =========================
# Figure
# =========================

fig, ax = plt.subplots(figsize=(11,6), dpi=180)

# =========================
# Plot Centralized
# =========================

ax.errorbar(
    centralized_runtime,
    centralized_f1,
    xerr=centralized_runtime_err,
    yerr=centralized_f1_err,
    fmt='o',
    color=colors["Centralized"],
    markersize=13,
    elinewidth=2,
    capsize=6,
    label="Centralized",
)

# =========================
# Plot Local PC
# =========================

ax.errorbar(
    local_runtime,
    local_f1,
    xerr=local_runtime_err,
    yerr=local_f1_err,
    fmt='o',
    color=colors["Local PC"],
    markersize=13,
    elinewidth=2,
    capsize=6,
    label="Local PC",
)

# =========================
# Plot FedPC (shared point)
# =========================

methods = [
    "FedPC-Naive",
    "FedPC-Consensus",
    "FedPC-Oracle",
]

for method in methods:
    ax.errorbar(
        fedpc_runtime,
        fedpc_f1,
        xerr=fedpc_runtime_err,
        yerr=fedpc_f1_err,
        fmt='o',
        color=colors[method],
        markersize=13,
        elinewidth=2,
        capsize=6,
        label=method,
    )

# =========================
# Labels
# =========================

# Centralized
ax.annotate(
    "Centralized",
    (centralized_runtime, centralized_f1),
    xytext=(8,6),
    textcoords="offset points",
    fontsize=11,
    color=colors["Centralized"],
)

# Local PC
ax.annotate(
    "Local PC",
    (local_runtime, local_f1),
    xytext=(8,6),
    textcoords="offset points",
    fontsize=11,
    color=colors["Local PC"],
)

# ---------- FedPC labels (NO OVERLAP) ----------

ax.annotate(
    "FedPC-Oracle",
    (fedpc_runtime, fedpc_f1),
    xytext=(8,10),
    textcoords="offset points",
    fontsize=11,
    color=colors["FedPC-Oracle"],
)

ax.annotate(
    "FedPC-Consensus",
    (fedpc_runtime, fedpc_f1),
    xytext=(45,4),
    textcoords="offset points",
    fontsize=11,
    color=colors["FedPC-Consensus"],
)

ax.annotate(
    "FedPC-Naive",
    (fedpc_runtime, fedpc_f1),
    xytext=(95,-2),
    textcoords="offset points",
    fontsize=11,
    color=colors["FedPC-Naive"],
)

# =========================
# Formatting
# =========================

ax.set_xlim(-1.5,31)
ax.set_ylim(0.59,0.83)

ax.set_xlabel("Runtime (s)", fontsize=16)
ax.set_ylabel("Skeleton F1 (↑)", fontsize=16)

ax.set_title(
    "§5.7 Accuracy-Efficiency Trade-off\n"
    "(F1 vs Runtime, baseline scenario)",
    fontsize=18,
    fontweight="bold",
    pad=18,
)

ax.text(
    0.5,
    1.02,
    "Upper-right = better (high F1, low runtime)",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=14,
)

ax.grid(True, linestyle="--", alpha=0.35)

ax.tick_params(labelsize=14)

ax.legend(
    loc="upper right",
    fontsize=12,
    frameon=True,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

# ==========================
# Accuracy efficiency scatter
# ==========================

import matplotlib.pyplot as plt

# =========================
# Data
# =========================

centralized_runtime = 19.4
centralized_f1 = 0.729
centralized_runtime_err = 10.1
centralized_f1_err = 0.092

local_runtime = 5.2
local_f1 = 0.694
local_runtime_err = 1.7
local_f1_err = 0.091

fedpc_runtime = 0.0
fedpc_f1 = 0.704
fedpc_runtime_err = 0.0
fedpc_f1_err = 0.099

# =========================
# Colors
# =========================

colors = {
    "Centralized": "#2C3E50",
    "Local PC": "#7F8C8D",
    "FedPC-Naive": "#E67E22",
    "FedPC-Consensus": "#27AE60",
    "FedPC-Oracle": "#3498DB",
}

# =========================
# Figure
# =========================

fig, ax = plt.subplots(figsize=(11, 6), dpi=180)

# Centralized
ax.errorbar(
    centralized_runtime,
    centralized_f1,
    xerr=centralized_runtime_err,
    yerr=centralized_f1_err,
    fmt='o',
    color=colors["Centralized"],
    markersize=13,
    elinewidth=2,
    capsize=6,
    label="Centralized",
)

# Local PC
ax.errorbar(
    local_runtime,
    local_f1,
    xerr=local_runtime_err,
    yerr=local_f1_err,
    fmt='o',
    color=colors["Local PC"],
    markersize=13,
    elinewidth=2,
    capsize=6,
    label="Local PC",
)

# FedPC
for method in ["FedPC-Naive", "FedPC-Consensus", "FedPC-Oracle"]:
    ax.errorbar(
        fedpc_runtime,
        fedpc_f1,
        xerr=fedpc_runtime_err,
        yerr=fedpc_f1_err,
        fmt='o',
        color=colors[method],
        markersize=13,
        elinewidth=2,
        capsize=6,
        label=method,
    )

# =========================
# Labels FIXED POSITION
# =========================

ax.annotate(
    "Centralized",
    (centralized_runtime, centralized_f1),
    xytext=(10, 6),
    textcoords="offset points",
    fontsize=10,
    color=colors["Centralized"],
)

ax.annotate(
    "Local PC",
    (local_runtime, local_f1),
    xytext=(8, 6),
    textcoords="offset points",
    fontsize=10,
    color=colors["Local PC"],
)

# FedPC (stack + adjusted)

ax.annotate(
    "FedPC-Oracle",
    (fedpc_runtime, fedpc_f1),
    xytext=(10, 18),
    textcoords="offset points",
    fontsize=10,
    fontweight="bold",
    color=colors["FedPC-Oracle"],
)

ax.annotate(
    "FedPC-Consensus",
    (fedpc_runtime, fedpc_f1),
    xytext=(10, 0),
    textcoords="offset points",
    fontsize=10,
    fontweight="bold",
    color=colors["FedPC-Consensus"],
)

ax.annotate(
    "FedPC-Naive",
    (fedpc_runtime, fedpc_f1),
    xytext=(10, -28),   # 🔥 lebih ke bawah biar tidak nabrak titik
    textcoords="offset points",
    fontsize=10,
    fontweight="bold",
    color=colors["FedPC-Naive"],
)

# =========================
# Formatting
# =========================

ax.set_xlim(-1.5, 31)
ax.set_ylim(0.59, 0.83)

ax.set_xlabel("Runtime (s)", fontsize=15)
ax.set_ylabel("Skeleton F1 (↑)", fontsize=15)

ax.set_title(
    "§5.7 Accuracy-Efficiency Trade-off\n(F1 vs Runtime, baseline scenario)",
    fontsize=17,
    fontweight="bold",
    pad=18,
)

# =========================
# FIX: subtitle lebih ke kiri + box
# =========================

ax.text(
    0.43,   # 🔥 digeser ke kiri (sebelumnya 0.5)
    0.98,
    "Upper-right = better (high F1, low runtime)",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=11,
    bbox=dict(
        facecolor="#F5F5F5",
        edgecolor="gray",
        boxstyle="round,pad=0.3",
        alpha=0.95,
    ),
)

ax.grid(True, linestyle="--", alpha=0.35)

ax.tick_params(axis="both", labelsize=13)

ax.legend(
    loc="upper right",
    fontsize=11,
    frameon=True,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# =========================
# Save
# =========================

plt.tight_layout()

plt.savefig(
    "fig5_7b_accuracy_efficiency_scatter.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()
plt.close(fig)