#!/usr/bin/env python3
"""
Failure Diversity Analysis
Compares how diverse the failing test cases are across algorithms.

Metric: Average Pairwise Distance (APD) of failing test cases in normalized parameter space.

  APD = (2 / n*(n-1)) * sum_{i<j} ||normalize(x_i) - normalize(x_j)||_2

  Higher APD = failures are more spread across the parameter space = more diverse scenarios.

References:
  - Chen, Bryce & Memon (2010), "Adaptive Random Testing: The ART of Test Case Diversity",
    J. Systems and Software 83(1):60-69
  - Zohdinasab et al. (2021), "DeepJanus: Testing the Operational Design Domain of
    Autonomous Systems", ISSTA -- uses pairwise diversity for ADS test suites

Plots generated:
  1. failure_diversity_boxplot.png   -- APD distribution per algorithm (box + points)
  2. failure_pca_scatter.png         -- 2D PCA projection showing WHERE failures cluster
  3. failure_distance_heatmap.png    -- Pairwise distance matrix for a representative run
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ============================================================
# Configuration: algorithms to compare
# Each entry: (display_name, glob_pattern_or_dir, color)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "plots", "failure_diversity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALGORITHMS = [
    (
        "PFES Baseline",
        os.path.join(BASE_DIR, "results_25runs_pfes", "run_*"),
        "#2196F3",
    ),
    (
        "PFES+SAMOTA",
        os.path.join(BASE_DIR, "results_25runs_samota_seeded", "run_*"),
        "#FF9800",
    ),
]

FALLBACK_ALGORITHMS = [
    (
        "PFES Baseline",
        os.path.join(BASE_DIR, "results_10runs_pfes", "run_*"),
        "#2196F3",
    ),
    (
        "PFES+SAMOTA",
        os.path.join(BASE_DIR, "results_25runs_samota", "run_*"),
        "#FF9800",
    ),
]


# ============================================================
# Data loading helpers
# ============================================================

def load_run(run_dir):
    """Load X (parameters) and R (violation matrix, 1=violated) from a run directory.
    Prefers Reqs_all_evaluations; falls back to F_all_evaluations (score < 0).
    Returns (X_array, R_array) or (None, None) if files missing."""
    x_path = os.path.join(run_dir, "X_all_evaluations_NSGA3_0.csv")
    if not os.path.exists(x_path):
        return None, None
    X = pd.read_csv(x_path).values.astype(float)

    reqs_path = os.path.join(run_dir, "Reqs_all_evaluations_NSGA3_0.csv")
    if os.path.exists(reqs_path):
        df = pd.read_csv(reqs_path, header=None)
        df = df.replace({"True": 1, "False": 0, True: 1, False: 0})
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        R = (~df.values.astype(bool)).astype(int)
        if X.shape[0] != R.shape[0]:
            X = X[:R.shape[0]]
        return X, R

    f_path = os.path.join(run_dir, "F_all_evaluations_NSGA3_0.csv")
    if os.path.exists(f_path):
        df = pd.read_csv(f_path, header=None)
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        F = df.values
        n_req = 3  # ADAS1 has 3 requirements
        if F.shape[1] == n_req:
            R = (F < 0).astype(int)
        else:
            # SAMOTA 5-col format: col0=R0, col1=R0(dup), col2/3=unused, col4=R1&R2
            R = np.zeros((F.shape[0], n_req), dtype=int)
            R[:, 0] = (F[:, 0] < 0).astype(int)
            R[:, 1] = (F[:, -1] < 0).astype(int)
            R[:, 2] = (F[:, -1] < 0).astype(int)
        if X.shape[0] != R.shape[0]:
            X = X[:R.shape[0]]
        return X, R

    return None, None


def load_algorithm(pattern_or_dir):
    """Return list of (X, R) for all runs matching the pattern."""
    matches = sorted(glob.glob(pattern_or_dir))
    if not matches:
        if os.path.isdir(pattern_or_dir):
            matches = [pattern_or_dir]
    runs = []
    for d in matches:
        X, R = load_run(d)
        if X is not None:
            runs.append((X, R))
    return runs


def get_failures(X, R):
    """Rows of X where at least one requirement is violated (R > 0)."""
    return X[np.any(R > 0, axis=1)]


# ============================================================
# Diversity metric
# ============================================================

def average_pairwise_distance(X_norm):
    """
    APD = mean Euclidean distance over all pairs of normalized failure vectors.
    Returns 0.0 when fewer than 2 failures exist.
    """
    n = X_norm.shape[0]
    if n < 2:
        return 0.0
    dists = [np.linalg.norm(X_norm[i] - X_norm[j]) for i, j in combinations(range(n), 2)]
    return float(np.mean(dists))


# ============================================================
# Load all data
# ============================================================
print("Loading experiment data...")
algo_runs = {}      # name -> list of (X, R)
algo_color = {}     # name -> hex color
all_X_pool = []     # for fitting the global scaler

for algo_list in [ALGORITHMS, FALLBACK_ALGORITHMS]:
    for name, pattern, color in algo_list:
        if name in algo_runs:
            continue
        runs = load_algorithm(pattern)
        if runs:
            algo_runs[name] = runs
            algo_color[name] = color
            all_X_pool.extend(X for X, _ in runs)
            print(f"  {name}: {len(runs)} run(s)")

for name, _, color in ALGORITHMS:
    if name not in algo_runs:
        algo_color[name] = color
        print(f"  {name}: not found, skipping")

if not algo_runs:
    print("No data found. Check directory paths in ALGORITHMS config.")
    raise SystemExit(1)

# Fit MinMaxScaler on ALL evaluations (violations + non-violations) for proper range
scaler = MinMaxScaler()
scaler.fit(np.vstack(all_X_pool))


# ============================================================
# Compute APD per run
# ============================================================
print("\nComputing APD per run...")
apd_per_algo = {}        # name -> list of float
failures_per_algo = {}   # name -> (X_norm_all_failures, run_index_labels)

for name, runs in algo_runs.items():
    apds = []
    all_fail_norm = []
    run_labels = []

    for run_idx, (X, R) in enumerate(runs):
        X_fail = get_failures(X, R)
        n_fail = X_fail.shape[0]

        if n_fail >= 2:
            X_fail_norm = scaler.transform(X_fail)
            apd = average_pairwise_distance(X_fail_norm)
            all_fail_norm.append(X_fail_norm)
            run_labels.extend([run_idx] * n_fail)
        else:
            apd = 0.0

        apds.append(apd)
        print(f"  {name} | run {run_idx+1:>2}: {n_fail:>3} failures, APD = {apd:.4f}")  # noqa

    apd_per_algo[name] = apds
    if all_fail_norm:
        failures_per_algo[name] = (np.vstack(all_fail_norm), np.array(run_labels))


# ============================================================
# Print summary table
# ============================================================
print("\n" + "=" * 65)
print(f"{'Algorithm':<25} {'Runs':>5} {'Mean APD':>10} {'Std APD':>10} {'Avg Fails':>10}")
print("-" * 65)
for name in algo_runs:
    apds = apd_per_algo[name]
    runs = algo_runs[name]
    avg_fails = np.mean([get_failures(X, R).shape[0] for X, R in runs])
    print(f"{name:<25} {len(runs):>5} {np.mean(apds):>10.4f} {np.std(apds):>10.4f} {avg_fails:>10.1f}")
print("=" * 65)


# ============================================================
# Plot 1: APD box plot + individual points
# ============================================================
names_present = [n for n, _, _ in ALGORITHMS if n in apd_per_algo]
colors_present = [algo_color[n] for n in names_present]

fig, ax = plt.subplots(figsize=(max(6, len(names_present) * 2), 5))
data_to_plot = [apd_per_algo[n] for n in names_present]

bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.45, zorder=2)
for patch, color in zip(bp["boxes"], colors_present):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
for element in ("whiskers", "caps", "medians", "fliers"):
    for item, color in zip(bp[element], np.repeat(colors_present, 2 if element in ("whiskers", "caps") else 1)):
        item.set_color(color)

for i, (name, color) in enumerate(zip(names_present, colors_present)):
    y = apd_per_algo[name]
    x = np.random.normal(i + 1, 0.05, size=len(y))
    ax.scatter(x, y, color=color, s=50, zorder=5, edgecolors="white", linewidths=0.5)

ax.set_xticks(range(1, len(names_present) + 1))
ax.set_xticklabels(names_present, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Average Pairwise Distance (APD)", fontsize=12)
ax.set_title(
    "Failure Diversity per Algorithm\n"
    "APD of failing test cases in normalized parameter space\n"
    "(higher = failures cover more diverse scenarios)",
    fontsize=11,
)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "failure_diversity_boxplot.png")
plt.savefig(out1, dpi=150)
plt.close()
print(f"\nSaved: {out1}")


# ============================================================
# Plot 2: PCA scatter of all failure vectors, colored by run
# ============================================================
algos_with_failures = [n for n in names_present if n in failures_per_algo]

if algos_with_failures:
    all_fail_combined = np.vstack([failures_per_algo[n][0] for n in algos_with_failures])
    pca = PCA(n_components=2)
    pca.fit(all_fail_combined)

    ncols = len(algos_with_failures)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5), sharey=True, sharex=True)
    if ncols == 1:
        axes = [axes]

    for ax, name in zip(axes, algos_with_failures):
        X_norm, run_labels = failures_per_algo[name]
        X_pca = pca.transform(X_norm)
        n_runs = len(np.unique(run_labels))

        sc = ax.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=run_labels, cmap="tab10",
            alpha=0.75, s=45, edgecolors="none",
        )

        # Draw convex hull to visualise coverage area
        if X_pca.shape[0] >= 3:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(X_pca)
                verts = np.append(hull.vertices, hull.vertices[0])
                ax.fill(X_pca[verts, 0], X_pca[verts, 1],
                        alpha=0.08, color=algo_color[name])
                ax.plot(X_pca[verts, 0], X_pca[verts, 1],
                        color=algo_color[name], lw=1, alpha=0.5)
                area_str = f", hull area={hull.volume:.3f}"
            except Exception:
                area_str = ""
        else:
            area_str = ""

        mean_apd = np.mean(apd_per_algo[name])
        ax.set_title(
            f"{name}\n{X_norm.shape[0]} failures across {n_runs} run(s)\n"
            f"mean APD={mean_apd:.4f}{area_str}",
            fontsize=9,
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=9)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=9)
    fig.suptitle(
        "PCA Projection of Failing Test Cases\n"
        "(each dot = one failure, colour = run index; wider spread = more diverse)",
        fontsize=11,
    )
    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "failure_pca_scatter.png")
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Saved: {out2}")


# ============================================================
# Plot 3: Pairwise distance heatmap (representative run per algo)
# ============================================================
ncols = len(algos_with_failures)
fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5))
if ncols == 1:
    axes = [axes]

for ax, name in zip(axes, algos_with_failures):
    apds = apd_per_algo[name]
    # Pick the run closest to median APD
    median_idx = int(np.argsort(apds)[len(apds) // 2])
    X, R = algo_runs[name][median_idx]
    X_fail = get_failures(X, R)

    if X_fail.shape[0] < 2:
        ax.text(0.5, 0.5, "< 2 failures in\nthis run",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(name)
        continue

    X_fail_norm = scaler.transform(X_fail)
    n = X_fail_norm.shape[0]

    # Build pairwise distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(X_fail_norm[i] - X_fail_norm[j])

    im = ax.imshow(dist_matrix, cmap="YlOrRd", aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, shrink=0.75, label="Euclidean distance")
    ax.set_title(
        f"{name}\nrun {median_idx+1} ({n} failures, APD={apds[median_idx]:.4f})",
        fontsize=9,
    )
    ax.set_xlabel("Failure index")
    ax.set_ylabel("Failure index")

fig.suptitle(
    "Pairwise Distance Matrix of Failing Test Cases\n"
    "(brighter = more different scenarios; uniform bright = high diversity)",
    fontsize=11,
)
plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "failure_distance_heatmap.png")
plt.savefig(out3, dpi=150)
plt.close()
print(f"Saved: {out3}")

print("\nDone. All plots saved to:", OUTPUT_DIR)


# ============================================================
# OBJECTIVE-SPACE DIVERSITY
# Measures diversity of TYPES of safety violations, not just
# where in parameter space they occur.
#
# Two metrics:
#   1. Violation Pattern Diversity: each failure = binary vector
#      [V0<0, V1<0, V2<0, V3<0, V4<0]. APD using Hamming distance.
#      Answers: "do we find violations in different objectives?"
#
#   2. Violation Severity APD: pairwise distance on actual score
#      values (only violated objectives, i.e. score < 0).
#      Answers: "are the violations of different severity?"
#
# Reference: objective-space diversity used in multi-objective
# test generation -- Panichella et al. (2018), "Automated Test
# Case Generation as a Many-Objective Optimisation Problem with
# Dynamic Selection of the Targets", IEEE TSE.
# ============================================================
print("\n\n" + "=" * 65)
print("OBJECTIVE-SPACE DIVERSITY ANALYSIS")
print("=" * 65)

REQ_NAMES = ["R0", "R1", "R2"]   # ADAS1 has 3 requirements
N_REQ = len(REQ_NAMES)


def get_failure_scores(X, R):
    """Return R rows where at least one requirement is violated (R > 0)."""
    mask = np.any(R > 0, axis=1)
    return R[mask].astype(float)


def violation_pattern(R_fail):
    """Binary matrix: 1 where requirement violated. R is already binary."""
    return R_fail.astype(float)


def hamming_apd(patterns):
    """Average pairwise Hamming distance between binary violation patterns."""
    n = patterns.shape[0]
    if n < 2:
        return 0.0
    dists = [
        np.sum(patterns[i] != patterns[j]) / N_REQ   # normalised Hamming
        for i, j in combinations(range(n), 2)
    ]
    return float(np.mean(dists))


def severity_apd(F_fail):
    """
    APD on violation severity: use only the violated scores (< 0),
    normalised by the global minimum across all algorithms.
    """
    n = F_fail.shape[0]
    if n < 2:
        return 0.0
    # Clip to violations only (set non-violations to 0)
    F_viol = np.where(F_fail < 0, F_fail, 0.0)
    # Normalise each objective column to [-1, 0]
    col_min = F_viol.min(axis=0)
    col_min = np.where(col_min < 0, col_min, -1e-9)  # avoid /0
    F_norm = F_viol / np.abs(col_min)
    dists = [
        np.linalg.norm(F_norm[i] - F_norm[j])
        for i, j in combinations(range(n), 2)
    ]
    return float(np.mean(dists))


# Compute per-run objective-space metrics
obj_apd_hamming = {}   # name -> list of per-run Hamming APD
obj_apd_severity = {}  # name -> list of per-run severity APD
all_patterns = {}      # name -> stacked binary patterns across all runs
obj_violation_rates = {}  # name -> (N_OBJ,) fraction of failures that violate each obj

for name, runs in algo_runs.items():
    h_apds, s_apds = [], []
    all_pat = []

    for X, R in runs:
        R_fail = get_failure_scores(X, R)
        if R_fail.shape[0] < 2:
            h_apds.append(0.0)
            s_apds.append(0.0)
            continue
        pat = violation_pattern(R_fail)
        h_apds.append(hamming_apd(pat))
        s_apds.append(severity_apd(R_fail))
        all_pat.append(pat)

    obj_apd_hamming[name] = h_apds
    obj_apd_severity[name] = s_apds

    if all_pat:
        stacked = np.vstack(all_pat)
        all_patterns[name] = stacked
        obj_violation_rates[name] = stacked.mean(axis=0)  # fraction violated per obj

# Print objective-space summary
print(f"\n{'Algorithm':<25} {'Hamming APD':>13} {'Severity APD':>14}")
print("-" * 55)
for name in names_present:
    if name not in obj_apd_hamming:
        continue
    h = np.mean(obj_apd_hamming[name])
    s = np.mean(obj_apd_severity[name])
    print(f"{name:<25} {h:>13.4f} {s:>14.4f}")
print("=" * 55)
print("Hamming APD: how often do failures violate DIFFERENT objectives")
print("             (0=always same obj violated, 0.5=maximally diverse)")
print("Severity APD: how varied are the violation magnitudes")

# ============================================================
# Plot 4: Per-objective violation rate (stacked bar)
# ============================================================
names_w_pat = [n for n in names_present if n in obj_violation_rates]
x_pos = np.arange(len(names_w_pat))
bar_width = 0.2
req_colors = ["#e41a1c", "#ff7f00", "#4daf4a"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: grouped bar — violation rate per requirement per algorithm
ax = axes[0]
for ri, (rname, rcolor) in enumerate(zip(REQ_NAMES, req_colors)):
    rates = [obj_violation_rates[n][ri] for n in names_w_pat]
    ax.bar(x_pos + ri * bar_width, rates, bar_width,
           label=rname, color=rcolor, alpha=0.85)

ax.set_xticks(x_pos + bar_width * (N_REQ - 1) / 2)
ax.set_xticklabels(names_w_pat, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Fraction of failures that violate this requirement")
ax.set_ylim(0, 1.05)
ax.legend(title="Requirement", fontsize=8)
ax.set_title("Per-Requirement Violation Rate\n(higher & more balanced = more diverse violation types)", fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Right: Hamming APD box plot
ax = axes[1]
data_h = [obj_apd_hamming[n] for n in names_w_pat]
bp = ax.boxplot(data_h, patch_artist=True, widths=0.45, zorder=2)
for patch, name in zip(bp["boxes"], names_w_pat):
    patch.set_facecolor(algo_color[name])
    patch.set_alpha(0.5)

for i, name in enumerate(names_w_pat):
    y = obj_apd_hamming[name]
    x_jitter = np.random.normal(i + 1, 0.05, size=len(y))
    ax.scatter(x_jitter, y, color=algo_color[name], s=50, zorder=5,
               edgecolors="white", linewidths=0.5)

ax.set_xticks(range(1, len(names_w_pat) + 1))
ax.set_xticklabels(names_w_pat, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Hamming APD (normalised)")
ax.set_title("Violation Type Diversity (Hamming APD)\n"
             "Higher = failures violate different requirements\n"
             f"(max = 0.5 for {N_REQ} requirements)", fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out4 = os.path.join(OUTPUT_DIR, "objective_space_diversity.png")
plt.savefig(out4, dpi=150)
plt.close()
print(f"\nSaved: {out4}")

# ============================================================
# Plot 5: Unique violation patterns heatmap
# Shows which combinations of objectives are violated together
# ============================================================
fig, axes = plt.subplots(1, len(names_w_pat),
                          figsize=(4 * len(names_w_pat), 4))
if len(names_w_pat) == 1:
    axes = [axes]

for ax, name in zip(axes, names_w_pat):
    patterns = all_patterns[name]
    # Count unique patterns
    pattern_strings = ["".join(map(str, row.astype(int))) for row in patterns]
    from collections import Counter
    counts = Counter(pattern_strings)

    # Sort by frequency
    sorted_patterns = sorted(counts.items(), key=lambda x: -x[1])
    pat_labels = [p for p, _ in sorted_patterns]
    pat_counts = [c for _, c in sorted_patterns]

    # Convert pattern string to matrix for heatmap
    mat = np.array([[int(c) for c in p] for p in pat_labels])

    im = ax.imshow(mat.T, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(N_REQ))
    ax.set_yticklabels(REQ_NAMES, fontsize=9)
    ax.set_xticks(range(len(pat_labels)))
    ax.set_xticklabels(
        [f"n={c}" for c in pat_counts], rotation=45, ha="right", fontsize=7
    )
    ax.set_title(
        f"{name}\n{len(counts)} unique patterns\n"
        f"Hamming APD={np.mean(obj_apd_hamming[name]):.3f}",
        fontsize=9,
    )
    ax.set_xlabel("Violation pattern (sorted by frequency)")

fig.suptitle(
    "Unique Violation Patterns per Algorithm\n"
    "(green=violated, red=not violated; more columns = more diverse violation types)",
    fontsize=11,
)
plt.tight_layout()
out5 = os.path.join(OUTPUT_DIR, "violation_patterns.png")
plt.savefig(out5, dpi=150)
plt.close()
print(f"Saved: {out5}")

print("\nAll objective-space diversity plots saved to:", OUTPUT_DIR)
