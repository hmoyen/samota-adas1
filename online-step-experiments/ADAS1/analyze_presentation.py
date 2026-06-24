#!/usr/bin/env python3
"""
Presentation Analysis: PFES Baseline vs PFES+SAMOTA
Produces summary table + 4 publication-quality plots.

Metrics:
  1. Cumulative violations curve        - speed of finding failures
  2. AUCC (Area Under Coverage Curve)   - requirement coverage speed
  3. APD boxplot                        - diversity of failing test inputs
  4. Per-requirement violation rate     - which requirements each algorithm finds

Run: python analyze_presentation.py
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "plots", "presentation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REQ_NAMES = ["R0", "R1", "R2"]
N_REQ     = len(REQ_NAMES)

HOME_DIR = os.path.expanduser("~")

ALGORITHMS = [
    {
        "name":    "PFES Baseline",
        "color":   "#2196F3",
        "primary": os.path.join(BASE_DIR, "results_25runs_pfes", "run_*"),
        "fallback": os.path.join(HOME_DIR, "results_25runs_pfes", "run_*"),
    },
    {
        "name":    "PFES+SAMOTA",
        "color":   "#FF5722",
        "primary": os.path.join(BASE_DIR, "results_25runs_samota", "run_*"),
        "fallback": os.path.join(HOME_DIR, "results_25runs_samota", "run_*"),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_run(run_dir):
    """Return (X, R) where R is binary violation matrix (1=violated).
    Prefers Reqs_all_evaluations; falls back to F_all_evaluations (score<0).

    SAMOTA runs produce F with 5 columns due to a 5-objective NSGA3 setup:
      col 0 = R0, col 1 = R0 (dup), col 2/3 = unused (0), col 4 = R1/R2.
    We remap to 3-column R when F has more columns than N_REQ.
    """
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
    else:
        f_path = os.path.join(run_dir, "F_all_evaluations_NSGA3_0.csv")
        if not os.path.exists(f_path):
            return None, None
        df = pd.read_csv(f_path, header=None)
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        F = df.values
        if F.shape[1] == N_REQ:
            # PFES format: columns map directly to R0, R1, R2
            R = (F < 0).astype(int)
        else:
            # SAMOTA 5-col format: col0=R0, col1=R0(dup), col2/3=unused, col4=R1&R2
            R = np.zeros((F.shape[0], N_REQ), dtype=int)
            R[:, 0] = (F[:, 0] < 0).astype(int)   # R0
            R[:, 1] = (F[:, -1] < 0).astype(int)  # R1 (last active col)
            R[:, 2] = (F[:, -1] < 0).astype(int)  # R2 (same signal as R1)

    n = min(X.shape[0], R.shape[0])
    return X[:n], R[:n]


def load_algo(pattern, fallback=None):
    """Return list of (X, R) for all runs found in pattern (or fallback)."""
    runs = []
    dirs = sorted(glob.glob(pattern))
    if not dirs and fallback:
        dirs = sorted(glob.glob(fallback))
    for d in dirs:
        X, R = load_run(d)
        if X is not None:
            runs.append((X, R))
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
algo_data = {}   # name -> list of (X, R)
for cfg in ALGORITHMS:
    runs = load_algo(cfg["primary"], cfg.get("fallback"))
    algo_data[cfg["name"]] = runs
    cfg["runs"] = runs
    print(f"  {cfg['name']}: {len(runs)} runs")

if not any(algo_data.values()):
    raise SystemExit("No data found. Check directory paths.")

# Global MinMaxScaler fitted on all X (for APD normalisation)
all_X = np.vstack([X for runs in algo_data.values() for X, R in runs])
scaler = MinMaxScaler().fit(all_X)

BUDGET = min(R.shape[0] for runs in algo_data.values() for X, R in runs)
print(f"  Budget (min rows across all runs): {BUDGET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def cumulative_violations(R, budget):
    """Cumulative count of rows where any requirement violated, up to budget."""
    R = R[:budget]
    return np.cumsum(np.any(R > 0, axis=1))


def apd(X_fail_norm):
    """Average Pairwise Euclidean Distance on normalised failing inputs."""
    n = X_fail_norm.shape[0]
    if n < 2:
        return 0.0
    return float(np.mean([
        np.linalg.norm(X_fail_norm[i] - X_fail_norm[j])
        for i, j in combinations(range(n), 2)
    ]))


def aucc(R, budget, reachable):
    """Area Under Coverage Curve, normalised to [0,1].
    At each eval t, count how many reachable requirements have been found so far."""
    R = R[:budget]
    first_found = {}
    for req in reachable:
        idx = np.where(R[:, req] > 0)[0]
        first_found[req] = int(idx[0]) if len(idx) > 0 else np.inf

    coverage = np.zeros(budget)
    for t in range(budget):
        coverage[t] = sum(1 for req in reachable if first_found[req] <= t)

    n_reachable = len(reachable)
    if n_reachable == 0:
        return 0.0, coverage
    return float(np.sum(coverage) / (budget * n_reachable)), coverage


def reachable_requirements(algo_data, threshold=0.05):
    """Requirements violated in >=threshold fraction of runs across all algorithms."""
    all_runs = [R for runs in algo_data.values() for X, R in runs]
    if not all_runs:
        return list(range(N_REQ))
    n_req = all_runs[0].shape[1]
    counts = np.zeros(n_req)
    for R in all_runs:
        for req in range(n_req):
            if np.any(R[:, req] > 0):
                counts[req] += 1
    total = len(all_runs)
    return [req for req in range(n_req) if counts[req] / total >= threshold]


# ─────────────────────────────────────────────────────────────────────────────
# Compute all metrics per algorithm
# ─────────────────────────────────────────────────────────────────────────────
REACHABLE = reachable_requirements(algo_data)
print(f"Reachable requirements (violated in >=5% of runs): "
      f"{[REQ_NAMES[r] for r in REACHABLE]}\n")

metrics = {}  # name -> dict of metric arrays
for cfg in ALGORITHMS:
    name  = cfg["name"]
    runs  = cfg["runs"]
    color = cfg["color"]

    viol_counts, apd_vals, aucc_vals = [], [], []
    cumviol_matrix = []
    aucc_curves    = []
    req_found      = np.zeros(N_REQ)  # fraction of runs that found each req

    for X, R in runs:
        R_b = R[:BUDGET]
        X_b = X[:BUDGET]

        # Violations
        is_viol   = np.any(R_b > 0, axis=1)
        viol_counts.append(int(np.sum(is_viol)))

        # Cumulative curve
        cumviol_matrix.append(np.cumsum(is_viol))

        # APD
        X_fail = X_b[is_viol]
        if X_fail.shape[0] >= 2:
            apd_vals.append(apd(scaler.transform(X_fail)))
        else:
            apd_vals.append(0.0)

        # AUCC
        a, curve = aucc(R_b, BUDGET, REACHABLE)
        aucc_vals.append(a)
        aucc_curves.append(curve)

        # Per-requirement coverage
        for req in range(N_REQ):
            if np.any(R_b[:, req] > 0):
                req_found[req] += 1

    req_found /= len(runs)  # convert to fraction

    metrics[name] = {
        "color":          color,
        "n_runs":         len(runs),
        "viol_counts":    np.array(viol_counts),
        "apd_vals":       np.array(apd_vals),
        "aucc_vals":      np.array(aucc_vals),
        "cumviol_matrix": np.array(cumviol_matrix),   # shape (n_runs, budget)
        "aucc_curves":    np.array(aucc_curves),       # shape (n_runs, budget)
        "req_found":      req_found,                   # shape (N_REQ,)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print(f"{'Metric':<30} {'PFES Baseline':>20} {'PFES+SAMOTA':>20}")
print("-" * 80)

names = [cfg["name"] for cfg in ALGORITHMS]

def fmt(arr):
    return f"{np.mean(arr):.2f} ± {np.std(arr):.2f}"

rows = [
    ("Violations / run",      [fmt(metrics[n]["viol_counts"]) for n in names]),
    ("AUCC (normalised)",     [fmt(metrics[n]["aucc_vals"])   for n in names]),
    ("APD (failure diversity)",[fmt(metrics[n]["apd_vals"])   for n in names]),
]
for ri, rname in enumerate(REQ_NAMES):
    rows.append((
        f"  {rname} coverage (%)",
        [f"{metrics[n]['req_found'][ri]*100:.1f}%" for n in names]
    ))

for label, vals in rows:
    print(f"{label:<30} {vals[0]:>20} {vals[1]:>20}")

print("=" * 80)

# Save CSV
csv_rows = [{"Metric": label, **dict(zip(names, vals))} for label, vals in rows]
pd.DataFrame(csv_rows).to_csv(os.path.join(OUTPUT_DIR, "summary_table.csv"), index=False)
print(f"\nSaved: {OUTPUT_DIR}/summary_table.csv\n")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Cumulative violations over evaluations
# ─────────────────────────────────────────────────────────────────────────────
evals = np.arange(1, BUDGET + 1)
fig, ax = plt.subplots(figsize=(9, 5))

for cfg in ALGORITHMS:
    name  = cfg["name"]
    color = cfg["color"]
    mat   = metrics[name]["cumviol_matrix"]
    mean  = mat.mean(axis=0)
    std   = mat.std(axis=0)
    ax.plot(evals, mean, color=color, lw=2, label=name)
    ax.fill_between(evals, mean - std, mean + std, color=color, alpha=0.15)

ax.axvline(300, color="gray", ls="--", lw=1, label="Phase 1 end (eval 300)")
ax.set_xlabel("Evaluations", fontsize=12)
ax.set_ylabel("Cumulative violations (mean ± std)", fontsize=12)
ax.set_title("Cumulative Violations Over Evaluations\nPFES Baseline vs PFES+SAMOTA", fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "1_cumulative_violations.png")
plt.savefig(out1, dpi=150)
plt.close()
print(f"Saved: {out1}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: AUCC coverage curves
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: mean coverage curve
ax = axes[0]
for cfg in ALGORITHMS:
    name  = cfg["name"]
    color = cfg["color"]
    curves = metrics[name]["aucc_curves"]   # (n_runs, budget)
    mean   = curves.mean(axis=0)
    std    = curves.std(axis=0)
    aucc_m = metrics[name]["aucc_vals"].mean()
    ax.plot(evals, mean, color=color, lw=2,
            label=f"{name}  (AUCC={aucc_m:.3f})")
    # Clip band to valid range [0, n_reachable] — mean±std can exceed bounds
    upper = np.clip(mean + std, 0, len(REACHABLE))
    lower = np.clip(mean - std, 0, len(REACHABLE))
    ax.fill_between(evals, lower, upper, color=color, alpha=0.15)

ax.set_xlabel("Evaluations", fontsize=12)
ax.set_ylabel(f"Requirements covered (out of {len(REACHABLE)})", fontsize=11)
ax.set_title("Requirement Coverage Over Time\n(Area under this curve = AUCC)", fontsize=12)
ax.set_ylim(0, len(REACHABLE) + 0.1)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Right: AUCC boxplot
ax = axes[1]
data  = [metrics[n]["aucc_vals"] for n in names]
colors = [metrics[n]["color"] for n in names]
bp = ax.boxplot(data, patch_artist=True, widths=0.45)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for i, name in enumerate(names):
    y = metrics[name]["aucc_vals"]
    ax.scatter(np.random.normal(i + 1, 0.05, len(y)), y,
               color=metrics[name]["color"], s=50, zorder=5, edgecolors="white", lw=0.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("Normalised AUCC", fontsize=12)
ax.set_title("AUCC Distribution Across 25 Runs", fontsize=12)
ax.grid(axis="y", alpha=0.3)

plt.suptitle("Area Under Coverage Curve (AUCC)", fontsize=14, y=1.01)
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "2_aucc.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: APD boxplot (failure diversity)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: APD boxplot
ax = axes[0]
data = [metrics[n]["apd_vals"] for n in names]
bp = ax.boxplot(data, patch_artist=True, widths=0.45)
for patch, color in zip(bp["boxes"], [metrics[n]["color"] for n in names]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for i, name in enumerate(names):
    y = metrics[name]["apd_vals"]
    ax.scatter(np.random.normal(i + 1, 0.05, len(y)), y,
               color=metrics[name]["color"], s=50, zorder=5, edgecolors="white", lw=0.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("Average Pairwise Distance (APD)", fontsize=12)
ax.set_title("Diversity of Failing Test Cases\n(higher = more spread across parameter space)", fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Right: violations boxplot
ax = axes[1]
data = [metrics[n]["viol_counts"] for n in names]
bp = ax.boxplot(data, patch_artist=True, widths=0.45)
for patch, color in zip(bp["boxes"], [metrics[n]["color"] for n in names]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for i, name in enumerate(names):
    y = metrics[name]["viol_counts"]
    ax.scatter(np.random.normal(i + 1, 0.05, len(y)), y,
               color=metrics[name]["color"], s=50, zorder=5, edgecolors="white", lw=0.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("Number of violations found", fontsize=12)
ax.set_title("Total Violations per Run", fontsize=12)
ax.grid(axis="y", alpha=0.3)

plt.suptitle("Failure Diversity (APD) and Violation Count", fontsize=14, y=1.01)
plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "3_diversity_and_violations.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Per-requirement violation rate
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

x      = np.arange(N_REQ)
width  = 0.35

for i, cfg in enumerate(ALGORITHMS):
    name  = cfg["name"]
    color = cfg["color"]
    rates = metrics[name]["req_found"] * 100
    bars  = ax.bar(x + i * width, rates, width, label=name, color=color, alpha=0.85)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x + width / 2)
ax.set_xticklabels(REQ_NAMES, fontsize=12)
ax.set_ylabel("% of runs where requirement was violated", fontsize=12)
ax.set_ylim(0, 115)
ax.set_title("Requirement Coverage: How Often Each Requirement Is Violated\n(across 25 runs)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out4 = os.path.join(OUTPUT_DIR, "4_requirement_coverage.png")
plt.savefig(out4, dpi=150)
plt.close()
print(f"Saved: {out4}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: PCA of failing test cases
# ─────────────────────────────────────────────────────────────────────────────
fail_pools = {}
for cfg in ALGORITHMS:
    name = cfg["name"]
    all_fail = []
    labels   = []
    for run_idx, (X, R) in enumerate(cfg["runs"]):
        mask = np.any(R[:BUDGET] > 0, axis=1)
        X_fail = X[:BUDGET][mask]
        if X_fail.shape[0] > 0:
            all_fail.append(scaler.transform(X_fail))
            labels.extend([run_idx] * X_fail.shape[0])
    if all_fail:
        fail_pools[name] = (np.vstack(all_fail), np.array(labels))

if len(fail_pools) >= 1:
    all_combined = np.vstack([v[0] for v in fail_pools.values()])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(all_combined)

    fig, axes = plt.subplots(1, len(fail_pools), figsize=(6 * len(fail_pools), 5),
                             sharex=True, sharey=True)
    if len(fail_pools) == 1:
        axes = [axes]

    for ax, (name, (X_norm, run_labels)) in zip(axes, fail_pools.items()):
        X_pca = pca.transform(X_norm)
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=run_labels,
                        cmap="tab20", alpha=0.7, s=30, edgecolors="none")
        if X_pca.shape[0] >= 3:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(X_pca)
                verts = np.append(hull.vertices, hull.vertices[0])
                color = next(cfg["color"] for cfg in ALGORITHMS if cfg["name"] == name)
                ax.fill(X_pca[verts, 0], X_pca[verts, 1], alpha=0.07, color=color)
                ax.plot(X_pca[verts, 0], X_pca[verts, 1], color=color, lw=1, alpha=0.4)
            except Exception:
                pass
        color = next(cfg["color"] for cfg in ALGORITHMS if cfg["name"] == name)
        apd_m = metrics[name]["apd_vals"].mean()
        ax.set_title(f"{name}\n{X_norm.shape[0]} failures · mean APD={apd_m:.4f}",
                     fontsize=10, color=color)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=9)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=9)
    fig.suptitle("PCA of Failing Test Cases\n(each dot = one failure, colour = run index; wider spread = more diverse)",
                 fontsize=12)
    plt.tight_layout()
    out5 = os.path.join(OUTPUT_DIR, "5_failure_pca.png")
    plt.savefig(out5, dpi=150)
    plt.close()
    print(f"Saved: {out5}")

print(f"\nAll plots saved to: {OUTPUT_DIR}/")
print("Done.")
