#!/usr/bin/env python3
"""
Convergence Analysis: how quickly and efficiently do algorithms find violations?

Plots:
  1. cumulative_violations.png   -- cumulative violations over evaluations (mean ± std)
  2. discovery_rate.png          -- new violations per 50-eval window (are we stuck?)
  3. time_to_first_violation.png -- at which eval was each objective first violated?

The key question: does the algorithm get stuck eventually?
  - A curve that flattens early = algorithm stuck, budget wasted
  - A curve that keeps rising = algorithm consistently finding new violations
  - PFES+SAMOTA's Phase 1 (evals 0-300) should look like random sampling,
    Phase 2 (300-900) should show accelerated discovery IF surrogates help.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "plots", "convergence")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALGORITHMS = [
    (
        "PFES Baseline",
        os.path.join(BASE_DIR, "results_25runs_pfes", "run_*"),
        "#2196F3",
    ),
    (
        "PFES+SAMOTA",
        os.path.join(BASE_DIR, "results_25runs_samota", "run_*"),
        "#FF9800",
    ),
    (
        "PFES+SAMOTA+EI",
        os.path.join(BASE_DIR, "results_25runs_ei", "run_*"),
        "#4CAF50",
    ),
]

# Fallback to 10-run directories if 25-run not available yet
FALLBACK_ALGORITHMS = [
    (
        "PFES Baseline",
        os.path.join(BASE_DIR, "results_10runs_pfes", "run_*"),
        "#2196F3",
    ),
    (
        "PFES+SAMOTA",
        os.path.join(BASE_DIR, "results_10runs_samota_900budget", "run_*"),
        "#FF9800",
    ),
    (
        "PFES+SAMOTA+EI",
        os.path.join(BASE_DIR, "pfes_samota_ei"),
        "#4CAF50",
    ),
]

PHASE1_END = 300      # SAMOTA Phase 1 ends here
WINDOW     = 50       # Window size for discovery rate
OBJ_NAMES  = ["V0", "V1", "V2", "V3", "V4"]
N_OBJ      = len(OBJ_NAMES)


# ============================================================
# Data loading
# ============================================================

def load_run(run_dir):
    f_path = os.path.join(run_dir, "F_all_evaluations_NSGA3_0.csv")
    if not os.path.exists(f_path):
        return None
    return pd.read_csv(f_path).values.astype(float)


def load_algorithm(pattern_or_dir):
    matches = sorted(glob.glob(pattern_or_dir))
    if not matches and os.path.isdir(pattern_or_dir):
        matches = [pattern_or_dir]
    runs = []
    for d in matches:
        F = load_run(d)
        if F is not None:
            runs.append(F)
    return runs


def cumulative_violations(F):
    """Cumulative count of test cases with at least one violation, over eval index."""
    is_viol = np.any(F < 0, axis=1)
    return np.cumsum(is_viol)


def discovery_rate(F, window=WINDOW):
    """New violations found per eval window."""
    is_viol = np.any(F < 0, axis=1)
    n = len(is_viol)
    rates = []
    for start in range(0, n, window):
        end = min(start + window, n)
        rates.append(is_viol[start:end].sum())
    return np.array(rates)


def first_violation_eval(F):
    """
    For each objective, return the eval index at which it was first violated.
    Returns inf if never violated.
    """
    result = []
    for obj in range(F.shape[1]):
        viol_evals = np.where(F[:, obj] < 0)[0]
        result.append(viol_evals[0] if len(viol_evals) > 0 else np.inf)
    return result


# ============================================================
# Load data with fallback
# ============================================================
print("Loading data...")
algo_data = {}   # name -> list of F arrays
algo_color = {}

for algo_list in [ALGORITHMS, FALLBACK_ALGORITHMS]:
    for name, pattern, color in algo_list:
        if name in algo_data:
            continue
        runs = load_algorithm(pattern)
        if runs:
            algo_data[name] = runs
            algo_color[name] = color
            print(f"  {name}: {len(runs)} run(s) loaded")

if not algo_data:
    print("No data found. Run experiments first.")
    raise SystemExit(1)

names = [n for n, _, _ in ALGORITHMS if n in algo_data] + \
        [n for n, _, _ in FALLBACK_ALGORITHMS if n in algo_data and
         n not in [x for x, _, _ in ALGORITHMS]]
names = list(dict.fromkeys(names))  # deduplicate preserving order


# ============================================================
# Plot 1: Cumulative violations over evaluations
# ============================================================
print("\nPlotting cumulative violations...")

# Find max eval length across all runs
max_evals = max(F.shape[0] for runs in algo_data.values() for F in runs)
eval_x = np.arange(1, max_evals + 1)

fig, ax = plt.subplots(figsize=(10, 6))

for name in names:
    color = algo_color[name]
    runs = algo_data[name]

    # Pad shorter runs to max_evals
    curves = []
    for F in runs:
        c = cumulative_violations(F)
        if len(c) < max_evals:
            c = np.pad(c, (0, max_evals - len(c)), mode="edge")
        curves.append(c)

    curves = np.array(curves)
    mean = curves.mean(axis=0)
    std  = curves.std(axis=0)

    ax.plot(eval_x, mean, color=color, lw=2, label=f"{name} (n={len(runs)})")
    ax.fill_between(eval_x, mean - std, mean + std, color=color, alpha=0.15)

# Mark Phase 1/2 boundary for SAMOTA
ax.axvline(PHASE1_END, color="gray", linestyle="--", lw=1, alpha=0.7)
ax.text(PHASE1_END + 5, ax.get_ylim()[0] + 1, "Phase 2\nstarts",
        fontsize=8, color="gray", va="bottom")

ax.set_xlabel("Evaluation number", fontsize=12)
ax.set_ylabel("Cumulative violations found", fontsize=12)
ax.set_title(
    "Cumulative Violations Over Time\n"
    "(mean ± std across runs; flattening = algorithm stuck)",
    fontsize=12,
)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "cumulative_violations.png")
plt.savefig(out1, dpi=150)
plt.close()
print(f"  Saved: {out1}")


# ============================================================
# Plot 2: Discovery rate per window (are we getting stuck?)
# ============================================================
print("Plotting discovery rate...")

n_windows = max_evals // WINDOW
window_centers = np.arange(WINDOW // 2, max_evals, WINDOW)[:n_windows]

fig, ax = plt.subplots(figsize=(10, 5))

for name in names:
    color = algo_color[name]
    runs = algo_data[name]

    rates_all = []
    for F in runs:
        r = discovery_rate(F, WINDOW)
        if len(r) < n_windows:
            r = np.pad(r, (0, n_windows - len(r)), mode="constant")
        rates_all.append(r[:n_windows])

    rates_all = np.array(rates_all)
    mean = rates_all.mean(axis=0)
    std  = rates_all.std(axis=0)

    ax.plot(window_centers, mean, color=color, lw=2,
            label=f"{name} (n={len(runs)})", marker="o", ms=4)
    ax.fill_between(window_centers, mean - std, mean + std,
                    color=color, alpha=0.15)

ax.axvline(PHASE1_END, color="gray", linestyle="--", lw=1, alpha=0.7)
ax.text(PHASE1_END + 5, ax.get_ylim()[1] * 0.95, "Phase 2\nstarts",
        fontsize=8, color="gray", va="top")

ax.set_xlabel("Evaluation number (window centre)", fontsize=12)
ax.set_ylabel(f"New violations per {WINDOW} evaluations", fontsize=12)
ax.set_title(
    f"Discovery Rate (violations per {WINDOW}-eval window)\n"
    "Dropping to 0 = algorithm stuck; rising after phase 2 = surrogates helping",
    fontsize=12,
)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "discovery_rate.png")
plt.savefig(out2, dpi=150)
plt.close()
print(f"  Saved: {out2}")


# ============================================================
# Plot 3: Time-to-first-violation per objective
# ============================================================
print("Plotting time-to-first-violation...")

fig, axes = plt.subplots(1, N_OBJ, figsize=(3 * N_OBJ, 5), sharey=False)

for oi, (obj_name, ax) in enumerate(zip(OBJ_NAMES, axes)):
    data_by_algo = []
    labels = []

    for name in names:
        color = algo_color[name]
        ttfv = []
        for F in algo_data[name]:
            first = first_violation_eval(F)
            val = first[oi]
            if val != np.inf:
                ttfv.append(val)
        # Runs that never violated this objective → show as max_evals+
        n_never = len(algo_data[name]) - len(ttfv)
        data_by_algo.append((ttfv, n_never, algo_color[name], name))
        labels.append(name)

    # Box plot of eval-to-first-violation
    positions = range(1, len(data_by_algo) + 1)
    for pos, (ttfv, n_never, color, name) in zip(positions, data_by_algo):
        if ttfv:
            bp = ax.boxplot([ttfv], positions=[pos], widths=0.5,
                            patch_artist=True,
                            boxprops=dict(facecolor=color, alpha=0.5),
                            medianprops=dict(color="black", lw=2))
            ax.scatter([pos] * len(ttfv),
                       ttfv, color=color, s=30, zorder=5,
                       edgecolors="white", lw=0.5)
        if n_never > 0:
            ax.text(pos, max_evals * 1.02,
                    f"{n_never} runs\nnever",
                    ha="center", fontsize=7, color="red")

    ax.set_xticks(list(positions))
    ax.set_xticklabels(
        [n.replace("PFES+SAMOTA+", "PFES+\nSAMOTA+").replace("PFES+SAMOTA", "PFES+\nSAMOTA")
         for n in labels],
        fontsize=7, rotation=20, ha="right"
    )
    ax.set_title(obj_name, fontsize=11, fontweight="bold")
    ax.set_ylim(-10, max_evals * 1.15)
    ax.set_ylabel("Eval index of first violation" if oi == 0 else "")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(PHASE1_END, color="gray", linestyle="--", lw=1, alpha=0.5)

fig.suptitle(
    "Time-to-First-Violation per Objective\n"
    "(lower = found earlier; 'never' = objective never violated in that run;\n"
    " dashed line = end of Phase 1 for SAMOTA)",
    fontsize=11,
)
plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "time_to_first_violation.png")
plt.savefig(out3, dpi=150)
plt.close()
print(f"  Saved: {out3}")

print("\nDone. All convergence plots saved to:", OUTPUT_DIR)
