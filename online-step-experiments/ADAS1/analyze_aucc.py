#!/usr/bin/env python3
"""
AUCC Analysis — Area Under Coverage Curve

Measures how quickly each algorithm covers reachable objectives.
Metrics:
  - Normalized AUCC: fraction of max possible coverage over time (higher = faster)
  - Time-to-full-coverage (TTFC): eval index when all reachable objectives first found
  - Objective coverage curve: #objectives covered over eval budget

Plots saved to plots/aucc/
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots", "aucc")
os.makedirs(PLOTS_DIR, exist_ok=True)

ALGORITHMS = {
    "PFES Baseline":    os.path.join(BASE_DIR, "results_25runs_pfes",           "run_*"),
    "PFES+SAMOTA":      os.path.join(BASE_DIR, "results_25runs_samota_seeded",  "run_*"),
    "PFES+SAMOTA+EI":   os.path.join(BASE_DIR, "results_25runs_ei",             "run_*"),
}
FALLBACK_ALGORITHMS = {
    "PFES Baseline":    os.path.join(BASE_DIR, "results_10runs_pfes",             "run_*"),
    "PFES+SAMOTA":      os.path.join(BASE_DIR, "results_25runs_samota",           "run_*"),
    "PFES+SAMOTA+EI":   os.path.join(BASE_DIR, "results_25runs_ei",              "run_*"),
}

COLORS = {
    "PFES Baseline":   "#2196F3",
    "PFES+SAMOTA":     "#FF5722",
    "PFES+SAMOTA+EI":  "#4CAF50",
}

# Objectives violated in < this fraction of runs are treated as unreachable
REACHABLE_THRESHOLD = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_runs(pattern):
    """Load per-evaluation requirement violation matrix (1=violated, 0=satisfied).
    Prefers Reqs_all_evaluations; falls back to F_all_evaluations (score < 0)."""
    runs = []
    for run_dir in sorted(glob.glob(pattern)):
        reqs_csvs = glob.glob(os.path.join(run_dir, "Reqs_all_evaluations_*.csv"))
        if reqs_csvs:
            df = pd.read_csv(reqs_csvs[0], header=None)
            df = df.replace({"True": 1, "False": 0, True: 1, False: 0})
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            R = (~df.values.astype(bool)).astype(int)
        else:
            f_csvs = glob.glob(os.path.join(run_dir, "F_all_evaluations_*.csv"))
            if not f_csvs:
                continue
            df = pd.read_csv(f_csvs[0], header=None)
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            R = (df.values < 0).astype(int)
        runs.append(R)
    return runs


def detect_reachable(algo_runs):
    """Return list of requirement indices violated in >=REACHABLE_THRESHOLD of all runs."""
    all_R = [R for runs in algo_runs.values() for R in runs]
    if not all_R:
        return []
    n_req = all_R[0].shape[1]
    counts = np.zeros(n_req)
    for R in all_R:
        for req in range(n_req):
            if np.any(R[:, req] > 0):
                counts[req] += 1
    total = len(all_R)
    return [req for req in range(n_req) if counts[req] / total >= REACHABLE_THRESHOLD]


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def first_violation_evals(R, reachable):
    """Dict: req -> first eval index where violated (np.inf if never)."""
    fv = {}
    for req in reachable:
        idx = np.where(R[:, req] > 0)[0]
        fv[req] = int(idx[0]) if len(idx) > 0 else np.inf
    return fv


def coverage_curve(fv, budget, reachable):
    """Array of length budget: number of reachable objectives covered by eval t."""
    fv_arr = np.array([fv[obj] for obj in reachable])
    curve = np.zeros(budget)
    for t in range(budget):
        curve[t] = np.sum(fv_arr <= t)
    return curve


def normalized_aucc(curve, n_reachable):
    if n_reachable == 0:
        return 0.0
    return float(curve.mean() / n_reachable)


def time_to_full_coverage(fv, reachable):
    """First eval where ALL reachable objectives are covered. None if never within budget."""
    if not reachable:
        return None
    t = max(fv[obj] for obj in reachable)
    return None if np.isinf(t) else int(t)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    algo_runs = {}
    for name, pattern in ALGORITHMS.items():
        runs = load_runs(pattern)
        if not runs:
            runs = load_runs(FALLBACK_ALGORITHMS.get(name, ""))
        if runs:
            algo_runs[name] = runs
            print(f"  {name}: {len(runs)} run(s)")
        else:
            print(f"  {name}: not found, skipping")

    if not algo_runs:
        print("No data found.")
        return

    reachable = detect_reachable(algo_runs)
    n_reachable = len(reachable)
    print(f"\nReachable objectives (V-indices): {reachable}  ({n_reachable} total)")

    # Use shortest run as shared budget to keep curves comparable
    budget = min(F.shape[0] for runs in algo_runs.values() for F in runs)
    print(f"Shared budget (rows):  {budget}\n")

    # ── Per-run metrics ───────────────────────────────────────────────────────
    results = {}
    for name, runs in algo_runs.items():
        auccs, ttfcs, curves = [], [], []
        for F in runs:
            fv    = first_violation_evals(F[:budget], reachable)
            curve = coverage_curve(fv, budget, reachable)
            auccs.append(normalized_aucc(curve, n_reachable))
            ttfcs.append(time_to_full_coverage(fv, reachable))
            curves.append(curve)
        results[name] = {
            "auccs":  np.array(auccs),
            "ttfcs":  [t for t in ttfcs if t is not None],
            "never":  sum(1 for t in ttfcs if t is None),
            "curves": np.array(curves),
            "n_runs": len(runs),
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print("=" * 72)
    print(f"{'Algorithm':<22} {'Runs':>5} {'AUCC mean':>10} {'AUCC std':>9} "
          f"{'TTFC mean':>10} {'Full cov%':>10}")
    print("-" * 72)
    for name, r in results.items():
        ttfc_vals = r["ttfcs"]
        mean_ttfc = np.mean(ttfc_vals) if ttfc_vals else float("inf")
        full_pct  = len(ttfc_vals) / r["n_runs"] * 100
        print(f"{name:<22} {r['n_runs']:>5} {r['auccs'].mean():>10.4f} "
              f"{r['auccs'].std():>9.4f} {mean_ttfc:>10.1f} {full_pct:>9.0f}%")
    print("=" * 72)

    evals = np.arange(budget)

    # ── Plot 1: Coverage curve ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in results.items():
        color    = COLORS.get(name, "gray")
        mean_c   = r["curves"].mean(axis=0) / n_reachable
        std_c    = r["curves"].std(axis=0)  / n_reachable
        ax.plot(evals, mean_c, label=name, color=color, linewidth=2)
        ax.fill_between(evals, mean_c - std_c, mean_c + std_c, alpha=0.15, color=color)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6, label="Full coverage")
    ax.set_xlabel("Evaluation index")
    ax.set_ylabel("Fraction of reachable objectives covered")
    ax.set_title("Objective Coverage Over Time (mean ± std)")
    ax.set_ylim(-0.05, 1.15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "coverage_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved: {out}")

    # ── Plot 2: AUCC boxplot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    data   = [r["auccs"] for r in results.values()]
    labels = list(results.keys())
    colors = [COLORS.get(n, "gray") for n in labels]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Normalized AUCC  [0–1]")
    ax.set_title("Area Under Coverage Curve\n(higher = found violations across more objectives earlier)")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "aucc_boxplot.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # ── Plot 3: Time-to-full-coverage boxplot ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    data_ttfc   = [r["ttfcs"] if r["ttfcs"] else [budget] for r in results.values()]
    labels_ttfc = [
        f"{n}\n({len(r['ttfcs'])}/{r['n_runs']} runs reach full coverage)"
        for n, r in results.items()
    ]
    colors_ttfc = [COLORS.get(n, "gray") for n in results]
    bp2 = ax.boxplot(data_ttfc, labels=labels_ttfc, patch_artist=True, widths=0.5)
    for patch, color in zip(bp2["boxes"], colors_ttfc):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(budget, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"Budget ({budget})")
    ax.set_ylabel("Evaluation index")
    ax.set_title("Time-to-Full-Coverage\n(lower = reaches all reachable objectives faster)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "time_to_full_coverage.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    print(f"\nAll AUCC plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
