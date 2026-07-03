#!/usr/bin/env python3
"""
Box plot: % of simulation budget used to reach first violation of ALL violatable requirements,
per algorithm across 30 independent runs.

- X-axis: Algorithm
- Y-axis: % of budget (0–100%)
- Each box = distribution across 30 runs
- Runs where full coverage was never achieved are excluded from the distribution
  and annotated as "N/A" counts.

Usage:
  python plot_coverage_boxplot.py --results_dir results --benchmark ADAS1
  python plot_coverage_boxplot.py --results_dir results --benchmark ADAS2 --budget 900
  python plot_coverage_boxplot.py --results_dir results  # all benchmarks, one figure per benchmark
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================================
# Configuration
# ============================================================================

ALGORITHMS = ["PF", "RS", "FF", "MERLOT", "SAMOTA", "SAMOTA_SW"]
ALG_LABELS = {
    "PF":       "PF\n(NSGA3)",
    "RS":       "RS\n(Random)",
    "FF":       "FF\n(Focused)",
    "MERLOT":   "MERLOT\n(RL)",
    "SAMOTA":   "SAMOTA",
    "SAMOTA_SW":"SAMOTA\n+SW",
}
ALG_COLORS = {
    "PF":       "#4C72B0",
    "RS":       "#DD8452",
    "FF":       "#55A868",
    "MERLOT":   "#C44E52",
    "SAMOTA":   "#8172B2",
    "SAMOTA_SW":"#937860",
}

PFES_PREFIXES = {"PF": "NSGA3", "RS": "RANDOM"}
TIMING_PREFIXES = {"FF": "FOC", "MERLOT": "MORLOT", "SAMOTA": "SAMOTA", "SAMOTA_SW": "SAMOTA"}


# ============================================================================
# Loaders
# ============================================================================

def load_pfes_timing(bench_dir: Path, alg: str, budget: int):
    """
    Compute first-coverage eval from per-run Reqs_all_evaluations_{PREFIX}_{run}.csv files.
    Each file has one row per evaluation with boolean columns R0, R1, ..., Rn.
    A violation = R_i == False (requirement NOT satisfied).
    Returns list of (full_coverage_eval | None) per run, one entry per run found.
    """
    prefix = PFES_PREFIXES[alg]
    pattern = str(bench_dir / f"Reqs_all_evaluations_{prefix}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return []

    results = []
    for f in files:
        df = pd.read_csv(f)
        req_cols = [c for c in df.columns if c.startswith("R")]
        if not req_cols:
            results.append(None)
            continue

        # Boolean: True = satisfied, False = violated
        # first_viol[r] = 1-indexed eval when req r first violated
        first_viol = {}
        for col in req_cols:
            violated_rows = df.index[~df[col].astype(bool)]
            if len(violated_rows) > 0:
                first_viol[col] = int(violated_rows[0]) + 1  # 1-indexed
            else:
                first_viol[col] = None

        # Only count reqs that are violatable (violated in this run)
        violatable = [c for c in req_cols if first_viol[c] is not None]
        if not violatable:
            results.append(None)
        else:
            results.append(max(first_viol[c] for c in violatable))

    return results


def load_timing_csv(bench_dir: Path, alg: str):
    """
    Load timing_*.csv and return list of (full_coverage_eval | None) per run.
    """
    prefix = TIMING_PREFIXES[alg]
    pattern = str(bench_dir / f"timing_{prefix}_*.csv")
    files = glob.glob(pattern)
    if not files:
        return []

    # Pick the latest file (highest N)
    files.sort(key=lambda f: int(Path(f).stem.split("_")[-1]) if Path(f).stem.split("_")[-1].isdigit() else 0)
    f = files[-1]
    df = pd.read_csv(f)

    if "full_coverage_eval" not in df.columns:
        return [None] * len(df)

    return [None if pd.isna(v) else int(v) for v in df["full_coverage_eval"]]


# ============================================================================
# Determine violatable requirements across all runs for PFES
# ============================================================================

def get_violatable_reqs_pfes(bench_dir: Path, prefix: str):
    """Return set of req column names violated at least once across any run."""
    pattern = str(bench_dir / f"Reqs_all_evaluations_{prefix}_*.csv")
    files = glob.glob(pattern)
    violatable = set()
    for f in files:
        df = pd.read_csv(f)
        for col in df.columns:
            if col.startswith("R") and (~df[col].astype(bool)).any():
                violatable.add(col)
    return violatable


# ============================================================================
# Main
# ============================================================================

def collect_data(results_dir: Path, benchmark: str, budget: int):
    """
    Returns dict: alg -> list of full_coverage_pct (float 0-100 or None).
    """
    data = {}
    for alg in ALGORITHMS:
        bench_dir = results_dir / benchmark / alg / "out"
        if not bench_dir.exists():
            continue

        if alg in PFES_PREFIXES:
            evals = load_pfes_timing(bench_dir, alg, budget)
        else:
            evals = load_timing_csv(bench_dir, alg)

        if not evals:
            continue

        # Convert to % of budget
        pcts = [100.0 * e / budget if e is not None else None for e in evals]
        data[alg] = pcts

    return data


def plot_benchmark(benchmark: str, data: dict, budget: int, save_path: Path = None):
    algs_present = [a for a in ALGORITHMS if a in data]
    if not algs_present:
        print(f"  No data found for {benchmark}")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(algs_present) * 1.5), 5))

    box_data = []
    na_counts = []
    labels = []
    colors = []

    for alg in algs_present:
        pcts = data[alg]
        achieved = [p for p in pcts if p is not None]
        na = len(pcts) - len(achieved)
        box_data.append(achieved if achieved else [float("nan")])
        na_counts.append(na)
        labels.append(ALG_LABELS.get(alg, alg))
        colors.append(ALG_COLORS.get(alg, "#888888"))

    bp = ax.boxplot(box_data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Annotate N/A counts
    for i, (na, box_vals) in enumerate(zip(na_counts, box_data), 1):
        if na > 0:
            ax.annotate(f"{na} N/A", xy=(i, ax.get_ylim()[1]),
                        ha="center", va="bottom", fontsize=8, color="gray")

    ax.set_xticks(range(1, len(algs_present) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% of budget to first cover all violatable reqs", fontsize=10)
    ax.set_title(f"{benchmark} — Coverage Speed (budget={budget})", fontsize=11)
    ax.set_ylim(0, 110)
    ax.axhline(100, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="Full budget")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results", help="Top-level results directory")
    parser.add_argument("--benchmark", default=None, help="Single benchmark (ADAS1/ADAS2/RR). Default: all.")
    parser.add_argument("--budget", type=int, default=900, help="Evaluation budget per run")
    parser.add_argument("--save", action="store_true", help="Save plots as PNG instead of showing")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    benchmarks = [args.benchmark] if args.benchmark else ["ADAS1", "ADAS2", "RR"]

    for benchmark in benchmarks:
        print(f"\nProcessing {benchmark}...")
        data = collect_data(results_dir, benchmark, args.budget)

        if not data:
            print(f"  No timing data found in {results_dir / benchmark}")
            continue

        # Summary table
        print(f"\n  Algorithm  | N runs | N/A | Median % | Mean %  | Std %")
        print(f"  {'-'*60}")
        for alg in ALGORITHMS:
            if alg not in data:
                continue
            pcts = data[alg]
            achieved = [p for p in pcts if p is not None]
            na = len(pcts) - len(achieved)
            if achieved:
                med = np.median(achieved)
                mean = np.mean(achieved)
                std = np.std(achieved)
                print(f"  {alg:<10} | {len(pcts):>6} | {na:>3} | {med:>8.1f} | {mean:>7.1f} | {std:>5.1f}")
            else:
                print(f"  {alg:<10} | {len(pcts):>6} | {na:>3} | {'N/A':>8} | {'N/A':>7} | {'N/A':>5}")

        save_path = (results_dir / f"boxplot_coverage_{benchmark}.png") if args.save else None
        plot_benchmark(benchmark, data, args.budget, save_path)

    if not args.save:
        print("\nShowing plots interactively (use --save to write PNG files).")


if __name__ == "__main__":
    main()
