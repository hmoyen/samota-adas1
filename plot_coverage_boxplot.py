#!/usr/bin/env python3
"""
Box plot: % of simulation budget used to reach first violation of ALL violatable requirements,
per algorithm across 30 independent runs.

- X-axis: Algorithm
- Y-axis: % of budget (0–100%)
- Each box = distribution across 30 runs
- Runs where full coverage was never achieved are punished at 100% of budget
  (spent the whole budget without covering everything) and annotated as
  "capped" counts, rather than being excluded from the distribution.

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

def load_pfes_timing(bench_dir: Path, alg: str, budget: int, violatable_reqs=None):
    """
    Compute first-coverage eval from per-run Reqs_all_evaluations_{PREFIX}_{run}.csv files.
    Each file has one row per evaluation with boolean columns R0, R1, ..., Rn.
    A violation = R_i == False (requirement NOT satisfied).
    If violatable_reqs is given, "full coverage" only requires those req columns
    (rather than every column in the file) to have been violated.
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
        if violatable_reqs is not None:
            req_cols = [c for c in req_cols if c in violatable_reqs]
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

        # Require all of the (possibly restricted) req set covered
        if all(first_viol[c] is not None for c in req_cols):
            results.append(max(first_viol[c] for c in req_cols))
        else:
            results.append(None)

    return results


def load_timing_csv(bench_dir: Path, alg: str, violatable_reqs=None):
    """
    Load timing_*.csv and return list of (full_coverage_eval | None) per run.

    Recomputes full coverage from the per-req R{i}_first_eval columns rather than
    trusting the file's own full_coverage_eval column, so it can be restricted to
    violatable_reqs (requirements actually observed as violated by at least one
    algorithm being compared) instead of requiring every declared requirement.
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

    req_cols = [c for c in df.columns if c.endswith("_first_eval")]
    if violatable_reqs is not None:
        req_cols = [c for c in req_cols if c[: -len("_first_eval")] in violatable_reqs]

    if not req_cols:
        return [None] * len(df)

    results = []
    for _, row in df.iterrows():
        vals = [row[c] for c in req_cols]
        if any(pd.isna(v) for v in vals):
            results.append(None)
        else:
            results.append(int(max(vals)))
    return results


# ============================================================================
# Determine violatable requirements across all runs, for either data format
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


def get_violatable_reqs_timing(bench_dir: Path, prefix: str):
    """Return set of req names (e.g. 'R0') violated at least once, from the
    latest timing_*.csv file's R{i}_first_eval columns."""
    pattern = str(bench_dir / f"timing_{prefix}_*.csv")
    files = glob.glob(pattern)
    if not files:
        return set()

    files.sort(key=lambda f: int(Path(f).stem.split("_")[-1]) if Path(f).stem.split("_")[-1].isdigit() else 0)
    df = pd.read_csv(files[-1])

    violatable = set()
    for col in df.columns:
        if col.endswith("_first_eval") and df[col].notna().any():
            violatable.add(col[: -len("_first_eval")])
    return violatable


# ============================================================================
# Main
# ============================================================================

def collect_data(results_dir: Path, benchmark: str, budget: int, algorithms=None):
    """
    Returns (data, violatable_reqs):
      - data: dict alg -> list of full_coverage_pct (float 0-100 or None)
      - violatable_reqs: set of req names (e.g. {'R0', 'R2'}) used as the "full
        coverage" bar -- the union of requirements observed as violated by at
        least one of the included algorithms that has data present. This is
        looser than requiring every declared requirement, so benchmarks where
        some requirements are never violated by anyone still get a meaningful
        (non-all-N/A) comparison.
    """
    algs = algorithms or ALGORITHMS
    bench_dirs = {}
    for alg in algs:
        bench_dir = results_dir / benchmark / alg / "out"
        if bench_dir.exists():
            bench_dirs[alg] = bench_dir

    violatable = set()
    for alg, bench_dir in bench_dirs.items():
        if alg in PFES_PREFIXES:
            violatable |= get_violatable_reqs_pfes(bench_dir, PFES_PREFIXES[alg])
        else:
            violatable |= get_violatable_reqs_timing(bench_dir, TIMING_PREFIXES[alg])

    data = {}
    for alg, bench_dir in bench_dirs.items():
        if alg in PFES_PREFIXES:
            evals = load_pfes_timing(bench_dir, alg, budget, violatable_reqs=violatable)
        else:
            evals = load_timing_csv(bench_dir, alg, violatable_reqs=violatable)

        if not evals:
            continue

        # Convert to % of budget
        pcts = [100.0 * e / budget if e is not None else None for e in evals]
        data[alg] = pcts

    return data, violatable


def plot_benchmark(benchmark: str, data: dict, budget: int, save_path: Path = None, n_violatable: int = None):
    algs_present = [a for a in ALGORITHMS if a in data]
    if not algs_present:
        print(f"  No data found for {benchmark}")
        return

    fig, (ax_rate, ax_speed) = plt.subplots(
        1, 2, figsize=(max(9, len(algs_present) * 2.2), 5),
        gridspec_kw={"width_ratios": [1, 1.4]},
    )

    box_data = []
    rates = []
    n_runs = []
    labels = []
    colors = []

    for alg in algs_present:
        pcts = data[alg]
        achieved = [p for p in pcts if p is not None]
        box_data.append(achieved if achieved else [float("nan")])
        rates.append(100.0 * len(achieved) / len(pcts) if pcts else 0.0)
        n_runs.append(len(pcts))
        labels.append(ALG_LABELS.get(alg, alg))
        colors.append(ALG_COLORS.get(alg, "#888888"))

    # Left panel: coverage rate (% of runs that ever achieved full coverage)
    bars = ax_rate.bar(range(1, len(algs_present) + 1), rates, color=colors, alpha=0.85)
    for i, (rate, n) in enumerate(zip(rates, n_runs), 1):
        n_hit = round(rate / 100.0 * n)
        ax_rate.annotate(f"{n_hit}/{n}", xy=(i, rate), xytext=(0, 3),
                          textcoords="offset points", ha="center", fontsize=8)
    ax_rate.set_xticks(range(1, len(algs_present) + 1))
    ax_rate.set_xticklabels(labels, fontsize=9)
    ax_rate.set_ylabel("% of runs achieving full coverage", fontsize=10)
    ax_rate.set_ylim(0, 105)
    ax_rate.set_title("Coverage rate", fontsize=11)
    ax_rate.grid(axis="y", alpha=0.3)

    # Right panel: speed among runs that succeeded, conditional on success
    bp = ax_speed.boxplot(box_data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax_speed.set_xticks(range(1, len(algs_present) + 1))
    ax_speed.set_xticklabels(labels, fontsize=9)
    ax_speed.set_ylabel("% of budget to first cover (successful runs only)", fontsize=10)
    ax_speed.set_ylim(0, 110)
    ax_speed.set_title("Coverage speed (conditional on success)", fontsize=11)
    ax_speed.axhline(100, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="Full budget")
    ax_speed.legend(fontsize=8, loc="upper right")
    ax_speed.grid(axis="y", alpha=0.3)

    title = f"{benchmark} — Coverage Rate & Speed (budget={budget}"
    title += f", {n_violatable} violatable reqs)" if n_violatable is not None else ")"
    fig.suptitle(title, fontsize=12)
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
    parser.add_argument("--algorithms", nargs="+", default=None, choices=ALGORITHMS,
                        help="Restrict to these algorithms only (default: all six)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    benchmarks = [args.benchmark] if args.benchmark else ["ADAS1", "ADAS2", "RR"]

    for benchmark in benchmarks:
        print(f"\nProcessing {benchmark}...")
        data, violatable = collect_data(results_dir, benchmark, args.budget, args.algorithms)

        if not data:
            print(f"  No timing data found in {results_dir / benchmark}")
            continue

        print(f"  'Full coverage' = violating all of {sorted(violatable)} "
              f"({len(violatable)} reqs observed as violated by at least one included algorithm)")

        # Summary table: coverage rate (all runs) + speed stats (successful runs only)
        print(f"\n  Algorithm  | N runs | Coverage rate | Median % | Mean %  | Std %  (speed, successes only)")
        print(f"  {'-'*90}")
        for alg in ALGORITHMS:
            if alg not in data:
                continue
            pcts = data[alg]
            achieved = [p for p in pcts if p is not None]
            rate = 100.0 * len(achieved) / len(pcts) if pcts else 0.0
            if achieved:
                med = np.median(achieved)
                mean = np.mean(achieved)
                std = np.std(achieved)
                print(f"  {alg:<10} | {len(pcts):>6} | {len(achieved):>3}/{len(pcts):<3} ({rate:>5.1f}%) | {med:>8.1f} | {mean:>7.1f} | {std:>5.1f}")
            else:
                print(f"  {alg:<10} | {len(pcts):>6} | {0:>3}/{len(pcts):<3} ({rate:>5.1f}%) | {'N/A':>8} | {'N/A':>7} | {'N/A':>5}")

        save_path = (results_dir / f"boxplot_coverage_{benchmark}.png") if args.save else None
        plot_benchmark(benchmark, data, args.budget, save_path, n_violatable=len(violatable))

    if not args.save:
        print("\nShowing plots interactively (use --save to write PNG files).")


if __name__ == "__main__":
    main()
