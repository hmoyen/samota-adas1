#!/usr/bin/env python3
"""
Analyze results from all algorithm/benchmark experiments.

Loads CSV output files, computes metrics, runs statistical tests,
and prints a publication-quality comparison table.

Metrics computed per algorithm per benchmark:
  - Total violations (sum over all runs and all requirements)
  - Mean violations per run (± std)
  - Requirement coverage: fraction of requirements violated at least once per run (mean)
  - Best score (mean minimum score across objectives, lower = harder violations found)
  - Efficiency: violations / budget

Statistical tests (pairwise, SAMOTA vs each baseline):
  - Mann-Whitney U (two-sided, p < 0.05)
  - Vargha-Delaney A_AB effect size

Usage:
  python analyze_all_results.py
  python analyze_all_results.py --results_dir my_results --nruns 30
  python analyze_all_results.py --save_tables  # saves CSV tables to results/tables/
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

BENCHMARKS = ["ADAS1", "ADAS2", "RR"]

# Maps CLI algorithm name -> CSV filename prefix
ALGO_CSV_PREFIX = {
    "PF":       "NSGA3",
    "RS":       "RANDOM",
    "FF":       "FOC",
    "MERLOT":   "MORLOT",
    "SAMOTA":   "SAMOTA",
    "SAMOTA_SW": "SAMOTA",   # same script, same prefix — lives in SAMOTA_SW/ subdir
}

ALGO_DISPLAY = {
    "PF":       "PF (NSGA3)",
    "RS":       "RS (Random)",
    "FF":       "FF (FOC)",
    "MERLOT":   "MERLOT",
    "SAMOTA":   "SAMOTA",
    "SAMOTA_SW": "SAMOTA+SW",
}

ALGORITHM_ORDER = ["PF", "RS", "FF", "MERLOT", "SAMOTA", "SAMOTA_SW"]

NREQS = {
    "ADAS1": 3,
    "ADAS2": 6,
    "RR":    6,
}

OBJECTIVES = {
    "ADAS1": 5,
    "ADAS2": 5,
    "RR":    5,
}


# ============================================================================
# DATA LOADING
# ============================================================================

def find_csv(out_dir: Path, prefix: str, kind: str) -> Path | None:
    """
    Find reqs_{prefix}_{N}.csv or score_{prefix}_{N}.csv in out_dir.
    Returns the path or None if not found.
    """
    pattern_kind = "reqs" if kind == "reqs" else "score"
    for p in out_dir.glob(f"{pattern_kind}_{prefix}_*.csv"):
        return p
    return None


def load_algorithm_results(results_dir: Path, benchmark: str, algorithm: str, nruns: int):
    """
    Load reqs, score, and optional meta CSVs for one (benchmark, algorithm) pair.
    Returns (reqs_df, score_df, meta_df_or_None) or (None, None, None) if files not found.
    """
    out_dir = results_dir / benchmark / algorithm / "out"
    prefix = ALGO_CSV_PREFIX.get(algorithm, algorithm)

    reqs_path = find_csv(out_dir, prefix, "reqs")
    score_path = find_csv(out_dir, prefix, "score")

    if reqs_path is None or not reqs_path.exists():
        return None, None, None
    if score_path is None or not score_path.exists():
        return None, None, None

    try:
        reqs_df = pd.read_csv(reqs_path)
        score_df = pd.read_csv(score_path)
        # Try loading meta (timing/cost) CSV if present
        meta_df = None
        for p in out_dir.glob(f"meta_{prefix}_*.csv"):
            meta_df = pd.read_csv(p)
            break
        return reqs_df, score_df, meta_df
    except Exception as e:
        print(f"  [WARN] Could not load {reqs_path} or {score_path}: {e}")
        return None, None, None


# ============================================================================
# METRICS
# ============================================================================

def req_columns(df: pd.DataFrame) -> list[str]:
    """Return R0, R1, ... columns from a reqs dataframe."""
    return [c for c in df.columns if c.startswith("R") and c[1:].isdigit()]


def compute_metrics(reqs_df: pd.DataFrame, score_df: pd.DataFrame, benchmark: str,
                    meta_df: pd.DataFrame = None) -> dict:
    """
    Compute all comparison metrics from a single (reqs_df, score_df, meta_df) tuple.
    Each row = one independent run.
    """
    req_cols = req_columns(reqs_df)
    n_reqs = len(req_cols)
    n_runs = len(reqs_df)

    # Total violations per run (sum across requirements)
    total_viol_per_run = reqs_df[req_cols].sum(axis=1).values

    # Per-run requirement coverage: fraction of reqs violated >= 1 time
    coverage_per_run = (reqs_df[req_cols] > 0).sum(axis=1).values / n_reqs

    # Score metrics (lower = closer to/past violation boundary)
    score_cols = [c for c in score_df.columns if c.startswith("V")]
    best_score_per_run = score_df[score_cols].min(axis=1).values  # min over objectives

    metrics = {
        "n_runs": n_runs,
        "n_reqs": n_reqs,
        # Violation counts
        "total_viol_per_run": total_viol_per_run,
        "mean_viol": float(np.mean(total_viol_per_run)),
        "median_viol": float(np.median(total_viol_per_run)),
        "std_viol": float(np.std(total_viol_per_run)),
        "sum_viol": int(np.sum(total_viol_per_run)),
        # Per-requirement breakdown
        "per_req_mean": {col: float(reqs_df[col].mean()) for col in req_cols},
        # Coverage
        "coverage_per_run": coverage_per_run,
        "mean_coverage": float(np.mean(coverage_per_run)),
        # Best score (min fitness, lower is better)
        "best_score_per_run": best_score_per_run,
        "mean_best_score": float(np.mean(best_score_per_run)),
        # Raw for stats tests
        "raw_reqs_df": reqs_df,
        "raw_score_df": score_df,
        # Timing / cost (from meta_df if available)
        "mean_elapsed_s": None,
        "mean_efficiency": None,
    }

    if meta_df is not None and "elapsed_s" in meta_df.columns:
        metrics["mean_elapsed_s"] = float(meta_df["elapsed_s"].mean())
        metrics["mean_elapsed_min"] = float(meta_df["elapsed_s"].mean() / 60)
        if "efficiency" in meta_df.columns:
            metrics["mean_efficiency"] = float(meta_df["efficiency"].mean())

    return metrics


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def vargha_delaney(a: np.ndarray, b: np.ndarray) -> float:
    """
    Vargha-Delaney A_12 effect size.
    A > 0.5 means a tends to be larger than b.
    A = 0.5 means no difference.
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.5
    rank_sum = 0.0
    for xi in a:
        for yi in b:
            if xi > yi:
                rank_sum += 1
            elif xi == yi:
                rank_sum += 0.5
    return rank_sum / (m * n)


def mann_whitney_test(a: np.ndarray, b: np.ndarray):
    """
    Two-sided Mann-Whitney U test. Returns (statistic, p_value).
    """
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    try:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def effect_size_label(a_ab: float) -> str:
    """
    Interpret Vargha-Delaney A_AB effect size magnitude.
    Convention: |0.5 - A| determines effect.
    """
    diff = abs(a_ab - 0.5)
    if diff < 0.06:
        return "negligible"
    elif diff < 0.14:
        return "small"
    elif diff < 0.21:
        return "medium"
    else:
        return "large"


def run_statistical_tests(all_metrics: dict, reference_algo: str = "SAMOTA"):
    """
    For each benchmark, compare reference_algo against each other algorithm.
    Returns nested dict: {benchmark: {algo: {stat results}}}
    """
    results = {}
    for benchmark, bench_data in all_metrics.items():
        results[benchmark] = {}
        ref_metrics = bench_data.get(reference_algo)
        if ref_metrics is None:
            continue

        ref_viol = ref_metrics["total_viol_per_run"]

        for algo, metrics in bench_data.items():
            if algo == reference_algo:
                continue
            other_viol = metrics["total_viol_per_run"]

            stat, p = mann_whitney_test(ref_viol, other_viol)
            a_ab = vargha_delaney(ref_viol, other_viol)

            results[benchmark][algo] = {
                "mw_stat": stat,
                "p_value": p,
                "significant": p < 0.05 if not np.isnan(p) else False,
                "a_ab": a_ab,
                "effect": effect_size_label(a_ab),
                "direction": "SAMOTA better" if a_ab > 0.5 else ("equal" if abs(a_ab - 0.5) < 0.06 else "baseline better"),
            }
    return results


# ============================================================================
# DISPLAY
# ============================================================================

def print_summary_table(all_metrics: dict, stat_tests: dict, nruns: int):
    """Print a formatted comparison table to stdout."""
    col_w = 14
    header_w = 10

    print(f"\n{'='*80}")
    print(f"COMPARISON TABLE: Mean violations per run (± std) across {nruns} runs")
    print(f"{'='*80}")

    # Header row
    header = f"{'Benchmark':{header_w}} | {'Algorithm':{header_w}} | {'Mean±Std':^12} | {'Coverage':^9} | {'BestScore':^10} | {'p-val':^7} | {'A_AB':^6} | Effect"
    print(header)
    print("-" * len(header))

    for benchmark in BENCHMARKS:
        bench_data = all_metrics.get(benchmark, {})
        bench_stats = stat_tests.get(benchmark, {})

        for i, algo in enumerate(ALGORITHM_ORDER):
            metrics = bench_data.get(algo)
            bench_label = benchmark if i == 0 else ""

            if metrics is None:
                row = f"{bench_label:{header_w}} | {ALGO_DISPLAY[algo]:{header_w}} | {'N/A':^12} | {'N/A':^9} | {'N/A':^10} | {'N/A':^7} | {'N/A':^6} | N/A"
                print(row)
                continue

            mean_v = metrics["mean_viol"]
            std_v = metrics["std_viol"]
            cov = metrics["mean_coverage"] * 100
            best_s = metrics["mean_best_score"]

            viol_str = f"{mean_v:.1f}±{std_v:.1f}"
            cov_str = f"{cov:.1f}%"
            score_str = f"{best_s:.4f}"

            # Statistical comparison vs SAMOTA
            if algo == "SAMOTA":
                p_str = "REF"
                a_str = "REF"
                eff_str = "(reference)"
            elif algo in bench_stats:
                ts = bench_stats[algo]
                p_str = f"{ts['p_value']:.3f}{'*' if ts['significant'] else ''}"
                a_str = f"{ts['a_ab']:.2f}"
                eff_str = ts["effect"]
            else:
                p_str = "N/A"
                a_str = "N/A"
                eff_str = "N/A"

            row = f"{bench_label:{header_w}} | {ALGO_DISPLAY[algo]:{header_w}} | {viol_str:^12} | {cov_str:^9} | {score_str:^10} | {p_str:^7} | {a_str:^6} | {eff_str}"
            print(row)

        print("-" * len(header))


def print_per_req_table(all_metrics: dict):
    """Print per-requirement violation breakdown."""
    print(f"\n{'='*80}")
    print(f"PER-REQUIREMENT VIOLATIONS (mean violations/run)")
    print(f"{'='*80}")

    for benchmark in BENCHMARKS:
        bench_data = all_metrics.get(benchmark, {})
        n_reqs = NREQS[benchmark]
        req_labels = [f"R{i}" for i in range(n_reqs)]

        print(f"\n  {benchmark}:")
        header = f"  {'Algorithm':{12}} | " + " | ".join(f"{r:^7}" for r in req_labels)
        print(header)
        print("  " + "-" * (len(header) - 2))

        for algo in ALGORITHM_ORDER:
            metrics = bench_data.get(algo)
            if metrics is None:
                vals = "  ".join(["N/A"] * n_reqs)
                print(f"  {ALGO_DISPLAY[algo]:{12}} | {vals}")
                continue

            per_req = metrics["per_req_mean"]
            vals = " | ".join(f"{per_req.get(f'R{i}', 0.0):^7.1f}" for i in range(n_reqs))
            print(f"  {ALGO_DISPLAY[algo]:{12}} | {vals}")


def print_stat_tests_detail(stat_tests: dict):
    """Print detailed statistical test results."""
    print(f"\n{'='*80}")
    print(f"STATISTICAL TESTS: SAMOTA vs Baselines (Mann-Whitney U, Vargha-Delaney A_AB)")
    print(f"  * = p < 0.05 (statistically significant)")
    print(f"  A_AB > 0.5 means SAMOTA finds more violations")
    print(f"{'='*80}")

    for benchmark in BENCHMARKS:
        bench_stats = stat_tests.get(benchmark, {})
        if not bench_stats:
            print(f"\n  {benchmark}: no data")
            continue

        print(f"\n  {benchmark}:")
        print(f"    {'vs':12} | {'p-value':^10} | {'Significant':^12} | {'A_AB':^6} | {'Effect':^12} | Direction")
        print("    " + "-" * 75)
        for algo in ALGORITHM_ORDER:
            if algo == "SAMOTA":
                continue
            ts = bench_stats.get(algo)
            if ts is None:
                print(f"    {ALGO_DISPLAY[algo]:12} | {'N/A':^10} | {'N/A':^12} | {'N/A':^6} | {'N/A':^12} | N/A")
                continue
            sig = "YES *" if ts["significant"] else "no"
            print(f"    {ALGO_DISPLAY[algo]:12} | {ts['p_value']:^10.4f} | {sig:^12} | {ts['a_ab']:^6.3f} | {ts['effect']:^12} | {ts['direction']}")


# ============================================================================
# SAVE TABLES
# ============================================================================

def save_tables(all_metrics: dict, stat_tests: dict, tables_dir: Path, nruns: int):
    """Save summary tables to CSV files."""
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for benchmark in BENCHMARKS:
        bench_data = all_metrics.get(benchmark, {})
        bench_stats = stat_tests.get(benchmark, {})

        for algo in ALGORITHM_ORDER:
            metrics = bench_data.get(algo)
            row = {
                "benchmark": benchmark,
                "algorithm": algo,
                "algorithm_display": ALGO_DISPLAY[algo],
            }
            if metrics is not None:
                row.update({
                    "n_runs": metrics["n_runs"],
                    "mean_violations": round(metrics["mean_viol"], 2),
                    "median_violations": round(metrics["median_viol"], 2),
                    "std_violations": round(metrics["std_viol"], 2),
                    "total_violations": metrics["sum_viol"],
                    "mean_coverage_pct": round(metrics["mean_coverage"] * 100, 1),
                    "mean_best_score": round(metrics["mean_best_score"], 5),
                    "mean_elapsed_min": round(metrics["mean_elapsed_min"], 1) if metrics.get("mean_elapsed_min") else None,
                    "mean_efficiency": round(metrics["mean_efficiency"], 4) if metrics.get("mean_efficiency") else None,
                })
                # Per-req breakdown
                for r, v in metrics["per_req_mean"].items():
                    row[f"mean_{r}"] = round(v, 2)
            else:
                row.update({k: None for k in [
                    "n_runs", "mean_violations", "median_violations",
                    "std_violations", "total_violations",
                    "mean_coverage_pct", "mean_best_score"
                ]})

            # Stats vs SAMOTA
            if algo != "SAMOTA" and algo in bench_stats:
                ts = bench_stats[algo]
                row.update({
                    "mw_p_value": round(ts["p_value"], 4) if not np.isnan(ts["p_value"]) else None,
                    "significant_vs_samota": ts["significant"],
                    "a_ab_samota_vs_algo": round(ts["a_ab"], 3),
                    "effect_size": ts["effect"],
                    "direction": ts["direction"],
                })

            rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = tables_dir / "summary_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved: {summary_path}")

    # Save per-benchmark raw violation data
    for benchmark in BENCHMARKS:
        bench_data = all_metrics.get(benchmark, {})
        all_runs = {}
        for algo in ALGORITHM_ORDER:
            metrics = bench_data.get(algo)
            if metrics is not None:
                all_runs[algo] = metrics["total_viol_per_run"]

        if all_runs:
            max_len = max(len(v) for v in all_runs.values())
            runs_dict = {}
            for algo, vals in all_runs.items():
                padded = list(vals) + [None] * (max_len - len(vals))
                runs_dict[algo] = padded
            runs_df = pd.DataFrame(runs_dict)
            runs_path = tables_dir / f"{benchmark}_violations_per_run.csv"
            runs_df.to_csv(runs_path, index=False)
            print(f"  Saved: {runs_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze all experiment results")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing experiment results (default: results/)")
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS,
                        choices=BENCHMARKS)
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHM_ORDER,
                        choices=ALGORITHM_ORDER)
    parser.add_argument("--nruns", type=int, default=30,
                        help="Expected number of runs (for display only)")
    parser.add_argument("--save_tables", action="store_true",
                        help="Save comparison tables to results/tables/")
    parser.add_argument("--reference", type=str, default="SAMOTA",
                        choices=ALGORITHM_ORDER,
                        help="Reference algorithm for statistical tests (default: SAMOTA)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir.resolve()}")
        sys.exit(1)

    print(f"\nLoading results from: {results_dir.resolve()}")

    # Load all data
    all_metrics = {}
    found_count = 0

    for benchmark in args.benchmarks:
        all_metrics[benchmark] = {}
        for algorithm in args.algorithms:
            reqs_df, score_df, meta_df = load_algorithm_results(
                results_dir, benchmark, algorithm, args.nruns
            )
            if reqs_df is not None:
                metrics = compute_metrics(reqs_df, score_df, benchmark, meta_df)
                all_metrics[benchmark][algorithm] = metrics
                found_count += 1
                timing_str = ""
                if metrics.get("mean_elapsed_min") is not None:
                    timing_str = f", {metrics['mean_elapsed_min']:.0f} min/run"
                print(f"  [OK] {benchmark}/{algorithm}: {metrics['n_runs']} runs, "
                      f"mean {metrics['mean_viol']:.1f} violations/run{timing_str}")
            else:
                print(f"  [MISSING] {benchmark}/{algorithm}")

    if found_count == 0:
        print("\n[ERROR] No result files found. Run experiments first with run_all_experiments.py")
        sys.exit(1)

    # Statistical tests
    stat_tests = run_statistical_tests(all_metrics, reference_algo=args.reference)

    # Print tables
    print_summary_table(all_metrics, stat_tests, args.nruns)
    print_per_req_table(all_metrics)
    print_stat_tests_detail(stat_tests)

    # Save tables
    if args.save_tables:
        tables_dir = results_dir / "tables"
        save_tables(all_metrics, stat_tests, tables_dir, args.nruns)

    print(f"\n[Done] Analyzed {found_count} algorithm/benchmark combinations.")


if __name__ == "__main__":
    main()
