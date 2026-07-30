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

    # Binary full-coverage indicator: 1 if every requirement was violated at least
    # once in that run, 0 otherwise.
    full_coverage_per_run = (reqs_df[req_cols] > 0).all(axis=1).values.astype(int)

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
        # Binary full coverage (all requirements violated within the same run)
        "full_coverage_per_run": full_coverage_per_run,
        "full_coverage_rate": float(np.mean(full_coverage_per_run)),
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


def run_statistical_tests(all_metrics: dict, reference_algo: str = "SAMOTA",
                          metric_key: str = "total_viol_per_run"):
    """
    For each benchmark, compare reference_algo against each other algorithm on the
    given per-run metric (default: total_viol_per_run, matching the original
    violation-count-only behavior). Pass metric_key="coverage_per_run" or
    metric_key="full_coverage_per_run" to run the identical MW/A_AB pipeline on
    coverage instead.
    Returns nested dict: {benchmark: {algo: {stat results}}}
    """
    results = {}
    for benchmark, bench_data in all_metrics.items():
        results[benchmark] = {}
        ref_metrics = bench_data.get(reference_algo)
        if ref_metrics is None:
            continue

        ref_vals = ref_metrics[metric_key]

        for algo, metrics in bench_data.items():
            if algo == reference_algo:
                continue
            other_vals = metrics[metric_key]

            stat, p = mann_whitney_test(ref_vals, other_vals)
            a_ab = vargha_delaney(ref_vals, other_vals)

            results[benchmark][algo] = {
                "mw_stat": stat,
                "p_value": p,
                "significant": p < 0.05 if not np.isnan(p) else False,
                "a_ab": a_ab,
                "effect": effect_size_label(a_ab),
                "direction": f"{reference_algo} better" if a_ab > 0.5 else ("equal" if abs(a_ab - 0.5) < 0.06 else "baseline better"),
            }
    return results


def kruskal_wallis_test(all_metrics: dict, benchmark: str, metric_key: str = "total_viol_per_run"):
    """
    Omnibus Kruskal-Wallis H-test across all algorithms with data for one benchmark,
    on the given per-run metric. Meant to be reported before the pairwise
    reference-vs-X comparisons, to justify running pairwise tests at all.
    Returns (stat, p_value, algos_included).
    """
    bench_data = all_metrics.get(benchmark, {})
    groups, algos_included = [], []
    for algo in ALGORITHM_ORDER:
        metrics = bench_data.get(algo)
        if metrics is None:
            continue
        vals = metrics.get(metric_key)
        if vals is None or len(vals) < 2:
            continue
        groups.append(vals)
        algos_included.append(algo)

    if len(groups) < 2:
        return float("nan"), float("nan"), algos_included, "too_few_algorithms"
    try:
        stat, p = stats.kruskal(*groups)
        return float(stat), float(p), algos_included, None
    except ValueError as e:
        # e.g. "All numbers are identical in kruskal" — every algorithm scored
        # identically (no variance), which is a real result, not missing data.
        return float("nan"), float("nan"), algos_included, str(e)


def holm_bonferroni_correction(p_values: list) -> list:
    """
    Holm-Bonferroni step-down correction for a family of p-values. Returns corrected
    p-values in the same order as the input list. NaN entries (e.g. a comparison with
    too few runs) are passed through as NaN and excluded from the family size m used
    for correction.
    """
    n = len(p_values)
    valid = [(i, p) for i, p in enumerate(p_values) if not np.isnan(p)]
    valid.sort(key=lambda t: t[1])
    m = len(valid)

    corrected = [float("nan")] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(valid):
        adj = (m - rank) * p
        running_max = max(running_max, adj)
        corrected[orig_idx] = min(running_max, 1.0)
    return corrected


def bootstrap_ci(values: np.ndarray, statistic=np.mean, n_resamples: int = 10000,
                  ci: float = 0.95, seed: int = 0):
    """Percentile bootstrap CI for a summary statistic of a single sample."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        boot_stats[i] = statistic(rng.choice(values, size=n, replace=True))
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_stats, alpha * 100))
    hi = float(np.percentile(boot_stats, (1 - alpha) * 100))
    return (lo, hi)


def bootstrap_ci_a12(a: np.ndarray, b: np.ndarray, n_resamples: int = 10000,
                      ci: float = 0.95, seed: int = 0):
    """
    Percentile bootstrap CI for the Vargha-Delaney A_12 effect size, resampling both
    groups independently each iteration. Reuses vargha_delaney() itself for every
    resample (not a reimplementation), so the CI is always consistent with whatever
    point estimate is reported alongside it.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot_vals = np.empty(n_resamples)
    for i in range(n_resamples):
        a_s = rng.choice(a, size=m, replace=True)
        b_s = rng.choice(b, size=n, replace=True)
        boot_vals[i] = vargha_delaney(a_s, b_s)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_vals, alpha * 100))
    hi = float(np.percentile(boot_vals, (1 - alpha) * 100))
    return (lo, hi)


def enrich_with_corrections_and_cis(stat_tests: dict, all_metrics: dict, metric_key: str,
                                     reference_algo: str, n_resamples: int = 10000,
                                     seed: int = 0):
    """
    In place: adds Holm-Bonferroni corrected p-values (p_value_holm,
    significant_holm) and bootstrap 95% CIs (ref_mean_ci, ref_median_ci,
    other_mean_ci, other_median_ci, a_ab_ci) to the output of run_statistical_tests().
    Raw p_value/a_ab fields are left untouched. Correction is applied within each
    benchmark's family of reference_algo-vs-X comparisons for this metric.
    """
    for benchmark, bench_stats in stat_tests.items():
        algos = list(bench_stats.keys())
        if not algos:
            continue
        raw_p = [bench_stats[a]["p_value"] for a in algos]
        corrected_p = holm_bonferroni_correction(raw_p)

        ref_vals = all_metrics[benchmark][reference_algo][metric_key]
        for algo, p_holm in zip(algos, corrected_p):
            ts = bench_stats[algo]
            ts["p_value_holm"] = p_holm
            ts["significant_holm"] = (p_holm < 0.05) if not np.isnan(p_holm) else False

            other_vals = all_metrics[benchmark][algo][metric_key]
            ts["ref_mean_ci"] = bootstrap_ci(ref_vals, np.mean, n_resamples, seed=seed)
            ts["ref_median_ci"] = bootstrap_ci(ref_vals, np.median, n_resamples, seed=seed)
            ts["other_mean_ci"] = bootstrap_ci(other_vals, np.mean, n_resamples, seed=seed)
            ts["other_median_ci"] = bootstrap_ci(other_vals, np.median, n_resamples, seed=seed)
            ts["a_ab_ci"] = bootstrap_ci_a12(ref_vals, other_vals, n_resamples, seed=seed)
    return stat_tests


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


def print_kruskal_wallis_summary(all_metrics: dict, metric_key: str, metric_label: str):
    """Print the per-benchmark omnibus Kruskal-Wallis result, before pairwise tests."""
    print(f"\n{'-'*80}")
    print(f"OMNIBUS: Kruskal-Wallis H-test across all algorithms — {metric_label}")
    print(f"  (run before the pairwise reference-vs-X comparisons below, to justify")
    print(f"   doing pairwise tests at all)")
    print(f"{'-'*80}")
    for benchmark in BENCHMARKS:
        stat, p, algos_included, reason = kruskal_wallis_test(all_metrics, benchmark, metric_key)
        if np.isnan(p):
            if reason == "too_few_algorithms":
                print(f"  {benchmark}: insufficient data (< 2 algorithms with results)")
            else:
                print(f"  {benchmark}: omnibus test undefined — {len(algos_included)} "
                      f"algorithms present, but all had identical values ({reason})")
            continue
        sig = "YES *" if p < 0.05 else "no"
        print(f"  {benchmark}: H={stat:.3f}, p={p:.4f} ({sig}), "
              f"{len(algos_included)} algorithms: {', '.join(algos_included)}")


def print_stat_tests_detail(stat_tests: dict, metric_label: str = "Violation Count",
                            reference_algo: str = "SAMOTA"):
    """Print detailed statistical test results, including Holm-corrected p-values
    and bootstrap 95% CIs alongside the raw values (raw values are never overwritten)."""
    print(f"\n{'='*80}")
    print(f"STATISTICAL TESTS: {reference_algo} vs Baselines — {metric_label}")
    print(f"  (Mann-Whitney U, Vargha-Delaney A_AB, Holm-Bonferroni-corrected p-value,")
    print(f"   bootstrap 95% CIs on A_AB and on each side's mean)")
    print(f"  * = p < 0.05 (statistically significant), raw and Holm-corrected shown separately")
    print(f"  A_AB > 0.5 means {reference_algo} scores higher on this metric")
    print(f"{'='*80}")

    for benchmark in BENCHMARKS:
        bench_stats = stat_tests.get(benchmark, {})
        if not bench_stats:
            print(f"\n  {benchmark}: no data")
            continue

        print(f"\n  {benchmark}:")
        print(f"    {'vs':12} | {'p-raw':^8} | {'p-holm':^8} | {'Sig(raw)':^9} | {'Sig(holm)':^10} "
              f"| {'A_AB':^6} | {'A_AB 95% CI':^16} | {'Effect':^10} | Direction")
        print("    " + "-" * 115)
        for algo in ALGORITHM_ORDER:
            if algo == reference_algo:
                continue
            ts = bench_stats.get(algo)
            if ts is None:
                print(f"    {ALGO_DISPLAY[algo]:12} | {'N/A':^8} | {'N/A':^8} | {'N/A':^9} | {'N/A':^10} "
                      f"| {'N/A':^6} | {'N/A':^16} | {'N/A':^10} | N/A")
                continue
            sig_raw = "YES*" if ts["significant"] else "no"
            p_holm = ts.get("p_value_holm", float("nan"))
            p_holm_str = f"{p_holm:.4f}" if not np.isnan(p_holm) else "N/A"
            sig_holm = "YES*" if ts.get("significant_holm") else "no"
            a_ci = ts.get("a_ab_ci", (float("nan"), float("nan")))
            a_ci_str = f"[{a_ci[0]:.2f},{a_ci[1]:.2f}]" if not np.isnan(a_ci[0]) else "N/A"
            print(f"    {ALGO_DISPLAY[algo]:12} | {ts['p_value']:^8.4f} | {p_holm_str:^8} | {sig_raw:^9} "
                  f"| {sig_holm:^10} | {ts['a_ab']:^6.3f} | {a_ci_str:^16} | {ts['effect']:^10} | {ts['direction']}")


# ============================================================================
# SAVE TABLES
# ============================================================================

def save_tables(all_metrics: dict, all_stat_tests: dict, tables_dir: Path, nruns: int,
                reference_algo: str = "SAMOTA"):
    """
    Save summary tables to CSV files.

    all_stat_tests: {metric_key: stat_tests_dict} — one entry per metric tested
    (total_viol_per_run, coverage_per_run, full_coverage_per_run), each already
    enriched with Holm-corrected p-values and bootstrap CIs via
    enrich_with_corrections_and_cis().
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    viol_stats = all_stat_tests.get("total_viol_per_run", {})
    cov_stats = all_stat_tests.get("coverage_per_run", {})
    fullcov_stats = all_stat_tests.get("full_coverage_per_run", {})

    rows = []
    for benchmark in BENCHMARKS:
        bench_data = all_metrics.get(benchmark, {})
        bench_viol_stats = viol_stats.get(benchmark, {})
        bench_cov_stats = cov_stats.get(benchmark, {})
        bench_fullcov_stats = fullcov_stats.get(benchmark, {})

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
                    "full_coverage_rate_pct": round(metrics["full_coverage_rate"] * 100, 1),
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
                    "mean_coverage_pct", "full_coverage_rate_pct", "mean_best_score"
                ]})

            # Stats vs reference_algo — violation count (kept for backward compatibility
            # with the original column names), plus coverage and full-coverage in
            # parallel, each clearly prefixed so the three are never confused.
            def _add_stat_cols(prefix, ts):
                if ts is None:
                    return {}
                a_ci = ts.get("a_ab_ci", (float("nan"), float("nan")))
                return {
                    f"{prefix}_mw_p_value_raw": round(ts["p_value"], 4) if not np.isnan(ts["p_value"]) else None,
                    f"{prefix}_mw_p_value_holm": round(ts["p_value_holm"], 4) if not np.isnan(ts.get("p_value_holm", float("nan"))) else None,
                    f"{prefix}_significant_raw": ts["significant"],
                    f"{prefix}_significant_holm": ts.get("significant_holm"),
                    f"{prefix}_a_ab": round(ts["a_ab"], 3),
                    f"{prefix}_a_ab_ci_lo": round(a_ci[0], 3) if not np.isnan(a_ci[0]) else None,
                    f"{prefix}_a_ab_ci_hi": round(a_ci[1], 3) if not np.isnan(a_ci[1]) else None,
                    f"{prefix}_effect_size": ts["effect"],
                    f"{prefix}_direction": ts["direction"],
                }

            if algo != reference_algo:
                row.update(_add_stat_cols("viol", bench_viol_stats.get(algo)))
                row.update(_add_stat_cols("coverage", bench_cov_stats.get(algo)))
                row.update(_add_stat_cols("full_coverage", bench_fullcov_stats.get(algo)))

            rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = tables_dir / "summary_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved: {summary_path}")

    # Save per-benchmark raw per-run data, one file per metric
    per_run_metrics = {
        "violations": "total_viol_per_run",
        "coverage": "coverage_per_run",
        "full_coverage": "full_coverage_per_run",
    }
    for benchmark in BENCHMARKS:
        bench_data = all_metrics.get(benchmark, {})
        for file_suffix, metric_key in per_run_metrics.items():
            all_runs = {}
            for algo in ALGORITHM_ORDER:
                metrics = bench_data.get(algo)
                if metrics is not None:
                    all_runs[algo] = metrics[metric_key]

            if all_runs:
                max_len = max(len(v) for v in all_runs.values())
                runs_dict = {}
                for algo, vals in all_runs.items():
                    padded = list(vals) + [None] * (max_len - len(vals))
                    runs_dict[algo] = padded
                runs_df = pd.DataFrame(runs_dict)
                runs_path = tables_dir / f"{benchmark}_{file_suffix}_per_run.csv"
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
    parser.add_argument("--bootstrap_resamples", type=int, default=10000,
                        help="Number of bootstrap resamples for CIs (default: 10000)")
    parser.add_argument("--bootstrap_seed", type=int, default=0,
                        help="RNG seed for bootstrap resampling, for reproducible CIs (default: 0)")
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

    # Statistical tests — run the identical MW/Vargha-Delaney pipeline on violation
    # count (original behavior), requirement coverage, and binary full coverage.
    # Coverage is the paper's actual headline claim; violation count alone was
    # previously the only metric tested. Each is enriched with Holm-Bonferroni
    # correction and bootstrap CIs, without altering the raw values.
    metrics_to_test = [
        ("total_viol_per_run", "Violation Count"),
        ("coverage_per_run", "Requirement Coverage (fraction of reqs ever violated)"),
        ("full_coverage_per_run", "Full Coverage (binary: all reqs violated in-run)"),
    ]

    all_stat_tests = {}
    for metric_key, _ in metrics_to_test:
        st = run_statistical_tests(all_metrics, reference_algo=args.reference, metric_key=metric_key)
        enrich_with_corrections_and_cis(
            st, all_metrics, metric_key, args.reference,
            n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed,
        )
        all_stat_tests[metric_key] = st

    stat_tests = all_stat_tests["total_viol_per_run"]  # kept for the existing summary table

    # Print tables
    print_summary_table(all_metrics, stat_tests, args.nruns)
    print_per_req_table(all_metrics)

    for metric_key, metric_label in metrics_to_test:
        print_kruskal_wallis_summary(all_metrics, metric_key, metric_label)
        print_stat_tests_detail(all_stat_tests[metric_key], metric_label=metric_label,
                                reference_algo=args.reference)

    # Save tables
    if args.save_tables:
        tables_dir = results_dir / "tables"
        save_tables(all_metrics, all_stat_tests, tables_dir, args.nruns, reference_algo=args.reference)

    print(f"\n[Done] Analyzed {found_count} algorithm/benchmark combinations.")


if __name__ == "__main__":
    main()
