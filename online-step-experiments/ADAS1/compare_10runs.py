#!/usr/bin/env python3
"""
Comprehensive comparison of 10 PFES vs 10 PFES+SAMOTA runs
Analyzes evaluations, violations, efficiency, and generates statistical summary
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def load_run_metrics(run_dir):
    """Load metrics from a single run directory"""
    metrics = {
        'evals': None,
        'violations': None,
        'efficiency': None,
        'objectives_covered': None,
    }

    # Try to load evaluation count
    for f in ['F_all_evaluations_NSGA3_0.csv', 'F_all_evaluations_NSGA3_1.csv']:
        fpath = f"{run_dir}/{f}"
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                metrics['evals'] = len(df)
                break
            except Exception as e:
                print(f"  ⚠️  Error reading {f}: {e}")

    # Try to load violations from summary
    for f in ['reqs_NSGA3_1.csv', 'reqs_NSGA3_30.csv']:
        fpath = f"{run_dir}/{f}"
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                if 'conjunction' in df.columns:
                    metrics['violations'] = int(df['conjunction'].iloc[0])
                # Count how many objectives were covered (> 0 = violated, i.e., covered)
                violated_cols = [col for col in df.columns if col.startswith('R')]
                if violated_cols:
                    covered = sum(1 for col in violated_cols if df[col].iloc[0] > 0)
                    metrics['objectives_covered'] = covered
                break
            except Exception as e:
                print(f"  ⚠️  Error reading {f}: {e}")

    # Calculate efficiency
    if metrics['evals'] and metrics['violations']:
        metrics['efficiency'] = metrics['violations'] / metrics['evals']

    return metrics

def load_all_runs(base_dir, num_runs=10):
    """Load all runs from a base directory"""
    results = []

    for run_num in range(1, num_runs + 1):
        run_dir = f"{base_dir}/run_{run_num}"
        if not os.path.exists(run_dir):
            print(f"  ✗ Missing: {run_dir}")
            continue

        metrics = load_run_metrics(run_dir)
        results.append({
            'run': run_num,
            **metrics
        })

        status = "✓" if metrics['evals'] else "✗"
        print(f"  {status} Run {run_num}: {metrics['evals']} evals, {metrics['violations']} violations")

    return results

def compute_statistics(results, label):
    """Compute statistics for a list of runs"""
    df = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print(f"{label} - STATISTICAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total runs: {len(df)}")
    print(f"\nEvaluations:")
    print(f"  Mean:     {df['evals'].mean():.1f}")
    print(f"  Std:      {df['evals'].std():.1f}")
    print(f"  Median:   {df['evals'].median():.1f}")
    print(f"  Min:      {df['evals'].min():.0f}")
    print(f"  Max:      {df['evals'].max():.0f}")

    print(f"\nViolations Found:")
    print(f"  Mean:     {df['violations'].mean():.1f}")
    print(f"  Std:      {df['violations'].std():.1f}")
    print(f"  Median:   {df['violations'].median():.1f}")
    print(f"  Min:      {df['violations'].min():.0f}")
    print(f"  Max:      {df['violations'].max():.0f}")

    print(f"\nEfficiency (violations/eval):")
    print(f"  Mean:     {df['efficiency'].mean():.4f}")
    print(f"  Std:      {df['efficiency'].std():.4f}")
    print(f"  Median:   {df['efficiency'].median():.4f}")
    print(f"  Min:      {df['efficiency'].min():.4f}")
    print(f"  Max:      {df['efficiency'].max():.4f}")

    if 'objectives_covered' in df.columns and df['objectives_covered'].notna().any():
        print(f"\nObjectives Covered (out of 3):")
        print(f"  Mean:     {df['objectives_covered'].mean():.2f}")
        print(f"  Median:   {df['objectives_covered'].median():.1f}")

    return df

def compare_results(pfes_df, samota_df):
    """Compare PFES vs PFES+SAMOTA"""
    print(f"\n{'='*70}")
    print(f"PFES vs PFES+SAMOTA COMPARISON")
    print(f"{'='*70}")

    evals_improvement = ((pfes_df['evals'].mean() - samota_df['evals'].mean()) /
                         pfes_df['evals'].mean() * 100)
    violations_improvement = ((samota_df['violations'].mean() - pfes_df['violations'].mean()) /
                             pfes_df['violations'].mean() * 100)
    efficiency_improvement = ((samota_df['efficiency'].mean() - pfes_df['efficiency'].mean()) /
                             pfes_df['efficiency'].mean() * 100)

    print(f"\nEvaluations used:")
    print(f"  PFES:         {pfes_df['evals'].mean():.1f} ± {pfes_df['evals'].std():.1f}")
    print(f"  PFES+SAMOTA:  {samota_df['evals'].mean():.1f} ± {samota_df['evals'].std():.1f}")
    print(f"  Difference:   {evals_improvement:+.1f}% (PFES uses {evals_improvement:.1f}% more)")

    print(f"\nViolations found:")
    print(f"  PFES:         {pfes_df['violations'].mean():.1f} ± {pfes_df['violations'].std():.1f}")
    print(f"  PFES+SAMOTA:  {samota_df['violations'].mean():.1f} ± {samota_df['violations'].std():.1f}")
    if violations_improvement > 0:
        print(f"  Improvement:  {violations_improvement:+.1f}% (SAMOTA finds {violations_improvement:.1f}% more)")
    else:
        print(f"  Degradation:  {violations_improvement:+.1f}% (SAMOTA finds {abs(violations_improvement):.1f}% fewer)")

    print(f"\nEfficiency (violations per evaluation):")
    print(f"  PFES:         {pfes_df['efficiency'].mean():.4f} ± {pfes_df['efficiency'].std():.4f}")
    print(f"  PFES+SAMOTA:  {samota_df['efficiency'].mean():.4f} ± {samota_df['efficiency'].std():.4f}")
    if efficiency_improvement > 0:
        print(f"  Improvement:  {efficiency_improvement:+.1f}% (SAMOTA is {efficiency_improvement:.1f}% more efficient)")
    else:
        print(f"  Degradation:  {efficiency_improvement:+.1f}% (SAMOTA is {abs(efficiency_improvement):.1f}% less efficient)")

    print(f"\nObjectives covered (mean):")
    if 'objectives_covered' in pfes_df.columns:
        pfes_cov = pfes_df['objectives_covered'].mean()
        samota_cov = samota_df['objectives_covered'].mean()
        print(f"  PFES:         {pfes_cov:.2f}/3")
        print(f"  PFES+SAMOTA:  {samota_cov:.2f}/3")

    # T-test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(pfes_df['efficiency'], samota_df['efficiency'])
    print(f"\nStatistical significance (t-test on efficiency):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.4f}")
    if p_value < 0.05:
        print(f"  Result:      SIGNIFICANT difference (p < 0.05)")
    else:
        print(f"  Result:      No significant difference (p >= 0.05)")

def save_comparison_csv(pfes_df, samota_df, output_file="comparison_summary.csv"):
    """Save comparison to CSV"""
    summary = pd.DataFrame({
        'Metric': [
            'Mean Evaluations',
            'Std Evaluations',
            'Mean Violations',
            'Std Violations',
            'Mean Efficiency',
            'Std Efficiency',
        ],
        'PFES': [
            f"{pfes_df['evals'].mean():.1f}",
            f"{pfes_df['evals'].std():.1f}",
            f"{pfes_df['violations'].mean():.1f}",
            f"{pfes_df['violations'].std():.1f}",
            f"{pfes_df['efficiency'].mean():.4f}",
            f"{pfes_df['efficiency'].std():.4f}",
        ],
        'PFES+SAMOTA': [
            f"{samota_df['evals'].mean():.1f}",
            f"{samota_df['evals'].std():.1f}",
            f"{samota_df['violations'].mean():.1f}",
            f"{samota_df['violations'].std():.1f}",
            f"{samota_df['efficiency'].mean():.4f}",
            f"{samota_df['efficiency'].std():.4f}",
        ]
    })
    summary.to_csv(output_file, index=False)
    print(f"\n✓ Summary saved to {output_file}")

def main():
    print("\n" + "="*70)
    print("COMPARING 10 PFES vs 10 PFES+SAMOTA RUNS")
    print("="*70)

    # Load PFES results
    print("\nLoading PFES baseline runs...")
    pfes_results = load_all_runs("results_10runs_pfes", num_runs=10)

    # Load PFES+SAMOTA results
    print("\nLoading PFES+SAMOTA hybrid runs...")
    samota_results = load_all_runs("results_10runs_samota", num_runs=10)

    # Compute statistics
    pfes_df = compute_statistics(pfes_results, "PFES BASELINE")
    samota_df = compute_statistics(samota_results, "PFES+SAMOTA HYBRID")

    # Compare
    compare_results(pfes_df, samota_df)

    # Save to CSV
    save_comparison_csv(pfes_df, samota_df)

    print("\n" + "="*70)
    print("Comparison complete!")
    print("="*70)

if __name__ == "__main__":
    main()
