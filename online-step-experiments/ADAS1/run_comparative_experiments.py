#!/usr/bin/env python3
"""
Comparative Experiment Framework: PFES vs PFES+SAMOTA
Author: Helena Moyen
Date: 2026-05-20

PURPOSE:
  Run both PFES (baseline) and PFES+SAMOTA (hybrid) approaches on identical budgets
  and compare:
  - Total violations found (R0, R1, R2 breakdown)
  - Objectives covered
  - Time elapsed
  - Efficiency (violations/evaluation)
  - Surrogate effectiveness (PFES+SAMOTA only)

USAGE:
  python run_comparative_experiments.py --runs 5 --budget 900 --output results_comparison

OUTPUT:
  results_comparison/
  ├── summary.csv                  # Overall comparison
  ├── pfes_runs_<1-N>.csv         # Per-run PFES results
  ├── pfes_samota_runs_<1-N>.csv  # Per-run PFES+SAMOTA results
  ├── comparison_report.txt       # Formatted summary
  └── efficiency_analysis.txt     # Statistical analysis
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import PFES_falsification
import PFES_SAMOTA


class ComparativeExperiment:
    """Run and compare PFES vs PFES+SAMOTA"""

    def __init__(self, n_runs=5, budget=900, output_dir='results_comparison'):
        self.n_runs = n_runs
        self.budget = budget
        self.output_dir = output_dir
        self.pfes_results = []
        self.pfes_samota_results = []

        os.makedirs(output_dir, exist_ok=True)

    def run_pfes(self, run_id):
        """Run pure PFES baseline"""
        print(f"\n{'='*80}")
        print(f"RUN {run_id+1}/{self.n_runs} - PFES (Baseline)")
        print(f"{'='*80}")

        try:
            result = PFES_falsification.run_pfes(
                max_evaluations=self.budget,
                seed=run_id
            )

            # Normalize result format
            normalized = {
                'run_id': run_id + 1,
                'approach': 'PFES',
                'time_seconds': result.get('elapsed', 0),
                'evaluations': result.get('eval_count', 0),
                'total_violations': result.get('violations', 0),
                'r0_violations': result.get('unsatisfied_reqs', [0, 0, 0])[0],
                'r1_violations': result.get('unsatisfied_reqs', [0, 0, 0])[1],
                'r2_violations': result.get('unsatisfied_reqs', [0, 0, 0])[2],
                'objectives_covered': result.get('objectives_covered', 0),
                'archive_size': result.get('archive_size', 0),
                'efficiency': result.get('violations', 0) / max(result.get('eval_count', 1), 1)
            }

            print(f"✓ PFES Result: {normalized['total_violations']} violations, {normalized['time_seconds']:.1f}s")
            return normalized

        except Exception as e:
            print(f"✗ PFES Run {run_id+1} failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_pfes_samota(self, run_id):
        """Run PFES+SAMOTA hybrid approach"""
        print(f"\n{'='*80}")
        print(f"RUN {run_id+1}/{self.n_runs} - PFES+SAMOTA (Hybrid)")
        print(f"{'='*80}")

        try:
            result = PFES_SAMOTA.run_pfes_samota(
                budget=self.budget,
                max_iterations=30,
                max_time_seconds=86400  # 24 hours
            )

            # Normalize result format
            normalized = {
                'run_id': run_id + 1,
                'approach': 'PFES+SAMOTA',
                'time_seconds': result.get('elapsed', 0),
                'evaluations': result.get('eval_count', 0),
                'total_violations': result.get('violations', 0),
                'r0_violations': result.get('unsatisfied_reqs', [0, 0, 0])[0],
                'r1_violations': result.get('unsatisfied_reqs', [0, 0, 0])[1],
                'r2_violations': result.get('unsatisfied_reqs', [0, 0, 0])[2],
                'objectives_covered': result.get('objectives_covered', 0),
                'archive_size': result.get('archive_size', 0),
                'efficiency': result.get('violations', 0) / max(result.get('eval_count', 1), 1)
            }

            print(f"✓ PFES+SAMOTA Result: {normalized['total_violations']} violations, {normalized['time_seconds']:.1f}s")
            return normalized

        except Exception as e:
            print(f"✗ PFES+SAMOTA Run {run_id+1} failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_all_experiments(self):
        """Run all comparative experiments"""
        print(f"\nStarting Comparative Experiments")
        print(f"  Runs: {self.n_runs}")
        print(f"  Budget per run: {self.budget} evaluations")
        print(f"  Output: {self.output_dir}")

        for run_id in range(self.n_runs):
            # Run PFES
            pfes_result = self.run_pfes(run_id)
            if pfes_result:
                self.pfes_results.append(pfes_result)

            # Run PFES+SAMOTA
            samota_result = self.run_pfes_samota(run_id)
            if samota_result:
                self.pfes_samota_results.append(samota_result)

        # Generate reports
        self.generate_reports()

    def generate_reports(self):
        """Generate comparison reports"""
        print(f"\n{'='*80}")
        print("GENERATING REPORTS")
        print(f"{'='*80}")

        # Save individual run results
        if self.pfes_results:
            df_pfes = pd.DataFrame(self.pfes_results)
            csv_path = os.path.join(self.output_dir, 'pfes_runs.csv')
            df_pfes.to_csv(csv_path, index=False)
            print(f"✓ Saved PFES runs to {csv_path}")

        if self.pfes_samota_results:
            df_samota = pd.DataFrame(self.pfes_samota_results)
            csv_path = os.path.join(self.output_dir, 'pfes_samota_runs.csv')
            df_samota.to_csv(csv_path, index=False)
            print(f"✓ Saved PFES+SAMOTA runs to {csv_path}")

        # Generate comparison summary
        if self.pfes_results and self.pfes_samota_results:
            self._generate_comparison_summary()
            self._generate_efficiency_analysis()

        print(f"\n✓ All reports generated in {self.output_dir}/")

    def _generate_comparison_summary(self):
        """Generate detailed comparison summary"""
        df_pfes = pd.DataFrame(self.pfes_results)
        df_samota = pd.DataFrame(self.pfes_samota_results)

        # Calculate statistics
        stats = {
            'PFES': self._calculate_stats(df_pfes),
            'PFES+SAMOTA': self._calculate_stats(df_samota)
        }

        # Write summary
        summary_path = os.path.join(self.output_dir, 'summary.csv')

        with open(summary_path, 'w') as f:
            f.write("Comparative Experiment Summary\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Budget per run: {self.budget} evaluations\n")
            f.write(f"Runs: {self.n_runs}\n\n")

            f.write("METRIC,PFES (Mean),PFES (Std),PFES+SAMOTA (Mean),PFES+SAMOTA (Std),IMPROVEMENT\n")

            metrics = {
                'Total Violations': 'total_violations',
                'R0 Violations': 'r0_violations',
                'R1 Violations': 'r1_violations',
                'R2 Violations': 'r2_violations',
                'Objectives Covered': 'objectives_covered',
                'Time (seconds)': 'time_seconds',
                'Efficiency (v/e)': 'efficiency'
            }

            for metric_name, metric_key in metrics.items():
                pfes_mean = stats['PFES'][metric_key]['mean']
                pfes_std = stats['PFES'][metric_key]['std']
                samota_mean = stats['PFES+SAMOTA'][metric_key]['mean']
                samota_std = stats['PFES+SAMOTA'][metric_key]['std']

                # Calculate improvement (positive = better for violations, negative = worse)
                if metric_key in ['time_seconds']:
                    # For time, lower is better
                    improvement = ((pfes_mean - samota_mean) / pfes_mean * 100) if pfes_mean > 0 else 0
                else:
                    # For violations, higher is better
                    improvement = ((samota_mean - pfes_mean) / pfes_mean * 100) if pfes_mean > 0 else 0

                f.write(f"{metric_name},{pfes_mean:.2f},{pfes_std:.2f},"
                       f"{samota_mean:.2f},{samota_std:.2f},{improvement:+.1f}%\n")

        print(f"✓ Saved comparison summary to {summary_path}")

        # Also print to console
        self._print_summary_table(stats)

    def _generate_efficiency_analysis(self):
        """Generate detailed efficiency analysis"""
        df_pfes = pd.DataFrame(self.pfes_results)
        df_samota = pd.DataFrame(self.pfes_samota_results)

        analysis_path = os.path.join(self.output_dir, 'efficiency_analysis.txt')

        with open(analysis_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EFFICIENCY ANALYSIS: PFES vs PFES+SAMOTA\n")
            f.write("="*80 + "\n\n")

            f.write("VIOLATIONS PER EVALUATION (Efficiency Metric)\n")
            f.write("-"*80 + "\n")
            pfes_eff = df_pfes['efficiency'].values
            samota_eff = df_samota['efficiency'].values

            f.write(f"PFES:\n")
            f.write(f"  Mean: {np.mean(pfes_eff):.4f} violations/eval\n")
            f.write(f"  Std:  {np.std(pfes_eff):.4f}\n")
            f.write(f"  Min:  {np.min(pfes_eff):.4f}\n")
            f.write(f"  Max:  {np.max(pfes_eff):.4f}\n\n")

            f.write(f"PFES+SAMOTA:\n")
            f.write(f"  Mean: {np.mean(samota_eff):.4f} violations/eval\n")
            f.write(f"  Std:  {np.std(samota_eff):.4f}\n")
            f.write(f"  Min:  {np.min(samota_eff):.4f}\n")
            f.write(f"  Max:  {np.max(samota_eff):.4f}\n\n")

            improvement = ((np.mean(samota_eff) - np.mean(pfes_eff)) / np.mean(pfes_eff) * 100)
            f.write(f"Efficiency Improvement: {improvement:+.1f}%\n\n")

            f.write("VIOLATIONS BREAKDOWN (R0, R1, R2)\n")
            f.write("-"*80 + "\n")
            f.write(f"PFES:\n")
            f.write(f"  R0: {df_pfes['r0_violations'].mean():.1f} ± {df_pfes['r0_violations'].std():.1f}\n")
            f.write(f"  R1: {df_pfes['r1_violations'].mean():.1f} ± {df_pfes['r1_violations'].std():.1f}\n")
            f.write(f"  R2: {df_pfes['r2_violations'].mean():.1f} ± {df_pfes['r2_violations'].std():.1f}\n\n")

            f.write(f"PFES+SAMOTA:\n")
            f.write(f"  R0: {df_samota['r0_violations'].mean():.1f} ± {df_samota['r0_violations'].std():.1f}\n")
            f.write(f"  R1: {df_samota['r1_violations'].mean():.1f} ± {df_samota['r1_violations'].std():.1f}\n")
            f.write(f"  R2: {df_samota['r2_violations'].mean():.1f} ± {df_samota['r2_violations'].std():.1f}\n\n")

            f.write("TIME COMPARISON\n")
            f.write("-"*80 + "\n")
            f.write(f"PFES:        {df_pfes['time_seconds'].mean():.1f}s ± {df_pfes['time_seconds'].std():.1f}s\n")
            f.write(f"PFES+SAMOTA: {df_samota['time_seconds'].mean():.1f}s ± {df_samota['time_seconds'].std():.1f}s\n")
            time_ratio = df_samota['time_seconds'].mean() / df_pfes['time_seconds'].mean()
            f.write(f"Ratio: {time_ratio:.2f}x\n")

        print(f"✓ Saved efficiency analysis to {analysis_path}")

    def _calculate_stats(self, df):
        """Calculate statistics for a results dataframe"""
        stats = {}
        for col in df.columns:
            if col not in ['run_id', 'approach']:
                try:
                    values = pd.to_numeric(df[col], errors='coerce').dropna()
                    stats[col] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max())
                    }
                except:
                    pass
        return stats

    def _print_summary_table(self, stats):
        """Print summary table to console"""
        print("\n" + "="*100)
        print("COMPARATIVE RESULTS SUMMARY")
        print("="*100)
        print(f"{'Metric':<25} {'PFES (Mean)':<20} {'PFES+SAMOTA (Mean)':<20} {'Improvement':<15}")
        print("-"*100)

        metrics = {
            'Total Violations': 'total_violations',
            'R0 Violations': 'r0_violations',
            'R1 Violations': 'r1_violations',
            'R2 Violations': 'r2_violations',
            'Objectives Covered': 'objectives_covered',
            'Time (seconds)': 'time_seconds',
            'Efficiency (v/e)': 'efficiency'
        }

        for metric_name, metric_key in metrics.items():
            if metric_key in stats['PFES'] and metric_key in stats['PFES+SAMOTA']:
                pfes_mean = stats['PFES'][metric_key]['mean']
                samota_mean = stats['PFES+SAMOTA'][metric_key]['mean']

                if metric_key in ['time_seconds']:
                    improvement = ((pfes_mean - samota_mean) / pfes_mean * 100) if pfes_mean > 0 else 0
                else:
                    improvement = ((samota_mean - pfes_mean) / pfes_mean * 100) if pfes_mean > 0 else 0

                print(f"{metric_name:<25} {pfes_mean:>15.2f}      {samota_mean:>15.2f}      {improvement:>10.1f}%")

        print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run comparative experiments: PFES vs PFES+SAMOTA'
    )
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of runs per approach (default: 5)')
    parser.add_argument('--budget', type=int, default=900,
                       help='Evaluation budget per run (default: 900)')
    parser.add_argument('--output', type=str, default='results_comparison',
                       help='Output directory for results (default: results_comparison)')

    args = parser.parse_args()

    exp = ComparativeExperiment(
        n_runs=args.runs,
        budget=args.budget,
        output_dir=args.output
    )

    exp.run_all_experiments()

    print(f"\n✓ Experiments complete! Results saved to {args.output}/")
    print(f"  - summary.csv: Overall comparison")
    print(f"  - pfes_runs.csv: PFES per-run results")
    print(f"  - pfes_samota_runs.csv: PFES+SAMOTA per-run results")
    print(f"  - efficiency_analysis.txt: Detailed analysis")


if __name__ == '__main__':
    main()
