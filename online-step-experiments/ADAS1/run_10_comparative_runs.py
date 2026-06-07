#!/usr/bin/env python3
"""
Run PFES and PFES+SAMOTA 10 times each for proper statistical comparison
"""

import subprocess
import os
import shutil
import json
from pathlib import Path

def run_pfes_baseline(run_num, size=30, niterations=30, output_base="results_10runs_pfes"):
    """Run PFES baseline once"""
    os.makedirs(output_base, exist_ok=True)

    logdir = f"{output_base}/run_{run_num}"

    cmd = [
        "python3", "PFES_falsification.py",
        "--size", str(size),
        "--niterations", str(niterations),
        "--nruns", "1",
        "--optalg", "NSGA3",
        "--logdir", logdir
    ]

    print(f"\n{'='*70}")
    print(f"PFES RUN {run_num}/10")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    return result.returncode == 0, logdir

def run_pfes_samota(run_num, output_base="results_10runs_samota"):
    """Run PFES+SAMOTA once"""
    os.makedirs(output_base, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PFES+SAMOTA RUN {run_num}/10")
    print(f"{'='*70}")
    print(f"Command: python3 PFES_SAMOTA.py\n")

    # Run PFES_SAMOTA
    result = subprocess.run(["python3", "PFES_SAMOTA.py"])

    if result.returncode == 0:
        # Move results to run directory
        run_dir = f"{output_base}/run_{run_num}"

        # Check where PFES_SAMOTA saves results
        if os.path.exists("pfes_samota_baseline"):
            shutil.move("pfes_samota_baseline", run_dir)
            return True, run_dir
        else:
            print(f"⚠️  Could not find PFES+SAMOTA output directory")
            return False, None

    return False, None

def load_run_summary(run_dir):
    """Load summary from a single run"""
    import pandas as pd

    summary = {
        'evals': None,
        'violations': None,
        'efficiency': None,
    }

    # Try to load evaluation count
    for f in ['F_all_evaluations_NSGA3_0.csv', 'F_all_evaluations_NSGA3_1.csv']:
        if os.path.exists(f"{run_dir}/{f}"):
            df = pd.read_csv(f"{run_dir}/{f}")
            summary['evals'] = len(df)
            break

    # Try to load violations from summary
    for f in ['reqs_NSGA3_1.csv', 'reqs_NSGA3_30.csv']:
        if os.path.exists(f"{run_dir}/{f}"):
            df = pd.read_csv(f"{run_dir}/{f}")
            if 'conjunction' in df.columns:
                summary['violations'] = int(df['conjunction'].iloc[0])
            break

    # Calculate efficiency
    if summary['evals'] and summary['violations']:
        summary['efficiency'] = summary['violations'] / summary['evals']

    return summary

def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--size", type=int, default=30, help="PFES population size")
    parser.add_argument("--niterations", type=int, default=30, help="PFES iterations")
    args = parser.parse_args()

    pfes_output = "results_10runs_pfes"
    samota_output = "results_10runs_samota"

    # Run PFES baseline multiple times
    print(f"\n{'='*70}")
    print(f"RUNNING PFES BASELINE - {args.runs} RUNS")
    print(f"{'='*70}")

    pfes_results = []
    for run in range(1, args.runs + 1):
        success, logdir = run_pfes_baseline(run, args.size, args.niterations, pfes_output)
        if success:
            summary = load_run_summary(logdir)
            pfes_results.append({
                'run': run,
                **summary
            })
            print(f"✓ PFES run {run} complete: {summary['evals']} evals, {summary['violations']} violations")
        else:
            print(f"✗ PFES run {run} failed")

    # Run PFES+SAMOTA multiple times
    print(f"\n{'='*70}")
    print(f"RUNNING PFES+SAMOTA HYBRID - {args.runs} RUNS")
    print(f"{'='*70}")

    samota_results = []
    for run in range(1, args.runs + 1):
        success, logdir = run_pfes_samota(run, samota_output)
        if success:
            summary = load_run_summary(logdir)
            samota_results.append({
                'run': run,
                **summary
            })
            print(f"✓ PFES+SAMOTA run {run} complete: {summary['evals']} evals, {summary['violations']} violations")
        else:
            print(f"✗ PFES+SAMOTA run {run} failed")

    # Save results
    if pfes_results:
        pfes_df = pd.DataFrame(pfes_results)
        pfes_df.to_csv(f"{pfes_output}/summary.csv", index=False)

    if samota_results:
        samota_df = pd.DataFrame(samota_results)
        samota_df.to_csv(f"{samota_output}/summary.csv", index=False)

    # Display comparison
    print(f"\n{'='*70}")
    print(f"SUMMARY - 10 RUNS COMPARISON")
    print(f"{'='*70}\n")

    if pfes_results:
        pfes_df = pd.DataFrame(pfes_results)
        print("PFES BASELINE:")
        print(f"  Evaluations: {pfes_df['evals'].mean():.0f} ± {pfes_df['evals'].std():.0f}")
        print(f"  Violations:  {pfes_df['violations'].mean():.1f} ± {pfes_df['violations'].std():.1f}")
        print(f"  Efficiency:  {pfes_df['efficiency'].mean():.4f} ± {pfes_df['efficiency'].std():.4f}")

    if samota_results:
        samota_df = pd.DataFrame(samota_results)
        print("\nPFES+SAMOTA HYBRID:")
        print(f"  Evaluations: {samota_df['evals'].mean():.0f} ± {samota_df['evals'].std():.0f}")
        print(f"  Violations:  {samota_df['violations'].mean():.1f} ± {samota_df['violations'].std():.1f}")
        print(f"  Efficiency:  {samota_df['efficiency'].mean():.4f} ± {samota_df['efficiency'].std():.4f}")

    if pfes_results and samota_results:
        print(f"\nCOMPARISON:")
        avg_pfes_evals = pfes_df['evals'].mean()
        avg_samota_evals = samota_df['evals'].mean()
        eval_savings = (avg_pfes_evals - avg_samota_evals) / avg_pfes_evals * 100
        print(f"  Evaluation savings: {eval_savings:.1f}%")

    print(f"\n✓ Results saved to:")
    print(f"  - {pfes_output}/")
    print(f"  - {samota_output}/")

if __name__ == "__main__":
    main()
