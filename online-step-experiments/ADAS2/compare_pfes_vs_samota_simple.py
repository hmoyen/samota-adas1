#!/usr/bin/env python3
"""
Compare PFES Baseline vs PFES+SAMOTA Hybrid
Runs both algorithms and generates comparison analysis
"""

import subprocess
import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

def run_pfes_baseline(size=30, niterations=30, nruns=1, logdir="results_pfes_baseline"):
    """Run PFES baseline using Click CLI"""
    print(f"\n{'='*80}")
    print(f"RUNNING PFES BASELINE")
    print(f"{'='*80}")

    os.makedirs(logdir, exist_ok=True)

    cmd = [
        "python3", "PFES_falsification.py",
        "--size", str(size),
        "--niterations", str(niterations),
        "--nruns", str(nruns),
        "--optalg", "NSGA3",
        "--logdir", logdir
    ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"✓ PFES baseline completed successfully")
        return True, logdir
    else:
        print(f"✗ PFES baseline failed")
        return False, logdir

def run_pfes_samota(budget=900, niterations=30, logdir="results_pfes_samota"):
    """Run PFES+SAMOTA hybrid"""
    print(f"\n{'='*80}")
    print(f"RUNNING PFES+SAMOTA HYBRID")
    print(f"{'='*80}")

    os.makedirs(logdir, exist_ok=True)

    # Note: PFES_SAMOTA.py doesn't use Click, so we need to modify how we call it
    # For now, we'll just try running it directly
    cmd = ["python3", "PFES_SAMOTA.py"]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"✓ PFES+SAMOTA completed successfully")
        return True, logdir
    else:
        print(f"✗ PFES+SAMOTA failed")
        return False, logdir

def load_results(logdir):
    """Load results from a directory"""
    data = {
        'evaluations': None,
        'violations': None,
        'scores': None,
    }

    # Try to load evaluation data
    f_file = f"{logdir}/F_all_evaluations_NSGA3_0.csv"
    reqs_file = f"{logdir}/Reqs_all_evaluations_NSGA3_0.csv"

    if os.path.exists(f_file):
        data['evaluations'] = pd.read_csv(f_file)

    if os.path.exists(reqs_file):
        data['violations'] = pd.read_csv(reqs_file)

    # Try to load summary data
    score_file = f"{logdir}/score_NSGA3_1.csv"
    if os.path.exists(score_file):
        data['scores'] = pd.read_csv(score_file)

    return data

def analyze_results(pfes_data, samota_data):
    """Generate comparison analysis"""
    print(f"\n{'='*80}")
    print(f"COMPARISON ANALYSIS")
    print(f"{'='*80}\n")

    analysis = {}

    # Count evaluations
    pfes_evals = len(pfes_data['evaluations']) if pfes_data['evaluations'] is not None else 0
    samota_evals = len(samota_data['evaluations']) if samota_data['evaluations'] is not None else 0

    print(f"Total Evaluations:")
    print(f"  PFES:        {pfes_evals}")
    print(f"  PFES+SAMOTA: {samota_evals}")
    analysis['evaluations'] = {'pfes': pfes_evals, 'pfes_samota': samota_evals}

    # Count violations
    if pfes_data['violations'] is not None and samota_data['violations'] is not None:
        pfes_total_viol = pfes_data['violations'].sum().sum()
        samota_total_viol = samota_data['violations'].sum().sum()

        print(f"\nTotal Violations Found:")
        print(f"  PFES:        {int(pfes_total_viol)}")
        print(f"  PFES+SAMOTA: {int(samota_total_viol)}")
        print(f"  Improvement: {((pfes_total_viol - samota_total_viol) / max(pfes_total_viol, 1) * 100):.1f}%")

        analysis['violations'] = {
            'pfes': int(pfes_total_viol),
            'pfes_samota': int(samota_total_viol)
        }

    # Efficiency (violations per evaluation)
    if pfes_evals > 0 and samota_evals > 0:
        pfes_eff = pfes_total_viol / pfes_evals if 'pfes_total_viol' in locals() else 0
        samota_eff = samota_total_viol / samota_evals if 'samota_total_viol' in locals() else 0

        print(f"\nEfficiency (violations/evaluation):")
        print(f"  PFES:        {pfes_eff:.4f}")
        print(f"  PFES+SAMOTA: {samota_eff:.4f}")
        if samota_eff > 0:
            print(f"  Speedup:     {pfes_eff/samota_eff:.2f}x")

        analysis['efficiency'] = {
            'pfes': float(pfes_eff),
            'pfes_samota': float(samota_eff)
        }

    return analysis

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare PFES vs PFES+SAMOTA")
    parser.add_argument("--size", type=int, default=30, help="Population size for PFES")
    parser.add_argument("--niterations", type=int, default=30, help="Iterations for PFES")
    parser.add_argument("--nruns", type=int, default=1, help="Number of PFES runs")
    parser.add_argument("--output", type=str, default="comparison_results", help="Output directory")
    parser.add_argument("--skip-pfes", action="store_true", help="Skip PFES baseline run")
    parser.add_argument("--skip-samota", action="store_true", help="Skip PFES+SAMOTA run")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    pfes_dir = f"{args.output}/pfes_baseline"
    samota_dir = f"{args.output}/pfes_samota"

    # Run PFES baseline
    pfes_success = True
    if not args.skip_pfes:
        pfes_success, pfes_dir = run_pfes_baseline(
            size=args.size,
            niterations=args.niterations,
            nruns=args.nruns,
            logdir=pfes_dir
        )

    # Run PFES+SAMOTA
    samota_success = True
    if not args.skip_samota:
        samota_success, samota_dir = run_pfes_samota(logdir=samota_dir)

    # Load and compare results
    if os.path.exists(pfes_dir) and os.path.exists(samota_dir):
        pfes_data = load_results(pfes_dir)
        samota_data = load_results(samota_dir)

        analysis = analyze_results(pfes_data, samota_data)

        # Save analysis
        with open(f"{args.output}/analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        print(f"\n✓ Results saved to {args.output}/")
    else:
        print(f"\n⚠️  Could not find both result directories")
        if not os.path.exists(pfes_dir):
            print(f"  Missing: {pfes_dir}")
        if not os.path.exists(samota_dir):
            print(f"  Missing: {samota_dir}")

if __name__ == "__main__":
    main()
