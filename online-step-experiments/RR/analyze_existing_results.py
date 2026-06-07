#!/usr/bin/env python3
"""
Compare existing PFES vs PFES+SAMOTA results from CSV files
Handles missing files gracefully
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_results(directory):
    """Load CSV results from a directory"""
    results = {
        'F_all': None,
        'reqs_all': None,
        'score_summary': None,
        'reqs_summary': None,
        'X_all': None,
    }

    # Try different filename patterns
    f_files = [
        f'{directory}/F_all_evaluations_NSGA3_0.csv',
        f'{directory}/F_all_evaluations_NSGA3_1.csv',
    ]

    reqs_files = [
        f'{directory}/Reqs_all_evaluations_NSGA3_0.csv',  # Capital R
        f'{directory}/reqs_all_evaluations_NSGA3_0.csv',  # Lowercase r
        f'{directory}/Reqs_all_evaluations_NSGA3_1.csv',
        f'{directory}/reqs_all_evaluations_NSGA3_1.csv',
    ]

    score_files = [
        f'{directory}/score_NSGA3_1.csv',
        f'{directory}/score_NSGA3_30.csv',
    ]

    reqs_summary_files = [
        f'{directory}/reqs_NSGA3_1.csv',
        f'{directory}/reqs_NSGA3_30.csv',
    ]

    x_files = [
        f'{directory}/X_all_evaluations_NSGA3_0.csv',
        f'{directory}/X_all_evaluations_NSGA3_1.csv',
    ]

    # Load F (fitness scores)
    for f_file in f_files:
        if os.path.exists(f_file):
            results['F_all'] = pd.read_csv(f_file)
            break

    # Load Reqs (requirements/violations) - detailed per-evaluation
    for reqs_file in reqs_files:
        if os.path.exists(reqs_file):
            results['reqs_all'] = pd.read_csv(reqs_file)
            break

    # Load score summary
    for score_file in score_files:
        if os.path.exists(score_file):
            results['score_summary'] = pd.read_csv(score_file)
            break

    # Load reqs summary
    for reqs_summary_file in reqs_summary_files:
        if os.path.exists(reqs_summary_file):
            results['reqs_summary'] = pd.read_csv(reqs_summary_file)
            break

    # Load X (test cases)
    for x_file in x_files:
        if os.path.exists(x_file):
            results['X_all'] = pd.read_csv(x_file)
            break

    return results

def analyze_results(name, results):
    """Analyze a single result set"""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")

    analysis = {}

    # Count evaluations
    if results['F_all'] is not None:
        n_evals = len(results['F_all'])
        print(f"\n📊 Total Evaluations: {n_evals}")
        analysis['evaluations'] = n_evals
    else:
        print(f"\n⚠️  Could not load F_all_evaluations file")

    # Count violations from detailed requirements
    if results['reqs_all'] is not None:
        total_violations = results['reqs_all'].sum().sum()
        print(f"📊 Total Violations Found: {int(total_violations)}")
        analysis['violations'] = int(total_violations)

        print(f"\n   Violations per constraint:")
        for col in results['reqs_all'].columns:
            col_sum = results['reqs_all'][col].sum()
            print(f"     {col}: {int(col_sum)}")
    else:
        print(f"\n⚠️  Reqs_all_evaluations file not found - cannot count detailed violations")
        print(f"     (PFES+SAMOTA may not save this file)")

    # Count violations from summary if available
    if results['reqs_summary'] is not None and results['reqs_all'] is None:
        print(f"\n   Using summary requirements file:")
        for col in results['reqs_summary'].columns:
            val = results['reqs_summary'][col].iloc[0]
            print(f"     {col}: {int(val)}")
        # Try to sum it
        if 'conjunction' in results['reqs_summary'].columns:
            total_viol = results['reqs_summary']['conjunction'].iloc[0]
            analysis['violations'] = int(total_viol)
            print(f"   📊 Total Violations: {int(total_viol)}")

    # Efficiency
    if 'evaluations' in analysis and 'violations' in analysis:
        efficiency = analysis['violations'] / analysis['evaluations']
        print(f"\n⚡ Efficiency (violations/eval): {efficiency:.4f}")
        analysis['efficiency'] = efficiency

    # Best scores found
    if results['score_summary'] is not None:
        print(f"\n📈 Best Scores Found:")
        for col in results['score_summary'].columns:
            val = results['score_summary'][col].iloc[0]
            print(f"     {col}: {val:.6f}")

    # Objectives covered
    if results['reqs_all'] is not None:
        objectives_with_violations = (results['reqs_all'].sum() > 0).sum()
        total_objectives = len(results['reqs_all'].columns)
        print(f"\n🎯 Objectives Covered: {objectives_with_violations}/{total_objectives}")
        analysis['objectives_covered'] = f"{objectives_with_violations}/{total_objectives}"

    return analysis

def compare_results(pfes_analysis, samota_analysis):
    """Compare PFES vs PFES+SAMOTA"""
    print(f"\n\n{'='*70}")
    print(f"COMPARISON: PFES Baseline vs PFES+SAMOTA Hybrid")
    print(f"{'='*70}\n")

    if 'evaluations' in pfes_analysis and 'evaluations' in samota_analysis:
        pfes_evals = pfes_analysis['evaluations']
        samota_evals = samota_analysis['evaluations']

        print(f"📊 Evaluations Used:")
        print(f"  PFES:        {pfes_evals}")
        print(f"  PFES+SAMOTA: {samota_evals}")
        print(f"  Difference:  {abs(pfes_evals - samota_evals)} ({abs(pfes_evals - samota_evals)/max(pfes_evals, samota_evals)*100:.1f}%)")

        if samota_evals < pfes_evals:
            budget_saved = (pfes_evals - samota_evals) / pfes_evals * 100
            print(f"  ✓ PFES+SAMOTA uses {budget_saved:.1f}% FEWER evaluations!")

    if 'violations' in pfes_analysis and 'violations' in samota_analysis:
        pfes_viol = pfes_analysis['violations']
        samota_viol = samota_analysis['violations']

        print(f"\n📊 Violations Found:")
        print(f"  PFES:        {pfes_viol}")
        print(f"  PFES+SAMOTA: {samota_viol}")

        if pfes_viol > samota_viol:
            improvement = (pfes_viol - samota_viol) / pfes_viol * 100
            print(f"  ⚠️  PFES+SAMOTA found {improvement:.1f}% FEWER violations")
        elif samota_viol > pfes_viol:
            improvement = (samota_viol - pfes_viol) / pfes_viol * 100
            print(f"  ✓ PFES+SAMOTA found {improvement:.1f}% MORE violations (better!)")
        else:
            print(f"  = Same violations found")

    if 'efficiency' in pfes_analysis and 'efficiency' in samota_analysis:
        pfes_eff = pfes_analysis['efficiency']
        samota_eff = samota_analysis['efficiency']

        print(f"\n⚡ Efficiency (violations per evaluation):")
        print(f"  PFES:        {pfes_eff:.4f}")
        print(f"  PFES+SAMOTA: {samota_eff:.4f}")

        if samota_eff > pfes_eff:
            speedup = samota_eff / pfes_eff
            print(f"  ✓ PFES+SAMOTA is {speedup:.2f}x MORE EFFICIENT (better!)")
        elif pfes_eff > samota_eff:
            slowdown = pfes_eff / samota_eff
            print(f"  ⚠️  PFES+SAMOTA is {slowdown:.2f}x LESS efficient")
        else:
            print(f"  = Same efficiency")
    elif 'violations' in pfes_analysis and 'violations' not in samota_analysis:
        print(f"\n⚠️  Cannot calculate efficiency - PFES+SAMOTA violations file missing")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze existing PFES vs PFES+SAMOTA results")
    parser.add_argument("--pfes", type=str, default="pfes_baseline",
                       help="Directory with PFES baseline results")
    parser.add_argument("--samota", type=str, default="pfes_samota_baseline",
                       help="Directory with PFES+SAMOTA results")

    args = parser.parse_args()

    # Load results
    print("Loading results...")

    if not os.path.exists(args.pfes):
        print(f"⚠️  PFES directory not found: {args.pfes}")
        print(f"   Looking for: {os.path.abspath(args.pfes)}")
        pfes_results = None
    else:
        pfes_results = load_results(args.pfes)

    if not os.path.exists(args.samota):
        print(f"⚠️  PFES+SAMOTA directory not found: {args.samota}")
        print(f"   Looking for: {os.path.abspath(args.samota)}")
        samota_results = None
    else:
        samota_results = load_results(args.samota)

    # Analyze
    if pfes_results:
        pfes_analysis = analyze_results("PFES BASELINE", pfes_results)
    else:
        pfes_analysis = {}

    if samota_results:
        samota_analysis = analyze_results("PFES+SAMOTA HYBRID", samota_results)
    else:
        samota_analysis = {}

    # Compare
    if pfes_analysis and samota_analysis:
        compare_results(pfes_analysis, samota_analysis)
    else:
        print("\n⚠️  Could not load both result sets for comparison")

    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
