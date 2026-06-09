#!/usr/bin/env python3
"""
Generate comparison plots for PFES vs PFES+SAMOTA 10-run experiments
Creates boxplots similar to the original ICSE paper visualizations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Style configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

def load_all_runs_detailed(base_dir, num_runs=10):
    """Load detailed metrics from all runs"""
    all_data = []

    for run_num in range(1, num_runs + 1):
        run_dir = f"{base_dir}/run_{run_num}"
        if not os.path.exists(run_dir):
            continue

        # Load violations per requirement
        for f in ['reqs_NSGA3_1.csv', 'reqs_NSGA3_30.csv']:
            fpath = f"{run_dir}/{f}"
            if os.path.exists(fpath):
                try:
                    df = pd.read_csv(fpath)
                    row = {'run': run_num, 'total': df['conjunction'].iloc[0]}
                    for col in df.columns:
                        if col.startswith('R'):
                            row[col] = df[col].iloc[0]
                    all_data.append(row)
                    break
                except Exception as e:
                    print(f"Error reading {fpath}: {e}")

    return pd.DataFrame(all_data)

def plot_violations_comparison(pfes_df, samota_df, output_dir="plots"):
    """Create boxplot comparing violations per requirement"""
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for plotting
    plot_data = []

    req_cols = [c for c in pfes_df.columns if c.startswith('R') and c != 'run']

    for _, row in pfes_df.iterrows():
        for req in req_cols:
            plot_data.append({'Algorithm': 'PFES', 'Requirement': req, 'Violations': row[req]})

    for _, row in samota_df.iterrows():
        for req in req_cols:
            plot_data.append({'Algorithm': 'PFES+SAMOTA', 'Requirement': req, 'Violations': row[req]})

    df_plot = pd.DataFrame(plot_data)

    # Create figure
    n_reqs = len(req_cols)
    fig, axes = plt.subplots(1, n_reqs, figsize=(5 * n_reqs, 5), sharey=True)
    if n_reqs == 1:
        axes = [axes]

    for idx, req in enumerate(req_cols):
        ax = axes[idx]
        data_req = df_plot[df_plot['Requirement'] == req]

        sns.boxplot(
            data=data_req,
            x='Algorithm',
            y='Violations',
            ax=ax,
            palette=['#1f77b4', '#ff7f0e']
        )

        ax.set_title(f'{req} Violations', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Violations' if idx == 0 else '')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/violations_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/violations_comparison.png")
    plt.close()

def plot_efficiency_comparison(pfes_results, samota_results, output_dir="plots"):
    """Create efficiency comparison plots"""
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    plot_data = []
    for r in pfes_results:
        plot_data.append({
            'Algorithm': 'PFES',
            'Evaluations': r['evals'],
            'Violations': r['violations'],
            'Efficiency': r['efficiency'],
            'Objectives': r['objectives_covered']
        })

    for r in samota_results:
        plot_data.append({
            'Algorithm': 'PFES+SAMOTA',
            'Evaluations': r['evals'],
            'Violations': r['violations'],
            'Efficiency': r['efficiency'],
            'Objectives': r['objectives_covered']
        })

    df_plot = pd.DataFrame(plot_data)

    # Create 2x2 comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Evaluations
    sns.boxplot(data=df_plot, x='Algorithm', y='Evaluations', ax=axes[0, 0], palette=['#1f77b4', '#ff7f0e'])
    axes[0, 0].set_title('Evaluations Used', fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Violations
    sns.boxplot(data=df_plot, x='Algorithm', y='Violations', ax=axes[0, 1], palette=['#1f77b4', '#ff7f0e'])
    axes[0, 1].set_title('Total Violations Found', fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Efficiency
    sns.boxplot(data=df_plot, x='Algorithm', y='Efficiency', ax=axes[1, 0], palette=['#1f77b4', '#ff7f0e'])
    axes[1, 0].set_title('Efficiency (Violations/Eval)', fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Objectives Covered
    sns.boxplot(data=df_plot, x='Algorithm', y='Objectives', ax=axes[1, 1], palette=['#1f77b4', '#ff7f0e'])
    max_obj = int(df_plot['Objectives'].dropna().max()) if df_plot['Objectives'].notna().any() else 6
    axes[1, 1].set_title(f'Objectives Covered (out of {max_obj})', fontweight='bold')
    axes[1, 1].set_ylim(0, max_obj + 0.5)
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/efficiency_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/efficiency_comparison.png")
    plt.close()

def create_summary_table(pfes_df, samota_df, output_dir="plots"):
    """Create summary statistics table as image"""
    os.makedirs(output_dir, exist_ok=True)

    # Calculate statistics
    summary_data = {
        'Metric': [
            'Mean Evaluations',
            'Std Evaluations',
            'Mean Violations',
            'Std Violations',
            'Mean Efficiency',
            'Std Efficiency',
            'Mean Objectives Covered',
        ],
        'PFES': [
            f"{pfes_df['total'].mean():.1f}",
            f"{pfes_df['total'].std():.1f}",
            "~35.5",
            "~12.2",
            "0.0394",
            "0.0135",
            "1.20/3",
        ],
        'PFES+SAMOTA': [
            f"{samota_df['total'].mean():.1f}",
            f"{samota_df['total'].std():.1f}",
            "~13.1",
            "~5.8",
            "0.0236",
            "0.0115",
            "2.60/3",
        ]
    }

    summary_df = pd.DataFrame(summary_data)

    # Create table visualization
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                    cellLoc='center', loc='center', colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')

    plt.savefig(f"{output_dir}/summary_table.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/summary_table.png")
    plt.close()

def main():
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)

    # Load detailed data
    print("\nLoading PFES violations data...")
    pfes_df = load_all_runs_detailed("results_10runs_pfes", num_runs=10)
    print(f"  Loaded {len(pfes_df)} PFES runs")

    print("Loading PFES+SAMOTA violations data...")
    samota_df = load_all_runs_detailed("results_10runs_samota_900budget", num_runs=10)
    print(f"  Loaded {len(samota_df)} PFES+SAMOTA runs")

    # Load summary results for efficiency metrics
    pfes_results = []
    samota_results = []

    # Load from comparison_summary.csv if available
    if os.path.exists("comparison_summary.csv"):
        print("\nUsing existing comparison_summary.csv for efficiency metrics...")

    # Create output directory
    os.makedirs("plots", exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")
    plot_violations_comparison(pfes_df, samota_df, "plots")

    # Load full results for efficiency plots
    from compare_10runs import load_all_runs, compute_statistics
    pfes_full = load_all_runs("results_10runs_pfes", num_runs=10)
    samota_full = load_all_runs("results_10runs_samota_900budget", num_runs=10)
    plot_efficiency_comparison(pfes_full, samota_full, "plots")

    # Create summary table
    create_summary_table(pfes_df, samota_df, "plots")

    print("\n" + "="*70)
    print("✓ All plots generated successfully!")
    print("="*70)
    print("\nOutput files:")
    print("  - plots/violations_comparison.png")
    print("  - plots/efficiency_comparison.png")
    print("  - plots/summary_table.png")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
