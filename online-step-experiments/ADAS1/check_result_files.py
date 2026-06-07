#!/usr/bin/env python3
"""Check what CSV files exist in result directories"""

import os
import glob

def check_directory(directory):
    """List all CSV files in a directory"""
    print(f"\n📁 {directory}:")
    print(f"   {'='*60}")

    if not os.path.exists(directory):
        print(f"   ⚠️  Directory not found: {os.path.abspath(directory)}")
        return

    # List all CSV files
    csv_files = glob.glob(f"{directory}/*.csv")

    if not csv_files:
        print(f"   ⚠️  No CSV files found")
        # Check subdirectories
        subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        if subdirs:
            print(f"   📂 Subdirectories found: {subdirs}")
            for subdir in subdirs:
                subdir_path = os.path.join(directory, subdir)
                sub_csv_files = glob.glob(f"{subdir_path}/*.csv")
                if sub_csv_files:
                    print(f"      📂 {subdir}:")
                    for f in sorted(sub_csv_files):
                        size = os.path.getsize(f) / 1024
                        print(f"         - {os.path.basename(f)} ({size:.1f} KB)")
    else:
        for f in sorted(csv_files):
            size = os.path.getsize(f) / 1024
            print(f"   ✓ {os.path.basename(f)} ({size:.1f} KB)")

# Check directories
print("="*70)
print("CHECKING RESULT FILES")
print("="*70)

check_directory("pfes_baseline")
check_directory("pfes_samota_baseline")
check_directory("comparison_results")
check_directory("results_comparison")
