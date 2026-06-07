#!/usr/bin/env python3
"""Inspect what the requirements CSV files actually contain"""

import pandas as pd
import os

def inspect_file(filepath):
    """Show details about a requirements file"""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    df = pd.read_csv(filepath)

    print(f"\n📄 File: {filepath}")
    print(f"   Shape: {df.shape} (rows={len(df)}, cols={len(df.columns)})")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   First 10 rows:")
    print(df.head(10).to_string())
    print(f"\n   Data types: {df.dtypes.to_dict()}")
    print(f"\n   Summary statistics:")
    print(df.describe().to_string())

# Check PFES baseline
print("="*70)
print("PFES BASELINE - Requirements File")
print("="*70)
inspect_file("pfes_baseline/Reqs_all_evaluations_NSGA3_0.csv")

# Check PFES+SAMOTA summary
print("\n" + "="*70)
print("PFES+SAMOTA - Summary Requirements File")
print("="*70)
inspect_file("pfes_samota_baseline/reqs_NSGA3_1.csv")
