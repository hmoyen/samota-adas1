#!/usr/bin/env python3
"""
Single-run wrapper for PFES_SAMOTA_DIVERSE.
Called by run_25_diverse.py — do not run directly.

Saves outputs in the standard format expected by:
  - analyze_convergence.py  (F_all_evaluations_NSGA3_0.csv, X_all_evaluations_NSGA3_0.csv)
  - analyze_failure_diversity.py  (F_all_evaluations_NSGA3_0.csv, X_all_evaluations_NSGA3_0.csv)
  - compare scripts (score_NSGA3_1.csv, reqs_NSGA3_1.csv)
"""

import argparse
import os
import json
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--run_num", type=int, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--budget",  type=int, default=900)
parser.add_argument("--seed",    type=int, default=1)
args = parser.parse_args()

np.random.seed(args.seed)

from PFES_SAMOTA_DIVERSE import pfes_samota_diverse

results = pfes_samota_diverse(max_iterations=30, budget=args.budget)

# Extract arrays (not JSON-serialisable)
X_array = results.pop("_X_array")
F_array = results.pop("_F_array")

# Save CSVs
os.makedirs(args.out_dir, exist_ok=True)

n_obj = F_array.shape[1]
obj_cols = [f"V{i}" for i in range(n_obj)]

pd.DataFrame(F_array, columns=obj_cols).to_csv(
    os.path.join(args.out_dir, "F_all_evaluations_NSGA3_0.csv"), index=False)

pd.DataFrame(X_array).to_csv(
    os.path.join(args.out_dir, "X_all_evaluations_NSGA3_0.csv"), index=False)

min_scores = np.min(F_array, axis=0)
pd.DataFrame([min_scores], columns=obj_cols).to_csv(
    os.path.join(args.out_dir, "score_NSGA3_1.csv"), index=False)

reqs = results["unsatisfied_reqs"]
pd.DataFrame([reqs], columns=[f"R{i}" for i in range(len(reqs))]).to_csv(
    os.path.join(args.out_dir, "reqs_NSGA3_1.csv"), index=False)

with open(os.path.join(args.out_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nRun {args.run_num} saved to: {args.out_dir}")
print(f"  Evals: {results['eval_count']}, Violations: {results['violations']}, "
      f"Objectives: {results['objectives_covered']}/{n_obj}")
print(f"  Patterns: {results['pattern_counts']}")
