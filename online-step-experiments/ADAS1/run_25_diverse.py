#!/usr/bin/env python3
"""
Run PFES_SAMOTA_DIVERSE 25 times and save results.

Usage:
  python run_25_diverse.py               # run all 25
  python run_25_diverse.py --start 6    # resume from run 6
  python run_25_diverse.py --runs 5     # quick test with 5 runs

Results saved to: results_25runs_diverse/run_1/ .. run_25/
Each run_N/ contains:
  F_all_evaluations_NSGA3_0.csv  -- fitness per eval (convergence/diversity analysis)
  X_all_evaluations_NSGA3_0.csv  -- parameters per eval
  score_NSGA3_1.csv              -- min score per objective
  reqs_NSGA3_1.csv               -- requirement violation counts
  results.json                   -- full metrics including pattern_counts
"""

import subprocess
import os
import sys
import argparse


OUT_BASE = "results_25runs_diverse"
BUDGET   = 900


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start",  type=int, default=1,    help="Resume from this run number")
    p.add_argument("--runs",   type=int, default=25,   help="Total number of runs")
    p.add_argument("--budget", type=int, default=BUDGET, help="Eval budget per run")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_BASE, exist_ok=True)

    for run_num in range(args.start, args.runs + 1):
        run_dir = os.path.join(OUT_BASE, f"run_{run_num}")

        if os.path.exists(os.path.join(run_dir, "F_all_evaluations_NSGA3_0.csv")):
            print(f"Run {run_num}/{args.runs}: already done, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"DIVERSE  run {run_num}/{args.runs}  (seed={run_num}, budget={args.budget})")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "_run_diverse_single.py",
            "--run_num", str(run_num),
            "--out_dir", run_dir,
            "--budget",  str(args.budget),
            "--seed",    str(run_num),
        ]
        print(f"Command: {' '.join(cmd)}\n")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"!! Run {run_num} FAILED (exit code {result.returncode})")
        else:
            print(f"Run {run_num} done -> {run_dir}/")

    print(f"\nAll {args.runs} runs complete.")
    print(f"Results in: {OUT_BASE}/run_1/ .. run_{args.runs}/")


if __name__ == "__main__":
    main()
