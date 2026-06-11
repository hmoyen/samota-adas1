#!/usr/bin/env python3
"""
Run the original PFES baseline 25 times, exactly matching the original paper's setup:
  - SEED = 1 for every run  (original hardcodes SEED=1 for all runs)
  - size=30, niterations=30  (900 evaluations total, same as SAMOTA budget)
  - NSGA3 algorithm

This is used to verify whether our modified PFES_falsification.py produces
results comparable to the original 30-run replication package.

Results saved to: results_25runs_pfes_original/run_1/ .. run_25/
"""

import subprocess
import os

N_RUNS = 25
OUT_BASE = "results_25runs_pfes_original"


def main():
    os.makedirs(OUT_BASE, exist_ok=True)

    for run_num in range(1, N_RUNS + 1):
        run_dir = os.path.join(OUT_BASE, f"run_{run_num}")

        # Skip if already done
        if os.path.exists(os.path.join(run_dir, "score_NSGA3_1.csv")):
            print(f"Run {run_num}/{N_RUNS}: already exists, skipping.")
            continue

        os.makedirs(run_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"PFES ORIGINAL  run {run_num}/{N_RUNS}  (seed=1, matching original paper)")
        print(f"{'='*60}")

        cmd = [
            "python3", "PFES_falsification.py",
            "--size", "30",
            "--niterations", "30",
            "--nruns", "1",
            "--optalg", "NSGA3",
            "--logdir", run_dir,
            "--seed", "1",   # Original always uses SEED=1
        ]
        print(f"Command: {' '.join(cmd)}\n")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"!! Run {run_num} FAILED (exit code {result.returncode})")
        else:
            print(f"Run {run_num} done → {run_dir}/")

    print(f"\n\nAll {N_RUNS} runs complete.")
    print(f"Results in: {OUT_BASE}/run_1/ .. run_{N_RUNS}/")
    print("\nTo compare with original 30 runs, check:")
    print("  score_NSGA3_1.csv   -- min score per objective (V0-V4)")
    print("  reqs_NSGA3_1.csv    -- violation counts (R0, R1, R2)")


if __name__ == "__main__":
    main()
