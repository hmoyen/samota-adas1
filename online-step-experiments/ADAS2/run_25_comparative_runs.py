#!/usr/bin/env python3
"""
Run 25 comparative runs for ADAS2:
  1. PFES Baseline  → results_25runs_pfes/run_N/
  2. PFES+SAMOTA    → results_25runs_samota/run_N/

Usage:
  python run_25_comparative_runs.py                  # run both algorithms
  python run_25_comparative_runs.py --algo pfes      # only PFES baseline
  python run_25_comparative_runs.py --algo samota    # only PFES+SAMOTA
  python run_25_comparative_runs.py --start 6        # resume from run 6
"""

import subprocess
import os
import shutil
import argparse

N_RUNS = 25


def move_csv_files(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if fname.endswith(".csv") or fname.endswith(".log"):
            shutil.move(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))


def banner(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def run_pfes_baseline(run_num, out_base="results_25runs_pfes"):
    run_dir = os.path.join(out_base, f"run_{run_num}")
    if os.path.exists(os.path.join(run_dir, "score_NSGA3_1.csv")):
        print(f"  PFES run {run_num}: already exists, skipping.")
        return True

    os.makedirs(run_dir, exist_ok=True)
    banner(f"PFES BASELINE  run {run_num}/{N_RUNS}")

    cmd = [
        "python3", "PFES_falsification.py",
        "--size", "30",
        "--niterations", "30",
        "--nruns", "1",
        "--optalg", "NSGA3",
        "--logdir", run_dir,
        "--seed", str(run_num),
    ]
    print(f"  Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    ok = result.returncode == 0
    if not ok:
        print(f"  !! PFES run {run_num} FAILED (exit code {result.returncode})")
    return ok


def run_pfes_samota(run_num, out_base="results_25runs_samota_seeded"):
    run_dir = os.path.join(out_base, f"run_{run_num}")
    if os.path.exists(os.path.join(run_dir, "score_NSGA3_1.csv")):
        print(f"  PFES+SAMOTA run {run_num}: already exists, skipping.")
        return True

    if os.path.exists("pfes_samota_baseline"):
        shutil.rmtree("pfes_samota_baseline")

    banner(f"PFES+SAMOTA  run {run_num}/{N_RUNS}")
    result = subprocess.run(["python3", "PFES_SAMOTA.py", "--seed", str(run_num)])
    ok = result.returncode == 0

    if ok and os.path.exists("pfes_samota_baseline"):
        move_csv_files("pfes_samota_baseline", run_dir)
        shutil.rmtree("pfes_samota_baseline", ignore_errors=True)
    elif not ok:
        print(f"  !! PFES+SAMOTA run {run_num} FAILED (exit code {result.returncode})")

    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["pfes", "samota", "all"], default="all")
    parser.add_argument("--start", type=int, default=1,
                        help="Resume from run N (default: 1)")
    parser.add_argument("--n", type=int, default=N_RUNS,
                        help=f"Total number of runs (default: {N_RUNS})")
    args = parser.parse_args()

    run_pfes   = args.algo in ("pfes",   "all")
    run_samota = args.algo in ("samota", "all")

    stats = {"pfes": {"ok": 0, "fail": 0}, "samota": {"ok": 0, "fail": 0}}

    for run_num in range(args.start, args.n + 1):
        print(f"\n\n{'#'*70}")
        print(f"  OVERALL PROGRESS: run {run_num}/{args.n}")
        print(f"{'#'*70}")

        if run_pfes:
            ok = run_pfes_baseline(run_num)
            stats["pfes"]["ok" if ok else "fail"] += 1

        if run_samota:
            ok = run_pfes_samota(run_num)
            stats["samota"]["ok" if ok else "fail"] += 1

    print(f"\n\n{'='*70}")
    print("COMPLETED")
    print(f"{'='*70}")
    if run_pfes:
        s = stats["pfes"]
        print(f"  PFES Baseline: {s['ok']} succeeded, {s['fail']} failed  → results_25runs_pfes/")
    if run_samota:
        s = stats["samota"]
        print(f"  PFES+SAMOTA:   {s['ok']} succeeded, {s['fail']} failed  → results_25runs_samota/")


if __name__ == "__main__":
    main()
