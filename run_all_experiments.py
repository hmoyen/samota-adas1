#!/usr/bin/env python3
"""
Unified experiment runner for ICSE 2025 comparison study.
Runs all algorithms x all benchmarks with 30 independent runs each.

Algorithms:
  PF     - Parametric Falsification (PFES_falsification.py --optalg NSGA3)
  RS     - Random Search            (PFES_falsification.py --optalg RANDOM)
  FF     - Focused Falsification    (FOC_falsification.py)
  MERLOT - RL-based                 (PFRL_falsification.py)
  SAMOTA - PFES + SAMOTA hybrid     (PFES_SAMOTA.py)

Benchmarks: ADAS1, ADAS2, RR

Usage:
  python run_all_experiments.py
  python run_all_experiments.py --benchmarks ADAS1 ADAS2 --algorithms PF RS
  python run_all_experiments.py --nruns 5 --budget 900 --results_dir my_results
  python run_all_experiments.py --resume   # skip already-completed experiments

Output structure:
  results/
    ADAS1/PF/out/
      reqs_NSGA3_30.csv   (per-req violation counts, 30 rows)
      score_NSGA3_30.csv  (best scores, 30 rows)
    ADAS1/RS/out/
    ADAS1/FF/out/
    ADAS1/MERLOT/out/
    ADAS1/SAMOTA/out/
    ADAS2/...
    RR/...
    progress.json         (checkpoint file for --resume)
"""

import subprocess
import sys
import os
import json
import time
import argparse
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent / "online-step-experiments"

BENCHMARKS = ["ADAS1", "ADAS2", "RR"]

# Algorithm configurations: name -> (script, extra_args_template)
# {nruns}, {budget}, {seed}, {logdir} are filled in at runtime
ALGORITHMS = {
    "PF": {
        "script": "PFES_falsification.py",
        "args": "--size 30 --niterations 30 --nruns {nruns} --optalg NSGA3 --logdir {logdir} --seed {seed}",
        "desc": "Parametric Falsification (NSGA3)",
    },
    "RS": {
        "script": "PFES_falsification.py",
        "args": "--size 30 --niterations 30 --nruns {nruns} --optalg RANDOM --logdir {logdir} --seed {seed}",
        "desc": "Random Search",
    },
    "FF": {
        "script": "FOC_falsification.py",
        "args": "--size 30 --totbudget {budget} --nruns {nruns} --logdir {logdir} --seed {seed}",
        "desc": "Focused Falsification (FOC)",
    },
    "MERLOT": {
        "script": "PFRL_falsification.py",
        "args": "--nepisodes {budget} --nruns {nruns} --logdir {logdir} --seed {seed}",
        "desc": "MERLOT (RL-based)",
    },
    "SAMOTA": {
        "script": "PFES_SAMOTA.py",
        "args": "--nruns {nruns} --budget {budget} --logdir {logdir} --seed {seed}",
        "desc": "PFES + SAMOTA hybrid",
    },
    "SAMOTA_SW": {
        "script": "PFES_SAMOTA.py",
        "args": "--nruns {nruns} --budget {budget} --logdir {logdir} --seed {seed} --window_size 150",
        "desc": "PFES + SAMOTA + Sliding Window (last 150 samples)",
    },
}


# ============================================================================
# HELPERS
# ============================================================================

def load_progress(progress_file):
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {}


def save_progress(progress_file, progress):
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def experiment_key(benchmark, algorithm):
    return f"{benchmark}/{algorithm}"


def run_experiment(benchmark, algorithm, algo_cfg, nruns, budget, seed, results_dir):
    """
    Run one algorithm on one benchmark. Returns (success, duration, stdout_tail).
    """
    bench_dir = BASE_DIR / benchmark
    logdir = results_dir / benchmark / algorithm / "out"
    logdir.mkdir(parents=True, exist_ok=True)

    script = algo_cfg["script"]
    args_str = algo_cfg["args"].format(
        nruns=nruns,
        budget=budget,
        seed=seed,
        logdir=str(logdir),
    )
    cmd = [sys.executable, script] + args_str.split()

    log_file = results_dir / benchmark / algorithm / "run.log"
    print(f"\n  Command: python {script} {args_str}")
    print(f"  CWD:     {bench_dir}")
    print(f"  Log:     {log_file}")

    start = time.time()
    try:
        with open(log_file, "w") as lf:
            lf.write(f"# Experiment: {benchmark}/{algorithm}\n")
            lf.write(f"# Command: {' '.join(cmd)}\n")
            lf.write(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            lf.flush()

            proc = subprocess.Popen(
                cmd,
                cwd=bench_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            output_lines = []
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                output_lines.append(line)
                # Print progress lines to console (run headers, key metrics)
                if any(kw in line for kw in ["RUN ", "run ", "Duration", "violations", "saved", "Saved", "Error", "ERROR", "Traceback"]):
                    print(f"    {line.rstrip()}")

            proc.wait()

        duration = time.time() - start
        success = proc.returncode == 0

        if not success:
            print(f"  [FAILED] Exit code {proc.returncode} — see {log_file}")
            # Print last 20 lines for quick diagnosis
            tail = output_lines[-20:]
            for line in tail:
                print(f"    {line.rstrip()}")
        else:
            print(f"  [OK] Completed in {duration/60:.1f} min")

        return success, duration, "".join(output_lines[-5:])

    except FileNotFoundError as e:
        duration = time.time() - start
        print(f"  [FAILED] Script not found: {e}")
        return False, duration, str(e)
    except Exception as e:
        duration = time.time() - start
        print(f"  [FAILED] Exception: {e}")
        return False, duration, str(e)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run all algorithm/benchmark combinations")
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS,
                        choices=BENCHMARKS, help="Which benchmarks to run")
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS.keys()),
                        choices=list(ALGORITHMS.keys()), help="Which algorithms to run")
    parser.add_argument("--nruns", type=int, default=30,
                        help="Number of independent runs per experiment (default: 30)")
    parser.add_argument("--budget", type=int, default=900,
                        help="Evaluation budget per run (default: 900)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Base random seed; run i uses seed+i (default: 1)")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Top-level output directory (default: results/)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip experiments already marked as successful in progress.json")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    progress_file = results_dir / "progress.json"
    progress = load_progress(progress_file)

    # Build experiment list
    experiments = [
        (bench, alg)
        for bench in args.benchmarks
        for alg in args.algorithms
    ]

    total = len(experiments)
    print(f"\n{'='*70}")
    print(f"ICSE 2025 Experiment Runner")
    print(f"{'='*70}")
    print(f"  Benchmarks : {args.benchmarks}")
    print(f"  Algorithms : {args.algorithms}")
    print(f"  Runs/exp   : {args.nruns}")
    print(f"  Budget/run : {args.budget} evaluations")
    print(f"  Base seed  : {args.seed}")
    print(f"  Results dir: {results_dir.resolve()}")
    print(f"  Total exps : {total}")
    print(f"  Resume     : {args.resume}")
    print(f"  Dry run    : {args.dry_run}")
    print(f"{'='*70}\n")

    overall_start = time.time()
    completed = 0
    failed = []

    for idx, (benchmark, algorithm) in enumerate(experiments, 1):
        key = experiment_key(benchmark, algorithm)
        algo_cfg = ALGORITHMS[algorithm]

        print(f"\n[{idx}/{total}] {benchmark} / {algorithm} — {algo_cfg['desc']}")

        # Check if already done (resume mode)
        if args.resume and progress.get(key, {}).get("status") == "success":
            print("  [SKIP] Already completed (resume mode)")
            completed += 1
            continue

        # Check script exists
        script_path = BASE_DIR / benchmark / algo_cfg["script"]
        if not script_path.exists():
            msg = f"Script not found: {script_path}"
            print(f"  [SKIP] {msg}")
            progress[key] = {"status": "missing", "error": msg}
            save_progress(progress_file, progress)
            failed.append((benchmark, algorithm, msg))
            continue

        if args.dry_run:
            logdir = results_dir / benchmark / algorithm / "out"
            args_str = algo_cfg["args"].format(
                nruns=args.nruns, budget=args.budget,
                seed=args.seed, logdir=str(logdir)
            )
            print(f"  [DRY RUN] cd {BASE_DIR / benchmark} && python {algo_cfg['script']} {args_str}")
            completed += 1
            continue

        # Run experiment
        success, duration, tail = run_experiment(
            benchmark, algorithm, algo_cfg,
            nruns=args.nruns,
            budget=args.budget,
            seed=args.seed,
            results_dir=results_dir,
        )

        progress[key] = {
            "status": "success" if success else "failed",
            "duration_seconds": round(duration),
            "duration_min": round(duration / 60, 1),
            "tail": tail[:500],
        }
        save_progress(progress_file, progress)

        if success:
            completed += 1
        else:
            failed.append((benchmark, algorithm, "non-zero exit code"))

    # Summary
    total_elapsed = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Completed : {completed}/{total}")
    print(f"  Failed    : {len(failed)}")
    print(f"  Total time: {total_elapsed/3600:.1f} hours")

    if failed:
        print(f"\n  Failed experiments:")
        for bench, alg, reason in failed:
            print(f"    {bench}/{alg}: {reason}")

    print(f"\n  Results saved to: {results_dir.resolve()}")
    print(f"  Progress file  : {progress_file}")
    print(f"\n  Next step: python analyze_all_results.py --results_dir {results_dir}")


if __name__ == "__main__":
    main()
