#!/usr/bin/env python3
"""
Surrogate Precision Test — Does the surrogate actually pick better candidates?

Question: Given what the surrogate knows at checkpoint t, does it rank
truly-violating candidates higher than a naive random draw would?

Method (no new simulations needed):
  1. Load a completed run (X_all_evaluations + F_all_evaluations)
  2. At each checkpoint t (e.g., 300, 400, 500, 600, 700):
       - Training set: evaluations 0..t   (what the surrogate has seen)
       - Test set:     evaluations t..end (outcomes the surrogate hasn't seen yet)
  3. Train the SAMOTAPerObjectiveEnsemble on the training set
  4. Score every point in the test set using the surrogate
  5. Sort test set by predicted fitness (lowest = most likely violation)
  6. Measure:
       - Naive hit rate: fraction of violations in the full test set (= random baseline)
       - Surrogate Precision@K: fraction of violations in the top-K surrogate picks
       - Lift@K: Precision@K / naive hit rate  (>1 = surrogate is better than random)

Outputs:
  - plots/surrogate_precision/precision_at_k.png
  - plots/surrogate_precision/lift_by_checkpoint.png
  - plots/surrogate_precision/precision_summary.csv

Usage:
  python evaluate_surrogate_precision.py
  python evaluate_surrogate_precision.py --runs results_25runs_samota_seeded
  python evaluate_surrogate_precision.py --max_runs 5   # quick test on 5 runs
"""

import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from SAMOTA_ensemble import SAMOTAPerObjectiveEnsemble

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "plots", "surrogate_precision")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINTS = [300, 400, 500, 600, 700]   # training cutoff: surrogate knows evals 0..t
TOP_K_FRACS = [0.05, 0.10, 0.20, 0.30]   # top 5%, 10%, 20%, 30% of test set
REQ_NAMES   = ["R0", "R1", "R2"]
N_REQ       = len(REQ_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_run(run_dir):
    """Return (X, F, R) arrays. R is binary violation matrix (1=violated)."""
    x_path = os.path.join(run_dir, "X_all_evaluations_NSGA3_0.csv")
    f_path = os.path.join(run_dir, "F_all_evaluations_NSGA3_0.csv")
    if not os.path.exists(x_path) or not os.path.exists(f_path):
        return None, None, None

    X = pd.read_csv(x_path).values.astype(float)
    F = pd.read_csv(f_path, header=None).apply(
            pd.to_numeric, errors="coerce").dropna().values.astype(float)

    # Violation matrix R (1 = violated)
    reqs_path = os.path.join(run_dir, "Reqs_all_evaluations_NSGA3_0.csv")
    if os.path.exists(reqs_path):
        df = pd.read_csv(reqs_path, header=None)
        df = df.replace({"True": 1, "False": 0, True: 1, False: 0})
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        R = (~df.values.astype(bool)).astype(int)
    else:
        R = (F < 0).astype(int)

    n = min(X.shape[0], F.shape[0], R.shape[0])
    return X[:n], F[:n], R[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Core surrogate precision computation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_surrogate_at_checkpoint(X, F, R, t):
    """
    At checkpoint t:
      - Train surrogate on X[0:t], F[0:t]
      - Score test points X[t:] using surrogate
      - Return precision metrics vs naive random baseline

    Returns dict with:
      checkpoint, n_train, n_test, naive_hit_rate,
      precision_at_k (dict: top_frac -> hit_rate),
      lift_at_k      (dict: top_frac -> lift)
    """
    n_total = X.shape[0]
    if t >= n_total - 10:
        return None  # not enough test data

    X_train, F_train = X[:t], F[:t]
    X_test,  R_test  = X[t:], R[t:]

    n_test     = X_test.shape[0]
    n_obj      = F_train.shape[1]
    naive_rate = float(np.any(R_test > 0, axis=1).mean())

    if naive_rate == 0:
        return None  # no violations in test set — lift undefined

    # Train one surrogate per objective on training data
    surrogates = []
    for obj in range(n_obj):
        try:
            ens = SAMOTAPerObjectiveEnsemble(
                X_train, F_train[:, obj],
                normalize=True, obj_name=f"V{obj}"
            )
            surrogates.append(ens)
        except Exception:
            surrogates.append(None)

    # Score every test point: predicted fitness = min over objectives
    # (lower = more likely to violate)
    pred_scores = np.zeros((n_test, n_obj))
    for obj, ens in enumerate(surrogates):
        if ens is None:
            pred_scores[:, obj] = 0.0
        else:
            try:
                pred_scores[:, obj] = ens.predict(X_test)
            except Exception:
                pred_scores[:, obj] = 0.0

    # Rank by minimum predicted score across objectives (most negative = best pick)
    min_pred = pred_scores.min(axis=1)
    sorted_idx = np.argsort(min_pred)   # ascending: most negative first

    # Compute Precision@K and Lift@K for each top-fraction
    is_violation = np.any(R_test > 0, axis=1)
    precision_at_k = {}
    lift_at_k      = {}

    for frac in TOP_K_FRACS:
        k = max(1, int(np.ceil(frac * n_test)))
        top_k_idx = sorted_idx[:k]
        precision  = float(is_violation[top_k_idx].mean())
        lift       = precision / naive_rate if naive_rate > 0 else float("nan")
        precision_at_k[frac] = precision
        lift_at_k[frac]      = lift

    return {
        "checkpoint":      t,
        "n_train":         t,
        "n_test":          n_test,
        "naive_hit_rate":  naive_rate,
        "precision_at_k":  precision_at_k,
        "lift_at_k":       lift_at_k,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run experiment across all runs
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=None,
                        help="Glob pattern for run directories (default: auto-detect)")
    parser.add_argument("--max_runs", type=int, default=999,
                        help="Max number of runs to process (default: all)")
    return parser.parse_args()


def find_runs(pattern=None):
    for p in [
        pattern,
        os.path.join(BASE_DIR, "results_25runs_samota_seeded", "run_*"),
        os.path.join(BASE_DIR, "results_25runs_samota",        "run_*"),
        os.path.join(BASE_DIR, "results_10runs_samota_900budget", "run_*"),
    ]:
        if p is None:
            continue
        dirs = sorted(glob.glob(p))
        if dirs:
            print(f"Using runs from: {p}  ({len(dirs)} found)")
            return dirs
    return []


args = parse_args()
run_dirs = find_runs(args.runs)[:args.max_runs]

if not run_dirs:
    raise SystemExit("No run directories found.")

print(f"\nEvaluating surrogate precision across {len(run_dirs)} run(s)...")
print(f"Checkpoints: {CHECKPOINTS}")
print(f"Top-K fracs: {[f'{f*100:.0f}%' for f in TOP_K_FRACS]}\n")

# Collect results: checkpoint -> list of metric dicts
all_results = {t: [] for t in CHECKPOINTS}

for run_idx, run_dir in enumerate(run_dirs):
    X, F, R = load_run(run_dir)
    if X is None:
        print(f"  run {run_idx+1}: no data, skipping")
        continue
    print(f"  run {run_idx+1}: {X.shape[0]} evals, "
          f"{int(np.any(R > 0, axis=1).sum())} violations")

    for t in CHECKPOINTS:
        result = evaluate_surrogate_at_checkpoint(X, F, R, t)
        if result is not None:
            all_results[t].append(result)


# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 85)
print(f"SURROGATE PRECISION RESULTS  (mean across {len(run_dirs)} runs)")
print("=" * 85)
header = f"{'Checkpoint':>12} {'Naive rate':>11} " + \
         "".join(f"  P@{int(f*100)}%  Lift" for f in TOP_K_FRACS)
print(header)
print("-" * 85)

csv_rows = []
for t in CHECKPOINTS:
    results = all_results[t]
    if not results:
        continue

    naive   = np.mean([r["naive_hit_rate"] for r in results])
    row_str = f"  t={t:>4} ({len(results)} runs)  {naive:>8.3f}  "
    csv_row = {"checkpoint": t, "n_runs": len(results), "naive_hit_rate": round(naive, 4)}

    for frac in TOP_K_FRACS:
        precs = [r["precision_at_k"][frac] for r in results]
        lifts = [r["lift_at_k"][frac]      for r in results]
        p_mean = np.mean(precs)
        l_mean = np.mean(lifts)
        row_str += f"{p_mean:>6.3f}  {l_mean:>4.2f}  "
        csv_row[f"P@{int(frac*100)}pct"]    = round(p_mean, 4)
        csv_row[f"Lift@{int(frac*100)}pct"] = round(l_mean, 4)

    print(row_str)
    csv_rows.append(csv_row)

print("=" * 85)
print("Naive rate = fraction of violations in the UNSEEN test set (what random gives)")
print("P@K       = fraction of violations if you pick the top-K surrogate candidates")
print("Lift@K    = P@K / Naive rate  (>1 means surrogate beats random)")

pd.DataFrame(csv_rows).to_csv(
    os.path.join(OUTPUT_DIR, "precision_summary.csv"), index=False)
print(f"\nSaved: {OUTPUT_DIR}/precision_summary.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Precision@K vs naive across checkpoints
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

checkpoints_with_data = [t for t in CHECKPOINTS if all_results[t]]
naive_means = [np.mean([r["naive_hit_rate"] for r in all_results[t]])
               for t in checkpoints_with_data]

colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8"]

ax = axes[0]
ax.plot(checkpoints_with_data, naive_means, "k--", lw=2, label="Naive (random baseline)")
for frac, color in zip(TOP_K_FRACS, colors):
    prec_means = [np.mean([r["precision_at_k"][frac] for r in all_results[t]])
                  for t in checkpoints_with_data]
    prec_stds  = [np.std ([r["precision_at_k"][frac] for r in all_results[t]])
                  for t in checkpoints_with_data]
    ax.plot(checkpoints_with_data, prec_means, color=color, lw=2,
            marker="o", label=f"Surrogate top {int(frac*100)}%")
    ax.fill_between(checkpoints_with_data,
                    np.array(prec_means) - np.array(prec_stds),
                    np.array(prec_means) + np.array(prec_stds),
                    color=color, alpha=0.12)

ax.set_xlabel("Training checkpoint (evals seen by surrogate)", fontsize=11)
ax.set_ylabel("Hit rate (fraction that are violations)", fontsize=11)
ax.set_title("Surrogate Precision vs Naive Random\n"
             "Does the surrogate pick candidates that are actually violations?", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Plot 2: Lift@K across checkpoints
ax = axes[1]
ax.axhline(1.0, color="k", ls="--", lw=1.5, label="No improvement (lift=1)")
for frac, color in zip(TOP_K_FRACS, colors):
    lift_means = [np.mean([r["lift_at_k"][frac] for r in all_results[t]])
                  for t in checkpoints_with_data]
    lift_stds  = [np.std ([r["lift_at_k"][frac] for r in all_results[t]])
                  for t in checkpoints_with_data]
    ax.plot(checkpoints_with_data, lift_means, color=color, lw=2,
            marker="o", label=f"Top {int(frac*100)}%")
    ax.fill_between(checkpoints_with_data,
                    np.array(lift_means) - np.array(lift_stds),
                    np.array(lift_means) + np.array(lift_stds),
                    color=color, alpha=0.12)

ax.set_xlabel("Training checkpoint (evals seen by surrogate)", fontsize=11)
ax.set_ylabel("Lift@K  (surrogate / random hit rate)", fontsize=11)
ax.set_title("Surrogate Lift Over Random\n"
             "Lift > 1 = surrogate beats random; Lift = 1 = no benefit", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle("Surrogate Precision Test — Is the Surrogate Better Than Naive Random Selection?",
             fontsize=13, y=1.02)
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "surrogate_precision.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Precision@K bar chart (best checkpoint only)
# ─────────────────────────────────────────────────────────────────────────────
best_t = checkpoints_with_data[-2] if len(checkpoints_with_data) >= 2 else checkpoints_with_data[-1]
results_best = all_results[best_t]

fig, ax = plt.subplots(figsize=(8, 5))
x      = np.arange(len(TOP_K_FRACS))
naive  = np.mean([r["naive_hit_rate"] for r in results_best])

prec_m = [np.mean([r["precision_at_k"][f] for r in results_best]) for f in TOP_K_FRACS]
prec_s = [np.std ([r["precision_at_k"][f] for r in results_best]) for f in TOP_K_FRACS]

bars = ax.bar(x, prec_m, yerr=prec_s, capsize=5, color="#FF5722", alpha=0.8,
              label="Surrogate-guided selection")
ax.axhline(naive, color="steelblue", ls="--", lw=2, label=f"Naive random (rate={naive:.3f})")

for bar, p in zip(bars, prec_m):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{p:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([f"Top {int(f*100)}%\n(K={int(np.ceil(f*results_best[0]['n_test']))})"
                    for f in TOP_K_FRACS], fontsize=10)
ax.set_ylabel("Fraction of selected candidates that are violations", fontsize=11)
ax.set_title(f"Surrogate Precision at Checkpoint t={best_t}\n"
             f"(trained on {best_t} evals, tested on {results_best[0]['n_test']} unseen points)",
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, min(1.0, max(prec_m) * 1.4))
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "precision_bar.png")
plt.savefig(out2, dpi=150)
plt.close()
print(f"Saved: {out2}")

print(f"\nAll plots saved to: {OUTPUT_DIR}/")
print("\nInterpretation guide:")
print("  Lift > 1.5 at top 10%  → surrogate strongly outperforms random")
print("  Lift ≈ 1.0             → surrogate no better than random (not helping)")
print("  Lift < 1.0             → surrogate actively misleading the search")
