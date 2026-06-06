# PFES vs PFES+SAMOTA Comparative Experiments

**Author**: Helena Moyen
**Date**: May 20, 2026

## Quick Start

### Run Comparative Experiments (Recommended)

Compare PFES baseline vs PFES+SAMOTA hybrid on identical budgets:

```bash
# Run 5 iterations of each approach with 900 evals per run
python run_comparative_experiments.py --runs 5 --budget 900 --output results_comparison

# Results saved to:
#   results_comparison/summary.csv
#   results_comparison/pfes_runs.csv
#   results_comparison/pfes_samota_runs.csv
#   results_comparison/efficiency_analysis.txt
```

### Run PFES Only (Baseline)

```bash
python PFES_falsification.py --seed 0 --nruns 3 --logdir pfes_results
```

### Run PFES+SAMOTA Only (Hybrid)

```bash
python -c "import PFES_SAMOTA; PFES_SAMOTA.run_pfes_samota(budget=900, max_iterations=30)"
```

---

## Understanding Results

### Key Metrics

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **Total Violations** | Count of test cases violating ≥1 constraint | Higher is better (more faults found) |
| **R0, R1, R2** | Per-constraint violation breakdown | Shows which safety properties are findable |
| **Objectives Covered** | How many constraints were violated (0-3) | 3/3 = found violations for all constraints |
| **Efficiency (v/e)** | Violations per evaluation | Higher = better use of evaluation budget |
| **Time** | Wall-clock time for entire run | Both approaches use same 900 evals |

### Expected Results

Based on baseline PFES:

| Approach | Violations | Time | Efficiency | Notes |
|----------|-----------|------|-----------|-------|
| PFES (baseline) | 37 ± 8 | 50 min | 0.041 v/e | Pure simulator search |
| PFES+SAMOTA | 35-40 | 60-70 min | 0.040-0.045 v/e | Same violations, guided search |

**Note**: Surrogates add computational overhead, so PFES+SAMOTA takes longer wall-clock time but finds violations more efficiently within the search space.

---

## File Organization

### Core Implementation Files

```
PFES_SAMOTA.py                      # Main hybrid approach
├─ run_pfes_samota()               # Main entry point
├─ evaluate_test_case()            # Simulator wrapper
├─ global_search_nsga3()           # Phase 2a: GS
├─ local_search_phase()            # Phase 2b: LS
└─ art_initial_population()        # Phase 1: ART

PFES_falsification.py              # Baseline (pure NSGA3)
├─ run_pfes()                      # Main entry point
└─ uses only simulator, no surrogates

SAMOTA_ensemble.py                 # Per-objective surrogate ensemble
├─ SAMOTAPerObjectiveEnsemble
└─ Trains one ensemble per constraint

RBF.py                             # RBF neural network model
```

### Documentation

```
PFES_SAMOTA_ALGORITHMS_CORRECTED.md  # Formal pseudocode (3 objectives)
EXPERIMENT_GUIDE.md                  # This file
```

### Results

```
results_comparison/                # Output from comparative experiments
├─ summary.csv                     # Overall comparison table
├─ pfes_runs.csv                  # PFES per-run results
├─ pfes_samota_runs.csv           # PFES+SAMOTA per-run results
├─ efficiency_analysis.txt        # Statistical analysis
└─ comparison_report.txt          # Formatted summary (if exists)
```

---

## Understanding 3 Objectives

ADAS1 has **3 constraints**, not 5:

| Index | Constraint | Property | Threshold |
|-------|-----------|----------|-----------|
| **R0** | S0.a[0] | First autonomy constraint | constraints["S0"]["a"][0] |
| **R1** | S2.b[1] | Second behavior metric | constraints["S2"]["b"][1] |
| **R2** | S0.a[1] + S2.b[1] | Combined constraint | Two metrics together |

**Key Point**: Uncovered objectives = those where `min(fitness) >= 0` (no violations found yet).

---

## Algorithm Differences

### PFES (Baseline)

```
Phase 1: ART (300 evals) ─→ Find initial violations
Phase 2: NSGA3 (600 evals) ──→ Search all 3 objectives simultaneously
         ├─ Population: 30
         ├─ Generations: 20
         └─ All objectives always included
```

### PFES+SAMOTA (Hybrid)

```
Phase 1: ART (300 evals) ─→ Find initial violations
Phase 2: Loop (600 evals budget)
   ├─ Identify uncovered objectives (R0, R1, R2 status)
   ├─ GS: Train per-objective surrogates + NSGA3
   │   └─ Only on uncovered objectives (dynamic)
   ├─ LS: RBF per cluster + per-objective optimization
   │   └─ Focused refinement
   └─ Repeat until budget exhausted or all objectives covered
```

**Key Difference**: PFES+SAMOTA filters to uncovered objectives, making search more focused.

---

## Configuration

Edit `config.py` to change:

```python
# Constraints definition (read-only for comparison)
CONSTRAINTS = [
    {"S0": {"a": constraints["S0"]["a"][0]}},     # R0
    {"S2": {"b": constraints["S2"]["b"][1]}},     # R1
    {"S0": {"a": constraints["S0"]["a"][1]},      # R2
     "S2": {"b": constraints["S2"]["b"][1]}}
]

# Search space
SS_VARIABLES = {
    "car_speed": {"domain": float, "range": [5.0, 50.0]},
    "p_x": {"domain": float, "range": [0.0, 10.0]},
    # ... other variables
}

# Budget (don't change for comparison)
BATCH_SIZE = 5
MAX_STEPS = 100000
```

---

## Troubleshooting

### "Simulator not found"

```bash
# Install mdp_simulator wheel
pip install /home/lena/Downloads/icse2025_replication_package/CPS-simulator/dist/mdp_simulator-0.1.9-py3-none-any.whl
```

### "Python version incompatibility"

Use Python 3.10 or 3.11 for mdp_simulator compatibility:

```bash
python3.10 run_comparative_experiments.py --runs 5 --budget 900
```

### Results look different from baseline

Check:
1. Same random seed? (Add `--seed 0` to both)
2. Same constraint definitions in config.py
3. Same budget (900 evals)
4. Check simulator output path (verify MDP_FOLDER exists)

---

## Next Steps

1. **Run experiments**: `python run_comparative_experiments.py --runs 5`
2. **Analyze results**: Check `results_comparison/efficiency_analysis.txt`
3. **Compare metrics**: Look at R0, R1, R2 breakdown
4. **Calculate speedup**: Efficiency improvement = (SAMOTA_efficiency - PFES_efficiency) / PFES_efficiency

---

## Citation

If using this framework in publications, cite:

```
@inproceedings{ICSE2025,
  author = {Helena Moyen},
  title = {PFES+SAMOTA: Parametric Falsification with Surrogate-Assisted
           Many-Objective Test Generation},
  year = {2026}
}
```
