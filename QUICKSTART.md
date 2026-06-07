# Quick Start Guide - Multi-Subject PFES+SAMOTA

## ✅ Setup Complete!

Your repository now has **ADAS1**, **ADAS2**, and **RR** subjects configured and ready to test.

## Run Your First Experiment (2 minutes)

### Option 1: Quick Test (minimal run)

```bash
# Test ADAS1 baseline
cd online-step-experiments/ADAS1
python3 PFES_falsification.py --size 5 --niterations 1 --nruns 1 --optalg NSGA3 --logdir test_adas1

# Output: test_adas1/*.csv files
# Check if it completes without errors ✓
```

### Option 2: Small Baseline (30 minutes per subject)

```bash
# Run ADAS1 baseline with 900 evaluations
cd online-step-experiments/ADAS1
python3 PFES_falsification.py --size 30 --niterations 30 --nruns 1 --optalg NSGA3 --logdir results_adas1

# Run ADAS2 baseline (same code, extended constraints)
cd ../ADAS2
python3 PFES_falsification.py --size 30 --niterations 30 --nruns 1 --optalg NSGA3 --logdir results_adas2

# Run RR baseline (9 variables instead of 6)
cd ../RR
python3 PFES_falsification.py --size 30 --niterations 30 --nruns 1 --optalg NSGA3 --logdir results_rr

# Results are in results_adas1/, results_adas2/, results_rr/
```

### Option 3: Hybrid Algorithm (40 minutes per subject)

```bash
# Test ADAS1 hybrid with PFES+SAMOTA
cd online-step-experiments/ADAS1
python3 PFES_SAMOTA.py

# Test ADAS2 hybrid
cd ../ADAS2
python3 PFES_SAMOTA.py

# Test RR hybrid
cd ../RR
python3 PFES_SAMOTA.py

# Results are in pfes_samota_baseline/ directories
```

## Repository Structure

```
online-step-experiments/
├── ADAS1/     ← 6 variables, 3 constraints (baseline)
├── ADAS2/     ← 6 variables, 6 constraints (extended)
└── RR/        ← 9 variables, 6 constraints (Rescue Robot)

Each has:
  ├── config.py               (subject-specific settings)
  ├── PFES_falsification.py   (baseline algorithm)
  ├── PFES_SAMOTA.py          (hybrid algorithm)
  ├── INPUT/                  (MDP models)
  └── utils/                  (shared utilities)
```

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `variable_builder.py` | Config-driven variable extraction | ✅ New |
| `PFES_falsification.py` | PFES baseline (all subjects) | ✅ Updated |
| `PFES_SAMOTA.py` | PFES+SAMOTA hybrid (all subjects) | ✅ Updated |
| `config.py` | Subject-specific configuration | ✅ Per-subject |

## Understanding Your Results

After running experiments, check:

```bash
# Example: ADAS1 baseline results
cd online-step-experiments/ADAS1/results_adas1/

# Best fitness found for each objective
cat score_NSGA3_1.csv
# Output: V0,V1,V2,V3,V4
#         -0.001,-0.001,0.005,0.016,0.0007

# Violations per constraint
cat reqs_NSGA3_1.csv
# Output: R0,R1,R2,conjunction
#         26,0,0,26

# All test cases evaluated (rows = evaluations, cols = variables)
head -5 X_all_evaluations_NSGA3_0.csv
# car_speed,orientation,p_x,p_y,road_shape,weather
# 28.02,−29,5.16,2.74,0,0
# ...

# All fitness scores (rows = evaluations, cols = objectives)
head -5 F_all_evaluations_NSGA3_0.csv
# V0,V1,V2,V3,V4
# −0.0026,−0.0026,0.0053,0.0164,0.0007
# ...
```

## Comparison Template

To compare subjects, create a comparison script:

```bash
#!/bin/bash

echo "BASELINE COMPARISON (PFES)"
echo "=========================="
echo ""

for subject in ADAS1 ADAS2 RR; do
    echo "$subject:"
    tail -1 online-step-experiments/$subject/results_${subject,,}/reqs_NSGA3_1.csv
    echo ""
done

echo ""
echo "HYBRID COMPARISON (PFES+SAMOTA)"
echo "==============================="
echo ""

for subject in ADAS1 ADAS2 RR; do
    echo "$subject:"
    tail -1 online-step-experiments/$subject/pfes_samota_baseline/reqs_NSGA3_1.csv
    echo ""
done
```

## For Your Thesis

### Experiment Plan

```
Phase 1: Verify Setup (Today)
  ├─ Run mini tests (--size 5 --niterations 1)
  └─ Check outputs are generated

Phase 2: Baseline Comparison (1-2 hours)
  ├─ 5 runs of PFES on each subject
  ├─ Generate statistics (mean, std, min, max violations)
  └─ Analyze constraint coverage

Phase 3: Hybrid Comparison (2-3 hours)
  ├─ 5 runs of PFES+SAMOTA on each subject
  ├─ Calculate efficiency (violations per evaluation)
  └─ Statistical significance testing

Phase 4: Analysis & Results
  ├─ Compare PFES vs PFES+SAMOTA
  ├─ Analyze scalability (6 vs 9 variables)
  ├─ Evaluate constraint impact (3 vs 6 constraints)
  └─ Write thesis results
```

### Key Metrics to Track

For each run, collect:

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Violations** | Count of constraint violations | Effectiveness |
| **Efficiency** | Violations / evaluations | Efficiency |
| **Coverage** | # constraints with ≥1 violation | Diversity |
| **Time** | Wall-clock seconds | Scalability |
| **Objectives** | Count of covered objectives | Completeness |

## Troubleshooting

### Import Error: "No module named 'mdp_simulator'"
```bash
# Make sure PYTHONPATH includes CPS-simulator
export PYTHONPATH=/path/to/CPS-simulator:$PYTHONPATH
```

### Error: "Matrix is singular" (RBF training)
- This is normal! The code handles it by skipping problematic clusters
- See `try/except` in PFES_SAMOTA.py around line 584

### Results look wrong or suspiciously good?
- Check violations are actually negative fitness values (< 0)
- Verify test case counts match expected evaluations
- Compare with known PFES baseline results

## Need More Info?

- **Full documentation**: See `SUBJECTS_README.md`
- **Implementation details**: See `EXTENSION_SUMMARY.md`
- **Code changes**: See comments in `PFES_SAMOTA.py` and `PFES_falsification.py`

## Summary

✅ **What's ready**:
- 3 subjects configured (ADAS1, ADAS2, RR)
- Config-driven code (works for any number of variables)
- All dependencies in place
- Identical code across subjects (only config differs)

✅ **What to do next**:
1. Run quick test to verify setup
2. Run baseline (PFES) experiments
3. Run hybrid (PFES+SAMOTA) experiments
4. Compare results across subjects
5. Write thesis with findings

✅ **Expected time**:
- Quick test: 5 minutes
- One baseline run (900 evals): 30 minutes
- One hybrid run: 40 minutes
- Full comparison (3 subjects × 2 algorithms × 5 runs): 6-8 hours

Good luck! 🚀

