# PFES+SAMOTA Detailed Logging Guide

## What Gets Logged

The enhanced PFES_SAMOTA.py now creates a detailed log file for each run:

```
pfes_samota_baseline/samota_detailed.log
```

## Questions You Can Now Answer

### 1. **Why does PFES+SAMOTA stop early?**

Look for these stopping messages in the log:

```
⏱️  TIME LIMIT REACHED at iteration X
💰 BUDGET EXHAUSTED at iteration X
✓ ALL OBJECTIVES COVERED - stopping iteration loop
```

**What to check:**
- Search for "STOPPED" or "REACHED" or "EXHAUSTED" or "COVERED"
- This tells you the EXACT stopping reason

### 2. **What objectives were actually violated?**

Search for "Objective status" section:

```
Objective status (min score per objective):
  V0: -0.002813 ✓ COVERED
  V1: -0.002813 ✓ COVERED
  V2:  0.002159 ✗ UNCOVERED
  V3:  0.016351 ✗ UNCOVERED
  V4: -0.000165 ✓ COVERED
```

**Negative = COVERED (violated)**
**Positive = UNCOVERED (not violated)**

### 3. **How many candidates were generated vs evaluated?**

Search for "GS:" and "LS:" lines:

```
GS: 12 candidates generated, 8 evaluated, 2 new violations
LS: 5 candidates generated, 3 evaluated, 0 new violations
```

**Questions answered:**
- Did budget limit stop evaluating candidates mid-phase?
- How many candidates were wasted (generated but not evaluated)?
- Which phase (GS or LS) found more violations?

### 4. **Budget allocation breakdown**

Search for these summary lines:

```
✓ Phase 1 complete: 300 ART evaluations, 8 violations found
--- ITERATION 1 ---
Current eval_count: 308/1800
GS: 12 candidates, 8 evals, 2 new violations
LS: 5 candidates, 3 evals, 0 new violations
--- ITERATION 2 ---
Current eval_count: 319/1800
...
ITERATION X COMPLETE: evals_used=X, total_evals=Y
```

**Calculate:**
- Phase 1 used: 300 evals
- Phase 2 used: (total_evals - 300)
- Why was remaining budget not used?

### 5. **Per-iteration progress**

Each iteration logs:

```
--- ITERATION 1 ---
Current eval_count: 308/1800
Elapsed time: 45.3s / 3600s
Uncovered objectives: [2, 3]
Running GS with uncovered objectives: [2, 3]
GS generated 12 candidates
GS: 12 candidates, 8 evals, 2 new violations, total violations: 10
Running LS with uncovered objectives: [2, 3]
LS generated 5 candidates
LS: 5 candidates, 3 evals, 0 new violations, total violations: 10
ITERATION 1 COMPLETE: evals_used=11, total_evals=319
```

**You can track:**
- Which objectives remained uncovered each iteration
- How candidates decreased as budget approached limit
- When algorithm decided to stop

## How to Read the Log

```bash
# View the entire log
cat pfes_samota_baseline/samota_detailed.log

# Search for stopping reason
grep -i "stopped\|reached\|exhausted\|covered" pfes_samota_baseline/samota_detailed.log

# See only Phase 1
grep -A5 "PHASE 1:" pfes_samota_baseline/samota_detailed.log

# See iteration summaries
grep "ITERATION\|COMPLETE" pfes_samota_baseline/samota_detailed.log

# See objective coverage
grep "Objective status" -A10 pfes_samota_baseline/samota_detailed.log

# Final results
tail -30 pfes_samota_baseline/samota_detailed.log
```

## When You Wake Up

1. **Run the experiments:**
   ```bash
   python3 run_10_comparative_runs.py --runs 10 --size 30 --niterations 30
   ```

2. **Check the logs for answers:**
   ```bash
   # Check first run's log
   cat results_10runs_samota/run_1/pfes_samota_baseline/samota_detailed.log

   # Get summary from all runs
   for i in {1..10}; do
     echo "=== RUN $i ==="
     grep "PHASE 1 complete\|STOPPED\|COVERED\|Objective Coverage" results_10runs_samota/run_$i/pfes_samota_baseline/samota_detailed.log
   done
   ```

3. **Analyze patterns:**
   - Do all 10 runs stop for the same reason?
   - Do different runs cover different objectives?
   - How consistent is the evaluation count across runs?

## Key Log Sections

### Start
```
================================================================================
PFES + SAMOTA DETAILED LOG
================================================================================
Budget: 1800 evaluations
Max iterations: 30
Constraints: 3
MINIMAL_CONSTRAINTS objectives: 5
```

### Phase 1
```
PHASE 1: ADAPTIVE RANDOM TESTING (ART)
ART population generated: 300 samples
...
✓ Phase 1 complete: 300 ART evaluations, 8 violations found
```

### Phase 2
```
PHASE 2: ITERATIVE GS + LS
Number of objectives: 5
--- ITERATION 1 ---
Current eval_count: 308/1800
Objective status (min score per objective):
  V0: ... ✓ COVERED
  V1: ... ✓ COVERED
  ...
```

### Results
```
================================================================================
PFES + SAMOTA FINAL RESULTS
================================================================================
Total Time: 2342.6 seconds
Total Evaluations: 640 / 1800 (35.6%)
Archive Size: 7

Requirements Breakdown:
  R0: 6 violations
  R1: 1 violations
  R2: 1 violations

Objective Coverage:
  V0: min_score=-0.002813 ✓ COVERED (negative)
  V1: min_score=-0.002813 ✓ COVERED (negative)
  V2: min_score=0.002159 ✗ UNCOVERED (non-negative)
  ...

Efficiency: 0.0203 violations/eval
```

## Summary

**You'll be able to answer:**
- ✅ Why PFES+SAMOTA stops early (exact stopping condition)
- ✅ Which objectives were violated vs not violated
- ✅ How many candidates were generated vs evaluated
- ✅ Why remaining budget wasn't used
- ✅ Iteration-by-iteration progress
- ✅ Statistical consistency across 10 runs

Sleep well! The logs will have all the details. 🌙
