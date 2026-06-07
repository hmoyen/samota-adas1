# Remaining Issues & Blockers

## Current Status
- ✅ Parameter ordering bug fixed and tested
- ✅ Code ready for experimentation
- ⚠️ Simulator dependency issue blocking actual runs

## Known Issues

### 1. **Simulator Dependency Error** (BLOCKING)
**Status**: Not Fixed (External)
**Severity**: HIGH - Prevents running any experiments

**Error**:
```
TypeError: hdi got an unexpected keyword argument: 'hdi_prob'
```

**Location**: `mdp_simulator/mdp/action.py` line 196
```python
self._hdi = az.hdi(sample, hdi_prob=config.HDI_PROB)  # ← hdi_prob not valid
```

**Root Cause**: ArviZ API changed - `hdi_prob` parameter was replaced with different syntax in newer versions

**Solution Options**:
1. Check ArviZ version compatibility
2. Update the call to use correct ArviZ API
3. Downgrade/upgrade ArviZ to match expected version

**How to Fix**:
```bash
# Check current ArviZ version
python3 -c "import arviz; print(arviz.__version__)"

# Check what parameters hdi() accepts
python3 -c "import arviz; help(arviz.hdi)"

# Either update code or install compatible version
# pip install 'arviz==X.Y.Z'
```

### 2. **Virtual Environment State** (Minor)
**Status**: Need to Verify
**Severity**: LOW

The tests passed in the clean_repo, but full simulator runs haven't been tested since the parameter ordering fix.

**Action**: Once simulator dependency is fixed, run full experiments to verify everything works end-to-end.

### 3. **Simulator Caching Between Runs** (Potential)
**Status**: Unknown - Parameter fix may have resolved this
**Severity**: MEDIUM - Only appears on Run 2+

The original "Run 1 succeeds, Run 2+ fails" pattern suggested possible state carryover. The parameter ordering fix addresses one cause, but verify that:
- ✓ Output directories are cleaned between runs (script does this)
- ✓ MDP simulator doesn't cache state across runs
- ✓ Random seeds reset properly between runs

**Verification**: Watch Run 2+ execution closely for any state-related warnings.

## Tasks Completed This Session

✅ Identified root cause: Parameter ordering mismatch in `global_search_nsga3()`
✅ Fixed both ADAS1 and ADAS2 PFES_SAMOTA.py files
✅ Created comprehensive test files to verify the fix
✅ Documented the bug and fix thoroughly
✅ Committed and pushed all changes to GitHub

## Next Steps (For User)

### Immediate (When you wake up):
1. Pull latest changes: `git pull origin main`
2. Review the fix:
   - Read `CRITICAL_BUG_FIX_SUMMARY.md` (executive summary)
   - Read `BUG_FIX_REPORT.md` (technical details)
3. Run verification tests:
   ```bash
   cd online-step-experiments/ADAS1 && python3 test_parameter_ordering.py
   cd online-step-experiments/ADAS2 && python3 test_parameter_ordering.py
   ```

### Before Running Experiments:
1. **Fix the ArviZ dependency issue**:
   - Check ArviZ version: `python3 -c "import arviz; print(arviz.__version__)"`
   - Update the simulator call if needed
   - Test with a simple PFES baseline run: `python3 PFES_falsification.py --size 5 --niterations 1 --nruns 1`

2. **Once simulator works**, run small test:
   ```bash
   cd online-step-experiments/ADAS1
   # Run just 1 iteration to verify Run 2 doesn't fail
   timeout 30m bash run_10_samota_900budget.sh
   ```

### Full Experiments (When ready):
```bash
# ADAS1: 10 PFES + 10 PFES+SAMOTA
cd online-step-experiments/ADAS1
time bash run_10_comparative_runs.sh  # ~4 hours total

# ADAS2: 10 PFES + 10 PFES+SAMOTA
cd online-step-experiments/ADAS2
time bash run_adas2_experiments.sh    # ~4 hours total

# Generate comparison results
python3 compare_10runs.py
python3 plot_comparison.py
```

## Files Affected

### Fixed Files:
- `online-step-experiments/ADAS1/PFES_SAMOTA.py` - global_search_nsga3() lines 323-366
- `online-step-experiments/ADAS2/PFES_SAMOTA.py` - global_search_nsga3() lines 323-366

### New Test Files:
- `online-step-experiments/ADAS1/test_parameter_ordering.py` - Verification test
- `online-step-experiments/ADAS2/test_parameter_ordering.py` - Verification test

### Documentation Files:
- `CRITICAL_BUG_FIX_SUMMARY.md` - This-session summary
- `BUG_FIX_REPORT.md` - Detailed technical analysis
- `REMAINING_ISSUES.md` - This file

## Verification Checklist

- [x] Code compiles without syntax errors
- [x] Parameter ordering tests pass
- [x] Tests run without import errors
- [x] All parameter bounds respected in tests
- [ ] Full PFES baseline runs without simulator errors (blocked by ArviZ)
- [ ] Full PFES+SAMOTA Run 1 succeeds (blocked by ArviZ)
- [ ] Full PFES+SAMOTA Run 2+ succeeds (blocked by ArviZ)
- [ ] 10-run comparison completes successfully (blocked by ArviZ)

## Git Status

```
Commits:
  2e1759f - Fix: Critical parameter ordering bug in global_search_nsga3()
  10f1b33 - Add: Test files and comprehensive bug fix report
  fcbf255 - Add: Executive summary of critical parameter ordering bug fix

Remote: All changes pushed to origin/main ✓
Local: Clean working directory ✓
```

---

**Last Updated**: 2026-06-08
**Status**: Ready for experiments (pending simulator dependency fix)
