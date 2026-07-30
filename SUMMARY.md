# Summary: P0/P1 fixes from the research committee review

Reanalysis and code fixes implementing the P0/P1 items from
`RESEARCH_COMMITTEE_REVIEW.md`. No simulator or multi-hour experiment was run as part of this
work; `results/`/`results_diversity/` were not modified. Each item below was committed
separately; see `git log` for the full set of commits.

## §1-3: analyze_all_results.py (1 file, 1 commit — `cac5970`)

- **§1 Coverage significance testing**: `run_statistical_tests` now runs the same Mann-Whitney
  U + Vargha-Delaney A_12 pipeline (reference = SAMOTA) on `coverage_per_run` and a new
  `full_coverage_per_run` binary indicator, alongside the existing violation-count tests.
- **§2 Multiple-comparison correction + CIs**: Holm-Bonferroni step-down correction applied
  across each per-benchmark family of SAMOTA-vs-X comparisons (raw and corrected p-values both
  reported). Percentile bootstrap 95% CIs (10000 resamples, fixed seed) added to every reported
  mean, median, and A_12.
- **§3 Kruskal-Wallis omnibus**: Per-benchmark omnibus test across all six algorithms, printed
  before the pairwise comparisons, for both violation count and coverage. When
  `scipy.stats.kruskal` raises `ValueError` (e.g. RR's `full_coverage_per_run` is all-zero for
  every algorithm — nobody achieves full coverage there in the `results/` data — so every value
  is identical), the reason is surfaced in the summary output instead of being reported as
  "insufficient data".

## §4: GP surrogate seeding (3 copies, 2 commits — `41a77ad`, `98ce400`)

`SAMOTA_ensemble.py` (ADAS1/ADAS2/RR, byte-identical) hardcoded `random_state=42` for every
`GaussianProcessRegressor`, including on `retrain()`, so every run/retrain used the same GP
seed regardless of the experiment's actual seed. Both classes now take a `seed` parameter
threaded from the CLI `--seed` through `pfes_samota()` → `global_search_nsga3()` → each
per-objective ensemble (offset by `obj_idx` so objectives don't share a seed), matching the
pattern already used for that same loop's NSGA3 `minimize(seed=...)` call.

Separately (own commit `98ce400`), restored a pre-existing but uncommitted one-line fix from an
earlier session (`np.random.seed(seed)` at the top of Phase 1/ART in `pfes_samota()`) that had
been sitting unstaged in the same files — split out so it wouldn't be conflated with the GP
seeding change.

## §5: Ensemble uncertainty usage — findings only, no code changed

Traced the live call path (`pfes_samota()` → `global_search_nsga3()`, the only GS entry point
actually called — `global_search_hybrid()` and `GSMultiObjectivePerObjectiveSurrogateProblem`
are dead code, never invoked). Finding: `uncertainty` (`np.std(all_preds, axis=1)` across the
GP/Poly/RBF ensemble) **is** consumed, but only for post-hoc candidate selection after the
NSGA3 GA finishes — it picks the "best" and "most uncertain" points per objective from the
GA's final population. It is **not** used during the GA's fitness evaluation itself (NSGA3
optimizes on the ensemble's point prediction only, unweighted by uncertainty). The entirely
unused `SAMOTAGlobalSurrogates` class and `global_search_hybrid()` function were left alone
(dead code, out of scope for this task).

## §6: RBF fallback exception logging (3 copies, 1 commit — `3acb846`)

`RBF.py`'s `Model.train()` silently fell back from `multiquadric` to `thin_plate` on any
exception. The `except Exception` now logs `type(e).__name__` and `str(e)` before falling back;
fallback behavior is unchanged.

## §7: CI workflow (1 new file, 1 commit — `c676839`)

Added `.github/workflows/tests.yml`: runs `pytest tests/` on push/PR, Python 3.11, with the
same pinned dependency versions used everywhere else in this project. The opt-in
`--run-slow` simulator smoke tests are intentionally excluded from the default CI run (each
spawns a real, if short, simulator process).

## §8: Scaffold-only scripts (written and compile/`--help`-checked, NOT run)

- **8a.** `online-step-experiments/RR/reachability_check.py` — argues about whether RR's two
  never-violated requirements (R4, R5 in `results/`) are genuinely unreachable or just
  unexplored, by evaluating a bounded, targeted set of points (baseline + one-at-a-time
  min/max corners, optionally full corners or extra random points) through the real simulator.
  A found violation proves reachability; "not reached" is explicitly reported as inconclusive,
  not proof of unreachability. Run with:

  ```bash
  cd online-step-experiments/RR
  python3.11 reachability_check.py --req_indices 4,5 --strategy corners --random_samples 50
  ```

- **8b.** `noise_study.py` (identical copy in ADAS1/ADAS2/RR) — quantifies simulator estimation
  noise by calling `run_mdp(x)` N times on a single fixed input and reporting per-region-score
  variance and per-requirement pass/fail flip rate. Motivation: `action.get_expected()` is a
  posterior point estimate, not closed-form, so two calls on the same input aren't guaranteed
  identical — this affects every "30 independent runs" statistical comparison in the project.
  Run with:

  ```bash
  cd online-step-experiments/<ADAS1|ADAS2|RR>
  python3.11 noise_study.py --n 30
  ```

- **8c.** `--equalize_budget` flag added to `FOC_falsification.py` (ADAS1/ADAS2/RR). FF's
  focused-testing phase currently spends `BUDGET // NREQS` per requirement regardless of how
  much the preceding sensitivity phase already spent, so FF's real total exceeds the nominal
  budget every other algorithm uses (quantified for ADAS1: ~240 sensitivity evals + ~900
  focused evals ≈ 1140 vs. nominal 900). The new flag (default `False`, existing behavior
  unchanged) truncates the focused-test budget to what's actually left, using the
  pre-existing `shared_eval_count` counter. Not run as part of this change; to use it:

  ```bash
  cd online-step-experiments/ADAS1
  python3.11 FOC_falsification.py --equalize_budget true --nruns 30
  ```

## §9: Docs (2 files + pyproject.toml, 2 commits — `75fda13`, plus the SESSION_HANDOFF.md note)

- `README.md` rewritten to cover all three benchmarks (it previously described the repo as
  ADAS1-only, a leftover from before ADAS2/RR were added), to give a working install path, and
  to point at `analyze_all_results.py`/`results/` instead of a stale hardcoded expected-results
  table.
- `pyproject.toml`'s dependency versions had drifted from what the code actually needs
  (`pandas = "^2.0"` is incompatible with `arviz==0.14.0`; `arviz`/`colorama`/`termcolor`/
  `pymdptoolbox`/`reportlab` were missing entirely), so `poetry install` alone produced a
  broken environment. Pinned to the same versions used in CI.
- `SESSION_HANDOFF.md` got a top-of-file note pointing to `ANALYST_BRIEF.md` as the more
  current source of truth. Per the task's instruction not to delete it unilaterally, it was
  left in place — whether to delete it is an open question for the user.

## Verification

`python3.11 -m pytest tests/ -q` was run after every change in this list and stayed green
(55 passed, 12 skipped — the 12 skips are the opt-in `--run-slow` simulator smoke tests).

## Out of scope (per task instructions, not attempted)

Refactoring the duplicated `PFES_SAMOTA.py`/`RBF.py`/`SAMOTA_ensemble.py` copies across
benchmarks; changing the core statistical test choice, run count, or any seed-fairness fix
beyond §4; new baselines or acquisition-function variants; prose about novelty, related work,
or venue framing.
