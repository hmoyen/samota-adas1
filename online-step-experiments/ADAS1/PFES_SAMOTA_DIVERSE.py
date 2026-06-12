"""
PFES + SAMOTA + Diversity (SAMOTA-Diverse)
Pattern-Conditioned Global Search + Pattern-Aware Local Search

KEY DIFFERENCES from PFES_SAMOTA.py:
  1. PatternArchive: tracks which objectives are violated together per test case.
  2. GS: candidates scored by violation_potential * rarity_weight(estimated_pattern).
         Rare patterns get priority — breaks the V0+V1 co-violation dominance.
  3. LS: two-level clustering: by violation pattern first, then by parameter proximity.
         Budget allocated inversely proportional to pattern frequency.
         Rare patterns get proportionally more LS clusters.
  4. Main loop continues until budget is exhausted (not just until all objectives
     are individually covered), because diversity keeps looking for new pattern combos.

WHY THIS HELPS (observed from ADAS1/ADAS2 data):
  - PFES+SAMOTA finds V0+V1 in 88-99% of failures (co-violation bias).
  - V4-only is found in 10% of failures despite being reachable.
  - By weighting rare patterns higher in both GS and LS, we explicitly direct budget
    toward under-represented failure modes.

DOMAINS:
  This approach generalises beyond ADAS. The PatternArchive works for any system
  where failures can be characterised by which requirements/objectives are violated.
"""

import sys
sys.path.insert(0, '/home/lena/Downloads/icse2025_replication_package/online-step-experiments/ADAS1')

import numpy as np
import utils.helpers as helpers
import config as conf
import time
import json
from collections import defaultdict

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.mixed import (MixedVariableMating, MixedVariableSampling,
                               MixedVariableDuplicateElimination)
from pymoo.optimize import minimize
import hdbscan

from SAMOTA_ensemble import SAMOTAPerObjectiveEnsemble
from RBF import Model as RBF_Model

# ============================================================================
# CONFIGURATION
# ============================================================================
conf.MAX_STEPS = 20000
conf.BATCH_SIZE = 100
conf.MDP_FOLDER = "INPUT/AutonomousDriving_v1"
conf.PLOT = False
conf.MAX_SAMPLES = 100

PARAM_BOUNDS = {
    "car_speed":    (5.0,  50.0,  "real"),
    "p_x":          (0.0,  10.0,  "real"),
    "p_y":          (0.0,  10.0,  "real"),
    "orientation":  (-30,   30,   "int"),
    "weather":      (0,      2,   "int"),
    "road_shape":   (0,      2,   "int"),
}
LB = np.array([5.0, 0.0, 0.0, -30, 0, 0])
UB = np.array([50.0, 10.0, 10.0, 30, 2, 2])


def _make_pymoo_variables():
    variables = {}
    for name, (lo, hi, kind) in PARAM_BOUNDS.items():
        variables[name] = Real(bounds=(lo, hi)) if kind == "real" else Integer(bounds=(lo, hi))
    return variables


# ============================================================================
# SHARED HELPERS
# ============================================================================

def evaluate_test_case(params):
    """
    Evaluate a single test case. Returns constraint-mapped fitness scores.
    Same as PFES_SAMOTA.py — using region_scores() which are symmetric around 0.
    """
    from mdp_simulator import config, run, enums

    if conf.PLOT:
        config.OUTPUT_FOLDER_NAME = f"{conf.RQ}_{conf.EXTRA_NAME}"
        config.PLOT_DPI = 100
    config.MAX_PARTIAL_POSTERIOR_RETRIES = conf.HISTORY_RETRIES
    config.MAX_HISTORY_PARTIAL_POSTERIOR = conf.HISTORY_LEN
    config.FOLDER_NAME = conf.MDP_FOLDER
    config.BATCH_SIZE = conf.BATCH_SIZE
    config.DEBUG_LEVEL = enums.LogTypes.ERROR
    config.MAX_STEPS = conf.MAX_STEPS

    override_ss_variables = helpers.create_ss_variables(conf.SS_VARIABLES, params)
    mdp = run(override_ss_variables_starting_value=override_ss_variables)

    processed_scores = helpers.region_scores(mdp, conf.MINIMAL_CONSTRAINTS)
    reqs_satisfied = [helpers.check_requirement(mdp, c) for c in conf.CONSTRAINTS]

    scores = np.array(processed_scores)
    return scores, scores, reqs_satisfied


def params_to_dict(p):
    return {
        "car_speed":   float(p[0]),
        "p_x":         float(p[1]),
        "p_y":         float(p[2]),
        "orientation": int(p[3]),
        "weather":     int(p[4]),
        "road_shape":  int(p[5]),
    }


def dict_to_params(d):
    return np.array([d["car_speed"], d["p_x"], d["p_y"],
                     d["orientation"], d["weather"], d["road_shape"]])


def pymoo_to_params(x):
    return dict_to_params(x) if isinstance(x, dict) else np.array(x)


# ============================================================================
# PHASE 1: ART — Maximin sampling (unchanged from PFES_SAMOTA.py)
# ============================================================================

def art_initial_population(size=300):
    pop = []
    pop.append(np.random.uniform(LB, UB))

    for _ in range(size - 1):
        candidates = np.random.uniform(LB, UB, size=(500, len(LB)))
        pop_arr = np.array(pop)
        best, best_dist = None, -np.inf
        for c in candidates:
            d = np.min(np.linalg.norm(pop_arr - c, axis=1))
            if d > best_dist:
                best_dist = d
                best = c
        pop.append(best)

    pop = np.array(pop)
    return [params_to_dict(x) for x in pop], pop


# ============================================================================
# PATTERN ARCHIVE
# ============================================================================

class PatternArchive:
    """
    Tracks violation patterns (which objectives are violated) across all failures.

    pattern = tuple of 0/1 per objective: 1 if that objective was violated (F < 0).
    Only patterns with at least one violation are stored.

    Key operations:
      update(F_row)                    — register one evaluation
      rarity_weight(pattern)           — 1 / (count + 1); rare patterns get high weight
      get_indices_by_pattern(F_array)  — group row indices by their violation pattern
      budget_per_pattern(groups, N)    — allocate N units inversely proportional to group size
    """

    def __init__(self, n_objectives):
        self.n_objectives = n_objectives
        self.counts = defaultdict(int)

    def update(self, F_row):
        pat = tuple((np.asarray(F_row) < 0).astype(int))
        if any(pat):
            self.counts[pat] += 1

    def update_batch(self, F_array):
        for row in F_array:
            self.update(row)

    def rarity_weight(self, pattern):
        return 1.0 / (self.counts.get(tuple(pattern), 0) + 1)

    def get_indices_by_pattern(self, F_array):
        groups = defaultdict(list)
        for i, row in enumerate(F_array):
            pat = tuple((np.asarray(row) < 0).astype(int))
            if any(pat):
                groups[pat].append(i)
        return dict(groups)

    def budget_per_pattern(self, groups, total_budget):
        """
        Allocate total_budget units across patterns.
        Weight = 1 / group_size — rarest pattern gets the most.
        Returns dict: pattern -> int allocation.
        """
        if not groups:
            return {}

        weights = {pat: 1.0 / len(idx) for pat, idx in groups.items()}
        total_w = sum(weights.values())

        allocation = {}
        remaining = total_budget
        # Sort by weight descending (rarest first) so they get priority
        for pat in sorted(weights, key=lambda p: -weights[p]):
            alloc = max(1, round(total_budget * weights[pat] / total_w))
            alloc = min(alloc, remaining)
            allocation[pat] = alloc
            remaining -= alloc
            if remaining <= 0:
                break

        return allocation

    def summary(self):
        lines = ["  PatternArchive:"]
        for pat, cnt in sorted(self.counts.items(), key=lambda x: x[1]):
            names = "+".join(f"V{i}" for i, p in enumerate(pat) if p)
            lines.append(f"    {names or 'none'}: {cnt}")
        return "\n".join(lines)


# ============================================================================
# PATTERN-CONDITIONED GLOBAL SEARCH
# ============================================================================

class _GSProblem(ElementwiseProblem):
    def __init__(self, ensemble, **kwargs):
        super().__init__(vars=_make_pymoo_variables(), n_obj=1, **kwargs)
        self.ensemble = ensemble

    def _evaluate(self, x, out, *args, **kwargs):
        params = pymoo_to_params(x)
        pred, _ = self.ensemble.predict(params)
        out["F"] = np.array([pred])


def global_search_diverse(X_array, F_array, uncovered_objectives, pattern_archive,
                          pop_size=30, n_gen=20, n_candidates_per_obj=6):
    """
    Pattern-Conditioned Global Search.

    For each uncovered objective:
      1. Train per-objective surrogate ensemble (GP + Poly + RBF).
      2. Run NSGA3 to get a POOL of candidates (full population, not just Pareto front).
      3. For each candidate: predict all objective scores → estimate violation pattern.
      4. Diversity score = violation_potential * rarity_weight(estimated_pattern)
                         + 0.1 * uncertainty_bonus
      5. Return top n_candidates_per_obj per objective by diversity score.

    This breaks the co-violation bias: a candidate that would produce a rare pattern
    (e.g. V4-only) gets higher priority than one producing a common pattern (V0+V1).
    """
    if not uncovered_objectives:
        return []

    obj_names = ["V0", "V1", "V2", "V3", "V4"]
    n_objectives = F_array.shape[1]

    # Train surrogates for ALL objectives (needed to estimate full violation pattern)
    all_ensembles = {}
    for oi in range(n_objectives):
        name = obj_names[oi] if oi < len(obj_names) else f"V{oi}"
        all_ensembles[oi] = SAMOTAPerObjectiveEnsemble(
            X_array, F_array[:, oi], normalize=True, obj_name=name)

    selected_params = []

    for obj_idx in uncovered_objectives:
        # NSGA3 on the uncovered objective's surrogate
        problem = _GSProblem(all_ensembles[obj_idx])
        ref_dirs = get_reference_directions("das-dennis", 1, n_partitions=2)
        algorithm = NSGA3(
            ref_dirs=ref_dirs, pop_size=pop_size,
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(
                eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination()
        )
        res = minimize(problem, algorithm, ('n_gen', n_gen),
                       seed=obj_idx * 100, save_history=False, verbose=False)

        if res.X is None:
            continue

        pop_X = res.X if isinstance(res.X, list) else list(res.X)

        # Score each candidate by diversity-weighted violation potential
        scored = []
        for x in pop_X:
            params = pymoo_to_params(x)

            # Predict all objectives
            pred_scores = np.zeros(n_objectives)
            total_unc = 0.0
            for oi in range(n_objectives):
                pred, unc = all_ensembles[oi].predict(params)
                pred_scores[oi] = pred
                total_unc += unc

            # Estimated violation pattern (conservative: threshold 0.05)
            est_pattern = tuple((pred_scores < 0.05).astype(int))
            rarity = pattern_archive.rarity_weight(est_pattern)

            violation_potential = max(0.0, -pred_scores[obj_idx])
            uncertainty_bonus = total_unc / (n_objectives + 1e-9)

            diversity_score = (violation_potential + 0.1 * uncertainty_bonus) * rarity
            scored.append((diversity_score, params))

        scored.sort(key=lambda t: -t[0])
        for score, params in scored[:n_candidates_per_obj]:
            if score > 0:
                selected_params.append(params)

    # Remove near-duplicates
    unique_params = []
    for p in selected_params:
        if not any(np.allclose(p, e, atol=1e-3) for e in unique_params):
            unique_params.append(p)

    return [params_to_dict(p) for p in unique_params]


# ============================================================================
# PATTERN-AWARE LOCAL SEARCH
# ============================================================================

class _LSProblem(ElementwiseProblem):
    def __init__(self, rbf_model, **kwargs):
        super().__init__(vars=_make_pymoo_variables(), n_obj=1, **kwargs)
        self.rbf = rbf_model

    def _evaluate(self, x, out, *args, **kwargs):
        params = pymoo_to_params(x)
        res = self.rbf.predict(params.reshape(1, -1))
        pred = res[0] if isinstance(res, (list, np.ndarray)) else res
        out["F"] = np.array([pred])


def _ls_one_cluster(X_cluster, F_cluster, obj_idx, l_max, seed):
    """RBF surrogate + NSGA3 on one cluster for one objective. Returns params or None."""
    if len(X_cluster) < 2:
        return None

    train_data = [(X_cluster[i], F_cluster[i, obj_idx]) for i in range(len(X_cluster))]
    rbf = RBF_Model(n_neurons=10, train_data=train_data)

    problem = _LSProblem(rbf)
    alg = NSGA3(
        ref_dirs=get_reference_directions("das-dennis", 1, n_partitions=2),
        pop_size=6,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(
            eliminate_duplicates=MixedVariableDuplicateElimination()),
        eliminate_duplicates=MixedVariableDuplicateElimination()
    )
    result = minimize(problem, alg, ('n_gen', l_max), seed=seed, verbose=False)

    if result.X is not None and len(result.X) > 0:
        best_x = result.X[0] if isinstance(result.X, list) else result.X
        try:
            return pymoo_to_params(best_x)
        except (TypeError, KeyError, IndexError):
            return None
    return None


def local_search_diverse(X_all, F_all, uncovered_objectives, pattern_archive,
                         l_max=200, total_ls_clusters=20):
    """
    Pattern-Aware Local Search.

    Level 1: Group all failures by violation pattern.
             Rarest pattern → most cluster budget (inversely proportional to count).
    Level 2: Within each pattern group, cluster by parameter proximity (HDBSCAN).
    Per cluster: train RBF surrogate, run NSGA3 (same as PFES_SAMOTA LS).

    Example budget allocation with V0+V1: 433 failures, V4-only: 50 failures:
      V0+V1 weight = 1/433 = 0.0023
      V4   weight  = 1/50  = 0.020
      → V4 gets ~18 of 20 clusters, V0+V1 gets ~2.
    """
    n_objectives = F_all.shape[1]
    selected = []

    # Group failures by violation pattern
    groups = pattern_archive.get_indices_by_pattern(F_all)

    if not groups:
        # No failures yet — fall back to original uncovered-objective LS
        return _ls_fallback(X_all, F_all, uncovered_objectives, l_max)

    # Allocate cluster budget per pattern
    cluster_budget = pattern_archive.budget_per_pattern(groups, total_ls_clusters)

    print("    LS pattern budget:")
    for pat, n_clust in sorted(cluster_budget.items(), key=lambda t: t[1]):
        names = "+".join(f"V{i}" for i, p in enumerate(pat) if p)
        print(f"      {names}: {len(groups[pat])} failures → {n_clust} cluster(s)")

    seed_counter = 0

    for pattern, n_clust in cluster_budget.items():
        indices = np.array(groups[pattern])
        X_group = X_all[indices]
        F_group = F_all[indices]

        # Objectives to optimise: those actually violated in this pattern
        pattern_objs = [i for i, p in enumerate(pattern) if p]

        # HDBSCAN on this pattern group
        if len(X_group) >= 5:
            min_cs = max(2, len(X_group) // max(1, n_clust + 1))
            labels = hdbscan.HDBSCAN(
                min_cluster_size=min_cs, gen_min_span_tree=True
            ).fit_predict(X_group)
            unique_clusters = sorted(set(labels) - {-1})
        else:
            labels = np.zeros(len(X_group), dtype=int)
            unique_clusters = [0]

        clusters_done = 0
        for cl_idx in unique_clusters:
            if clusters_done >= n_clust:
                break

            mask = labels == cl_idx
            X_cl = X_group[mask]
            F_cl = F_group[mask]

            for obj_idx in pattern_objs:
                params = _ls_one_cluster(X_cl, F_cl, obj_idx, l_max, seed=seed_counter)
                seed_counter += 1
                if params is not None:
                    selected.append(params)

            clusters_done += 1

    return [params_to_dict(p) for p in selected]


def _ls_fallback(X_all, F_all, uncovered_objectives, l_max):
    """Original LS: top-20% per objective, HDBSCAN, RBF+NSGA3. Used before first failures."""
    selected = []
    for obj_idx in uncovered_objectives:
        sorted_idx = np.argsort(F_all[:, obj_idx])
        n_top = max(1, len(F_all) * 20 // 100)
        X_top = X_all[sorted_idx[:n_top]]
        F_top = F_all[sorted_idx[:n_top]]

        if len(X_top) >= 5:
            labels = hdbscan.HDBSCAN(
                min_cluster_size=5, gen_min_span_tree=True
            ).fit_predict(X_top)
            clusters = sorted(set(labels) - {-1})
        else:
            clusters = []

        for cl_idx in clusters:
            mask = labels == cl_idx
            params = _ls_one_cluster(X_top[mask], F_top[mask], obj_idx,
                                     l_max, seed=obj_idx * 1000 + cl_idx)
            if params is not None:
                selected.append(params)

    return [params_to_dict(p) for p in selected]


# ============================================================================
# MAIN ALGORITHM
# ============================================================================

def pfes_samota_diverse(max_iterations=30, max_time_seconds=3600, budget=900):
    """
    PFES + SAMOTA + Diversity on ADAS1.

    Phase 1: ART — 300 Maximin samples (same as PFES_SAMOTA).
    Phase 2: Pattern-conditioned GS + Pattern-aware LS until budget exhausted.
             Loop continues even after all objectives are individually covered,
             because diversity keeps targeting rare pattern combinations.
    """
    print("\n" + "="*80)
    print("PFES + SAMOTA + DIVERSITY")
    print("="*80)

    start_time = time.time()
    archive = []        # Failing test cases
    database_X = []
    database_F = []     # Constraint-mapped fitness (for surrogates + violation check)
    unsatisfied_reqs = [0] * len(conf.CONSTRAINTS)
    eval_count = 0

    # ========================================================================
    # PHASE 1: ART
    # ========================================================================
    print("\nPHASE 1: ART (Maximin sampling, 300 evaluations)")
    print("-" * 80)

    art_pop, _ = art_initial_population(size=300)

    for i, tc in enumerate(art_pop):
        if time.time() - start_time > max_time_seconds or eval_count >= budget:
            break

        params = [tc["car_speed"], tc["p_x"], tc["p_y"],
                  tc["orientation"], tc["weather"], tc["road_shape"]]
        scores, _, reqs_satisfied = evaluate_test_case(params)
        eval_count += 1

        x_arr = dict_to_params(tc)
        database_X.append(x_arr)
        database_F.append(scores)

        for ri, ok in enumerate(reqs_satisfied):
            if not ok:
                unsatisfied_reqs[ri] += 1

        if any(not r for r in reqs_satisfied):
            archive.append(tc)

        if (i + 1) % 50 == 0:
            print(f"  ART {i+1}/300 | evals: {eval_count} | violations: {len(archive)}")

    print(f"Phase 1 done: {eval_count} evals, {len(archive)} violations")

    X_array = np.array(database_X)
    F_array = np.array(database_F)

    # Initialise pattern archive from Phase 1 data
    n_objectives = F_array.shape[1]
    pattern_archive = PatternArchive(n_objectives)
    pattern_archive.update_batch(F_array)
    print(pattern_archive.summary())

    # ========================================================================
    # PHASE 2: ITERATIVE GS + LS (pattern-conditioned)
    # ========================================================================
    print("\nPHASE 2: Pattern-conditioned GS + Pattern-aware LS")
    print("-" * 80)

    for iteration in range(max_iterations):
        if time.time() - start_time > max_time_seconds or eval_count >= budget:
            print(f"Budget/time reached at iteration {iteration + 1}")
            break

        print(f"\nITERATION {iteration + 1} | evals: {eval_count}/{budget}")

        # Track uncovered objectives (not yet violated at all)
        min_per_obj = np.min(F_array, axis=0)
        uncovered_objectives = [i for i in range(n_objectives) if min_per_obj[i] >= 0]
        print(f"  Uncovered objectives: {uncovered_objectives or 'all covered'}")
        print(pattern_archive.summary())

        # ---- Global Search ----
        print("  GS (pattern-conditioned)...")
        gs_candidates = global_search_diverse(
            X_array, F_array, uncovered_objectives, pattern_archive,
            pop_size=30, n_gen=20, n_candidates_per_obj=6
        )

        for tc in gs_candidates:
            if eval_count >= budget:
                break
            params = [tc["car_speed"], tc["p_x"], tc["p_y"],
                      tc["orientation"], tc["weather"], tc["road_shape"]]
            scores, _, reqs_satisfied = evaluate_test_case(params)
            eval_count += 1

            x_arr = dict_to_params(tc)
            database_X.append(x_arr)
            database_F.append(scores)
            pattern_archive.update(scores)

            for ri, ok in enumerate(reqs_satisfied):
                if not ok:
                    unsatisfied_reqs[ri] += 1
            if any(not r for r in reqs_satisfied):
                archive.append(tc)

        X_array = np.array(database_X)
        F_array = np.array(database_F)
        print(f"    GS done: {len(gs_candidates)} candidates | {len(archive)} violations total")

        # ---- Local Search ----
        print("  LS (pattern-aware clustering)...")
        ls_candidates = local_search_diverse(
            X_array, F_array, uncovered_objectives, pattern_archive,
            l_max=200, total_ls_clusters=20
        )

        for tc in ls_candidates:
            if eval_count >= budget:
                break
            params = [tc["car_speed"], tc["p_x"], tc["p_y"],
                      tc["orientation"], tc["weather"], tc["road_shape"]]
            scores, _, reqs_satisfied = evaluate_test_case(params)
            eval_count += 1

            x_arr = dict_to_params(tc)
            database_X.append(x_arr)
            database_F.append(scores)
            pattern_archive.update(scores)

            for ri, ok in enumerate(reqs_satisfied):
                if not ok:
                    unsatisfied_reqs[ri] += 1
            if any(not r for r in reqs_satisfied):
                archive.append(tc)

        X_array = np.array(database_X)
        F_array = np.array(database_F)
        print(f"    LS done: {len(ls_candidates)} candidates | {len(archive)} violations total")
        print(f"    evals used: {eval_count}")

    # ========================================================================
    # RESULTS
    # ========================================================================
    elapsed = time.time() - start_time
    min_scores = np.min(F_array, axis=0)
    objectives_covered = int(np.sum(min_scores < 0))
    total_violations = int(np.sum(F_array < 0))

    print("\n" + "="*80)
    print("PFES + SAMOTA + DIVERSITY RESULTS")
    print("="*80)
    print(f"Time:              {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Evaluations:       {eval_count}")
    print(f"Archive size:      {len(archive)}")
    print(f"Objectives covered:{objectives_covered}/{n_objectives}")
    print(f"Total violations:  {total_violations}")
    print(f"Efficiency:        {total_violations/eval_count:.4f} v/eval")
    print(f"\nRequirements:")
    for i, cnt in enumerate(unsatisfied_reqs):
        print(f"  R{i}: {cnt} violations")
    print(f"\nMin scores per objective: {min_scores}")
    print(f"\nFinal pattern archive:")
    print(pattern_archive.summary())

    return {
        "elapsed": elapsed,
        "eval_count": int(eval_count),
        "archive_size": int(len(archive)),
        "violations": total_violations,
        "unsatisfied_reqs": [int(x) for x in unsatisfied_reqs],
        "objectives_covered": objectives_covered,
        "min_scores": [float(x) for x in min_scores.tolist()],
        "efficiency": float(total_violations / eval_count) if eval_count > 0 else 0.0,
        "pattern_counts": {
            "+".join(f"V{i}" for i, p in enumerate(pat) if p): cnt
            for pat, cnt in pattern_archive.counts.items()
        },
        # Raw arrays for CSV saving — excluded from JSON serialisation by runner
        "_X_array": X_array,
        "_F_array": F_array,
    }


if __name__ == "__main__":
    results = pfes_samota_diverse(max_iterations=30, budget=900)
    print("\nDone.")
    print(json.dumps(results, indent=2))
