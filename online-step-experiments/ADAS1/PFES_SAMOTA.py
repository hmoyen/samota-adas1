"""
PFES + SAMOTA Integration - HYBRID APPROACH
Per-objective surrogates + Multi-objective NSGA3 for Global Search

KEY FEATURES:
  Phase 1 (ART): Maximin sampling for diverse population
  Phase 2a (GS): PER-OBJECTIVE surrogates (GP + Polynomial + RBF) + MULTI-OBJECTIVE NSGA3
               = HYBRID: Specialized surrogates + holistic optimization with trade-off discovery
  Phase 2b (LS): NSGA3 with SINGLE RBF per cluster per objective

LOCAL SEARCH FIX (Session 8):
  - Previous: Just added noise to best point (BROKEN)
  - Now: Trains RBF surrogate per cluster, runs GA for 200 generations
  - This matches SAMOTA_CORRECTED_WITH_DISTINCTION implementation

DATA TRACKING:
  - database_F: Constraint-mapped fitness scores (for surrogates)
  - database_processed: Same constraint-mapped fitness (for violation detection)
  - Violation = processed_score < 0 (checked post-hoc, not by surrogates)
  - NOTE: Raw point_estimates approach (Session 9) performed 60% worse, reverted to constraint-mapped

Offline Phase: Parametric model checking → constraints
Online Phase: SAMOTA (Phase 1: ART + Phase 2: GS+LS with surrogates)
"""

import sys
import os
# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import utils.helpers as helpers
import config as conf
import pandas as pd
import time
import json
from scipy.spatial.distance import pdist, squareform
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
from sklearn.cluster import KMeans
import hdbscan

# Import per-objective ensemble (used in GS and LS)
from SAMOTA_ensemble import SAMOTAPerObjectiveEnsemble

# Import RBF model for Local Search (single RBF per cluster)
from RBF import Model as RBF_Model

# ============================================================================
# HELPER: Evaluate test case and return scaled + processed scores
# ============================================================================

def evaluate_test_case(params):
    """
    Evaluate a single test case using helpers.

    Returns:
    - surrogate_input: Constraint-mapped fitness scores (for surrogate training)
    - violation_check: Same constraint-mapped fitness (for violation detection < 0)
    - reqs_satisfied: boolean array of which constraints are satisfied

    NOTE: Using constraint-mapped values for both (not raw point_estimates)
    Rationale: Raw outputs approach tested in Session 9 showed 60% worse performance
    (15 violations vs 37 baseline). Constraint-mapped values are symmetric around
    violation boundary (0), making them easier for surrogates to model.
    """
    from mdp_simulator import config, run, enums

    # Run MDP simulator directly to get MDP object
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

    # Extract both raw and processed scores
    raw_estimates = helpers.raw_point_estimates(mdp, conf.MINIMAL_CONSTRAINTS)
    processed_scores = helpers.region_scores(mdp, conf.MINIMAL_CONSTRAINTS)

    # Check requirements
    reqs_satisfied = []
    for constraint in conf.CONSTRAINTS:
        reqs_satisfied.append(helpers.check_requirement(mdp, constraint))

    # Surrogates learn from PROCESSED scores (constraint-mapped, symmetric around 0)
    # This was tested: raw_point_estimates underperformed (15 viol vs 37 baseline)
    # Violation detection uses processed_scores (< 0 = violated)
    return np.array(processed_scores), np.array(processed_scores), reqs_satisfied


def scale_scores_for_surrogates(processed_scores):
    """
    Scale processed distance scores to expand value range.

    Original problem: distance scores have small range [-0.5, +0.5]
    Solution: Scale by 10 to get [-5, +5] range
    Benefit: Larger ranges help surrogates learn better!

    Preserves < 0 = violated property
    """
    return np.array(processed_scores) * 10  # Expand range 10x

# Configuration
THREADS_COUNT = 1
conf.MAX_STEPS = 20000
conf.BATCH_SIZE = 100
conf.MDP_FOLDER = "INPUT/AutonomousDriving_v1"
conf.PLOT = False
conf.MAX_SAMPLES = 100

# ============================================================================
# PHASE 1: ADAPTIVE RANDOM TESTING (ART) - Maximin Sampling
# ============================================================================

def generate_adaptive_random_population(size, lb, ub, n_candidates=100):
    """
    Generate diverse population using Maximin sampling (ART)
    - Start with 1 random sample
    - For each subsequent: generate N candidates, pick one farthest from all
    - Result: Maximally diverse set across parameter space
    """
    pop = []

    # First sample: random
    x0 = np.random.uniform(lb, ub)
    pop.append(x0)

    for _ in range(size - 1):
        # Generate candidate samples
        candidates = np.random.uniform(lb, ub, size=(n_candidates, len(lb)))

        if len(pop) > 0:
            # Find distances to nearest existing sample
            pop_array = np.array(pop)
            max_min_dist = -np.inf
            best_candidate = None

            for candidate in candidates:
                min_dist = np.min(np.linalg.norm(pop_array - candidate, axis=1))
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_candidate = candidate

            pop.append(best_candidate)

    return np.array(pop)


def art_initial_population(size=300):
    """
    Phase 1: ART - Adaptive Random Testing initialization
    Uses Maximin sampling for maximally diverse population
    """
    lb = np.array([5.0, 0.0, 0.0, -30, 0, 0])
    ub = np.array([50.0, 10.0, 10.0, 30, 2, 2])

    # Generate diverse population
    pop = generate_adaptive_random_population(size, lb, ub, n_candidates=500)

    # Convert to dict format for simulator
    test_cases = []
    for x in pop:
        test_cases.append({
            "car_speed": x[0],
            "p_x": x[1],
            "p_y": x[2],
            "orientation": int(x[3]),
            "weather": int(x[4]),
            "road_shape": int(x[5]),
        })

    return test_cases, pop


# ============================================================================
# PHASE 2: GLOBAL SEARCH - NSGA3 with Ensemble Surrogates
# ============================================================================

class GSPerObjectiveProblem(ElementwiseProblem):
    """
    DEPRECATED: Single-objective problem (old approach).
    Kept for reference - use GSMultiObjectivePerObjectiveSurrogateProblem instead.
    """
    def __init__(self, obj_ensemble, **kwargs):
        variables = {
            "car_speed": Real(bounds=(5.0, 50.0)),
            "p_x": Real(bounds=(0.0, 10.0)),
            "p_y": Real(bounds=(0.0, 10.0)),
            "orientation": Integer(bounds=(-30, 30)),
            "weather": Integer(bounds=(0, 2)),
            "road_shape": Integer(bounds=(0, 2)),
        }
        super().__init__(vars=variables, n_obj=1, **kwargs)
        self.obj_ensemble = obj_ensemble

    def _evaluate(self, x, out, *args, **kwargs):
        params = np.array([x["car_speed"], x["p_x"], x["p_y"],
                          x["orientation"], x["weather"], x["road_shape"]])
        pred, _ = self.obj_ensemble.predict(params)
        out["F"] = np.array([pred])


class GSMultiObjectivePerObjectiveSurrogateProblem(ElementwiseProblem):
    """
    HYBRID APPROACH: Multi-objective problem using PER-OBJECTIVE surrogates.

    Combines best of both worlds:
    - Surrogates: One ensemble per objective (specialized training)
    - Optimization: Multi-objective NSGA3 (finds trade-offs + maintains diversity via reference points)

    Returns n_objectives fitness values, one per objective.
    """
    def __init__(self, obj_ensembles_dict, uncovered_objectives, **kwargs):
        """
        Args:
            obj_ensembles_dict: Dictionary {obj_idx: SAMOTAPerObjectiveEnsemble}
            uncovered_objectives: List of objective indices to optimize
        """
        variables = {
            "car_speed": Real(bounds=(5.0, 50.0)),
            "p_x": Real(bounds=(0.0, 10.0)),
            "p_y": Real(bounds=(0.0, 10.0)),
            "orientation": Integer(bounds=(-30, 30)),
            "weather": Integer(bounds=(0, 2)),
            "road_shape": Integer(bounds=(0, 2)),
        }
        n_uncovered = len(uncovered_objectives)
        super().__init__(vars=variables, n_obj=n_uncovered, **kwargs)
        self.obj_ensembles_dict = obj_ensembles_dict
        self.uncovered_objectives = uncovered_objectives

    def _evaluate(self, x, out, *args, **kwargs):
        params = np.array([x["car_speed"], x["p_x"], x["p_y"],
                          x["orientation"], x["weather"], x["road_shape"]])

        # Get predictions from each per-objective surrogate
        F = []
        for obj_idx in self.uncovered_objectives:
            if obj_idx in self.obj_ensembles_dict:
                pred, _ = self.obj_ensembles_dict[obj_idx].predict(params)
                F.append(pred)
            else:
                F.append(np.inf)  # Penalty if surrogate not available

        out["F"] = np.array(F)


def global_search_nsga3(X_array, F_array, uncovered_objectives, pop_size=30, n_gen=20):
    """
    Phase 2a: Global Search - PER-OBJECTIVE surrogate + single-objective GA.

    For EACH uncovered objective:
      1. Train a separate SAMOTAPerObjectiveEnsemble (GP + Poly + RBF) on that objective's data
      2. Run single-objective NSGA3 to minimize predicted score for that objective
      3. From GA population, select:
         * Best (lowest predicted score) for this objective
         * Most uncertain (highest surrogate disagreement) for this objective
    Result: up to 2 x num_uncovered candidates (best + uncertain per objective)

    Mirrors LS design: per-objective surrogate, per-objective optimization.
    """
    if len(uncovered_objectives) == 0:
        return []

    from SAMOTA_ensemble import SAMOTAPerObjectiveEnsemble
    obj_names = ["V0", "V1", "V2", "V3", "V4"]
    selected_params = []

    for obj_idx in uncovered_objectives:
        # Train per-objective ensemble ONLY for this objective's data
        obj_ensemble = SAMOTAPerObjectiveEnsemble(
            X_array,
            F_array[:, obj_idx],
            normalize=True,
            obj_name=obj_names[obj_idx]
        )

        # Single-objective GA for this objective
        problem = GSPerObjectiveProblem(obj_ensemble)
        ref_dirs = get_reference_directions("das-dennis", 1, n_partitions=2)
        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination()
        )

        res = minimize(problem, algorithm, ('n_gen', n_gen),
                       seed=obj_idx * 100, save_history=False, verbose=False)

        if res.X is None:
            continue

        best_params = None
        best_score = np.inf
        uncertain_params = None
        best_uncertainty = -np.inf

        pop_X = res.X if isinstance(res.X, list) else list(res.X)
        for x in pop_X:
            # Handle both dict and array formats from pymoo
            if isinstance(x, dict):
                params = np.array([x["car_speed"], x["p_x"], x["p_y"],
                                   x["orientation"], x["weather"], x["road_shape"]])
            else:
                # x is a numpy array [car_speed, p_x, p_y, orientation, weather, road_shape]
                params = np.array(x)
            pred, unc = obj_ensemble.predict(params.reshape(1, -1))

            if pred < best_score:
                best_score = pred
                best_params = params

            if unc > best_uncertainty:
                best_uncertainty = unc
                uncertain_params = params

        if best_params is not None:
            selected_params.append(best_params)

        if uncertain_params is not None and not np.allclose(uncertain_params, best_params):
            selected_params.append(uncertain_params)

    # Remove duplicates across objectives
    unique_params = []
    for params in selected_params:
        if not any(np.allclose(params, existing) for existing in unique_params):
            unique_params.append(params)

    candidates = []
    for params in unique_params:
        candidates.append({
            "car_speed": float(params[0]),
            "p_x": float(params[1]),
            "p_y": float(params[2]),
            "orientation": int(params[3]),
            "weather": int(params[4]),
            "road_shape": int(params[5]),
        })

    return candidates


def global_search_hybrid(X_array, F_array, uncovered_objectives, pop_size=30, n_gen=20):
    """
    HYBRID APPROACH: Global Search with PER-OBJECTIVE surrogates + MULTI-OBJECTIVE NSGA3.

    Combines best of both approaches:
    1. Surrogate Training: Per-objective (specialized)
       - Each objective gets its own SAMOTAPerObjectiveEnsemble
       - Trained only on that objective's data
    2. Optimization: Multi-objective NSGA3 (holistic)
       - Optimizes ALL uncovered objectives simultaneously
       - Uses reference points for diversity across objective space
       - Finds solutions that trade-off between objectives

    Result: ~30 diverse candidates on Pareto front, then select best + uncertain per objective
            ~10 final candidates (2 per uncovered objective)

    This hybrid approach:
    - Exploits objective-specific patterns (per-objective training)
    - Discovers trade-off violations (NSGA3 multi-objective)
    - Maintains diversity (reference points)
    - More efficient than running separate GA per objective
    """
    if len(uncovered_objectives) == 0:
        return []

    from SAMOTA_ensemble import SAMOTAPerObjectiveEnsemble

    # Step 1: Train PER-OBJECTIVE ensembles (specialized)
    obj_names = ["V0", "V1", "V2", "V3", "V4"]
    obj_ensembles_dict = {}

    for obj_idx in uncovered_objectives:
        obj_ensemble = SAMOTAPerObjectiveEnsemble(
            X_array,
            F_array[:, obj_idx],
            normalize=True,
            obj_name=obj_names[obj_idx]
        )
        obj_ensembles_dict[obj_idx] = obj_ensemble

    # Step 2: Run MULTI-OBJECTIVE NSGA3 with per-objective surrogates
    problem = GSMultiObjectivePerObjectiveSurrogateProblem(
        obj_ensembles_dict,
        uncovered_objectives
    )

    # Create reference points for diversity across objective space
    n_uncovered = len(uncovered_objectives)
    ref_dirs = get_reference_directions("das-dennis", n_uncovered, n_partitions=3)

    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=pop_size,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
        eliminate_duplicates=MixedVariableDuplicateElimination()
    )

    res = minimize(problem, algorithm, ('n_gen', n_gen),
                   seed=0, save_history=False, verbose=False)

    if res.X is None or len(res.X) == 0:
        return []

    # Step 3: Select BEST + MOST UNCERTAIN per objective from Pareto front
    best_per_obj = {}  # obj_idx → (params, score)
    uncertain_per_obj = {}  # obj_idx → (params, uncertainty)

    pop_X = res.X if isinstance(res.X, list) else list(res.X)

    for x in pop_X:
        # Handle both dict and array formats from pymoo
        if isinstance(x, dict):
            params = np.array([x["car_speed"], x["p_x"], x["p_y"],
                               x["orientation"], x["weather"], x["road_shape"]])
        else:
            # x is a numpy array
            params = np.array(x)

        # For each objective, track best and most uncertain candidates
        for obj_idx in uncovered_objectives:
            if obj_idx not in obj_ensembles_dict:
                continue

            pred, unc = obj_ensembles_dict[obj_idx].predict(params)

            # Track BEST (lowest score)
            if obj_idx not in best_per_obj or pred < best_per_obj[obj_idx][1]:
                best_per_obj[obj_idx] = (params, pred)

            # Track MOST UNCERTAIN (highest disagreement)
            if obj_idx not in uncertain_per_obj or unc > uncertain_per_obj[obj_idx][1]:
                uncertain_per_obj[obj_idx] = (params, unc)

    # Collect selected candidates
    selected_params = []

    for obj_idx in uncovered_objectives:
        if obj_idx in best_per_obj:
            selected_params.append(best_per_obj[obj_idx][0])

        if obj_idx in uncertain_per_obj:
            params = uncertain_per_obj[obj_idx][0]
            # Only add if different from best
            if obj_idx not in best_per_obj or not np.allclose(params, best_per_obj[obj_idx][0]):
                selected_params.append(params)

    # Remove duplicates across objectives
    unique_params = []
    for params in selected_params:
        if not any(np.allclose(params, existing) for existing in unique_params):
            unique_params.append(params)

    # Convert to dict format
    candidates = []
    for params in unique_params:
        candidates.append({
            "car_speed": float(params[0]),
            "p_x": float(params[1]),
            "p_y": float(params[2]),
            "orientation": int(params[3]),
            "weather": int(params[4]),
            "road_shape": int(params[5]),
        })

    return candidates


# ============================================================================
# PHASE 2: LOCAL SEARCH - RBF-based single-objective optimization per cluster
# ============================================================================

class LSProblem(ElementwiseProblem):
    """Local Search: Uses SINGLE RBF surrogate per cluster (NOT ensemble!)"""

    def __init__(self, rbf_model, **kwargs):
        variables = {
            "car_speed": Real(bounds=(5.0, 50.0)),
            "p_x": Real(bounds=(0.0, 10.0)),
            "p_y": Real(bounds=(0.0, 10.0)),
            "orientation": Integer(bounds=(-30, 30)),
            "weather": Integer(bounds=(0, 2)),
            "road_shape": Integer(bounds=(0, 2)),
        }
        super().__init__(vars=variables, n_obj=1, **kwargs)
        self.rbf = rbf_model

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate using SINGLE RBF surrogate (trained on cluster data)"""
        params = np.array([x["car_speed"], x["p_x"], x["p_y"],
                          x["orientation"], x["weather"], x["road_shape"]])
        # Single RBF prediction (NOT ensemble)
        pred_result = self.rbf.predict(params.reshape(1, -1))
        # Handle both array and scalar returns
        pred = pred_result[0] if isinstance(pred_result, (list, np.ndarray)) else pred_result
        out["F"] = np.array([pred])


def local_search_phase(X_all, F_all, uncovered_objectives, eta_percent=20, l_max=200, n_clusters=20):
    """
    Phase 2b: Local Search with CLUSTER-SPECIFIC RBF surrogates (NOT ensemble!)

    Paper's Algorithm 4:
    - For EACH objective
    - Filter to top η% (20%) individuals for that objective
    - Cluster those top individuals (k=20)
    - For EACH cluster: train SINGLE RBF on cluster data (NOT ensemble!)
    - Run GA with cluster's RBF (200 gens)
    - Return best from cluster
    """
    selected = []

    for obj_idx in uncovered_objectives:
        # PAPER ALGORITHM 4 LINE 3: Select top η% individuals for this objective
        # Sort by fitness score for objective u, select top η%
        sorted_indices = np.argsort(F_all[:, obj_idx])
        n_top = max(1, int(len(F_all) * eta_percent / 100))
        top_indices = sorted_indices[:n_top]  # Best (lowest scores) first

        X_top = X_all[top_indices]
        F_top = F_all[top_indices]

        # Cluster the top individuals using HDBSCAN with minimum cluster size of 5
        if len(X_top) > 1:
            # HDBSCAN: automatic cluster detection, no need to specify k
            # min_cluster_size=5: Only clusters with ≥5 points are valid
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
            cluster_labels = clusterer.fit_predict(X_top)

            # Get unique cluster labels (excluding -1 for noise points)
            unique_clusters = set(cluster_labels)
            if -1 in unique_clusters:
                unique_clusters.remove(-1)  # Remove noise label
            cluster_indices = sorted(list(unique_clusters))
        else:
            cluster_indices = []

        for cluster_idx in cluster_indices:
            # Get cluster members
            mask = cluster_labels == cluster_idx
            X_cluster = X_top[mask]
            F_cluster = F_top[mask]

            # HDBSCAN already enforces min_cluster_size=5, but double-check
            if len(X_cluster) < 5:
                continue

            # ✓ Create SINGLE RBF per cluster per objective (NOT ensemble!)
            # RBF_Model expects: train_data = [(X, y), (X, y), ...]
            train_data = [(X_cluster[i], F_cluster[i, obj_idx]) for i in range(len(X_cluster))]
            local_surrogate = RBF_Model(n_neurons=10, train_data=train_data)

            # Create LS problem with cluster's RBF surrogate
            problem = LSProblem(local_surrogate)
            algorithm_ls = NSGA3(
                ref_dirs=get_reference_directions("das-dennis", 1, n_partitions=2),
                pop_size=6,
                sampling=MixedVariableSampling(),
                mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                eliminate_duplicates=MixedVariableDuplicateElimination()
            )

            result = minimize(problem, algorithm_ls, ('n_gen', l_max),
                            seed=obj_idx * 1000 + cluster_idx, verbose=False)

            if result.X is not None and len(result.X) > 0:
                # Get the first (best) solution
                best_x = result.X[0] if isinstance(result.X, list) else result.X

                # Handle dict-like access for mixed variables
                try:
                    params = np.array([best_x["car_speed"], best_x["p_x"], best_x["p_y"],
                                      best_x["orientation"], best_x["weather"], best_x["road_shape"]])
                    selected.append(params)
                except (TypeError, KeyError):
                    # If dict access fails, try positional access
                    try:
                        params = np.array([best_x[0], best_x[1], best_x[2], best_x[3], best_x[4], best_x[5]])
                        selected.append(params)
                    except (TypeError, IndexError):
                        pass  # Skip if we can't extract parameters

    # Convert numpy array params to dict format for rest of PFES_SAMOTA
    candidates = []
    for params in selected:
        candidates.append({
            "car_speed": float(params[0]),
            "p_x": float(params[1]),
            "p_y": float(params[2]),
            "orientation": int(params[3]),
            "weather": int(params[4]),
            "road_shape": int(params[5]),
        })

    return candidates


# ============================================================================
# MAIN PFES+SAMOTA ALGORITHM
# ============================================================================

def pfes_samota(max_iterations=30, max_time_seconds=3600, budget=1800):
    """
    PFES + SAMOTA Integration on ADAS1

    Offline: Parametric model checking (skipped - use constraints)
    Online:
      Phase 1: ART (Adaptive Random Testing)
      Phase 2: SAMOTA (GS + LS iterations)

    Returns: Archive A, Database D, metrics
    """

    print("\n" + "="*80)
    print("PFES + SAMOTA INTEGRATION")
    print("="*80)

    start_time = time.time()
    archive = []  # Test cases that violate at least one requirement
    database = []  # All evaluated test cases
    database_X = []  # Parameters
    database_F = []  # Fitness values (RAW simulator outputs - larger range!)
    database_processed = []  # Processed scores (for tracking violations)

    # Track violations per requirement (like PFES baseline: R0, R1, R2)
    unsatisfied_reqs = [0] * len(conf.CONSTRAINTS)  # [R0, R1, R2]

    eval_count = 0

    # ========================================================================
    # PHASE 1: ADAPTIVE RANDOM TESTING (ART)
    # ========================================================================

    print("\nPHASE 1: ADAPTIVE RANDOM TESTING (ART) - Maximin Sampling")
    print("-" * 80)

    art_pop, art_X = art_initial_population(size=300)

    for i, test_case in enumerate(art_pop):
        if time.time() - start_time > max_time_seconds or eval_count >= budget:
            break

        # Evaluate via simulator - get RAW outputs (not distances!)
        params = [
            test_case["car_speed"], test_case["p_x"], test_case["p_y"],
            test_case["orientation"], test_case["weather"], test_case["road_shape"]
        ]
        raw_estimates, processed_scores, reqs_satisfied = evaluate_test_case(params)

        eval_count += 1

        # Store in database (using RAW estimates for surrogates!)
        x_array = np.array([test_case["car_speed"], test_case["p_x"], test_case["p_y"],
                           test_case["orientation"], test_case["weather"], test_case["road_shape"]])
        database.append(test_case)
        database_X.append(x_array)
        database_F.append(raw_estimates)  # ← Constraint-mapped fitness (processed_scores)
        database_processed.append(processed_scores)  # ← Processed scores (for tracking violations)

        # Track violations per requirement
        for req_idx, req_satisfied in enumerate(reqs_satisfied):
            if not req_satisfied:
                unsatisfied_reqs[req_idx] += 1

        # Update archive if violates any requirement
        if any(not r for r in reqs_satisfied):
            archive.append(test_case)

        if (i + 1) % 50 == 0:
            print(f"  Evaluated {i+1}/300 ART samples... (evals: {eval_count})")

    print(f"✓ Phase 1 complete: {len(database)} evaluations, {len(archive)} violations found")

    X_array = np.array(database_X)
    F_array = np.array(database_F)  # RAW estimates (for surrogates)
    F_processed = np.array(database_processed)  # Processed scores (for violation checking)

    # ========================================================================
    # PHASE 2: ITERATIVE GS + LS
    # ========================================================================

    print("\nPHASE 2: ITERATIVE GS + LS")
    print("-" * 80)

    n_objectives = len(database_F[0]) if database_F else len(conf.CONSTRAINTS)
    uncovered_objectives = list(range(n_objectives))

    for iteration in range(max_iterations):
        if time.time() - start_time > max_time_seconds or eval_count >= budget:
            print(f"✓ Budget or time limit reached at iteration {iteration}")
            break

        print(f"\nITERATION {iteration + 1}")

        # Update uncovered objectives (those without violations yet)
        # Use PROCESSED scores to check violations (< 0 = violated)
        min_per_obj_processed = np.min(F_processed, axis=0)
        uncovered_objectives = [i for i in range(n_objectives) if min_per_obj_processed[i] >= 0]

        if len(uncovered_objectives) == 0:
            print("✓ All objectives covered!")
            break

        print(f"  Uncovered objectives: {uncovered_objectives}")

        # ====================================================================
        # Global Search Phase (PER-OBJECTIVE: one surrogate+GA per uncovered objective)
        # ====================================================================
        # PER-OBJECTIVE APPROACH: Mirrors LS design
        # - Surrogates: Per-objective (specialized training, same as LS)
        # - Optimization: Single-objective NSGA3 per uncovered objective (same as LS)
        # - Result: Focused search per objective (not multi-obj trade-offs)

        print(f"  Global Search (GS) - PER-OBJECTIVE (one surrogate+GA per uncovered obj)...")
        gs_candidates = global_search_nsga3(X_array, F_array, uncovered_objectives,
                                            pop_size=30, n_gen=20)

        for test_case in gs_candidates:
            if eval_count >= budget:
                break

            # Evaluate and get RAW simulator outputs (not distances!)
            params = [
                test_case["car_speed"], test_case["p_x"], test_case["p_y"],
                test_case["orientation"], test_case["weather"], test_case["road_shape"]
            ]
            raw_estimates, processed_scores, reqs_satisfied = evaluate_test_case(params)

            eval_count += 1

            x_array = np.array([test_case["car_speed"], test_case["p_x"], test_case["p_y"],
                               test_case["orientation"], test_case["weather"], test_case["road_shape"]])
            database.append(test_case)
            database_X.append(x_array)
            database_F.append(raw_estimates)  # ← Constraint-mapped fitness (processed_scores)
            database_processed.append(processed_scores)  # ← Track processed scores for violations

            # Track violations per requirement
            for req_idx, req_satisfied in enumerate(reqs_satisfied):
                if not req_satisfied:
                    unsatisfied_reqs[req_idx] += 1

            if any(not r for r in reqs_satisfied):
                archive.append(test_case)

        X_array = np.array(database_X)
        F_array = np.array(database_F)  # RAW estimates for surrogates
        F_processed = np.array(database_processed)  # Processed for violation checks

        print(f"    GS: {len(gs_candidates)} candidates, {len(archive)} total violations")

        # ====================================================================
        # Local Search Phase
        # ====================================================================

        print(f"  Local Search (LS)...")
        ls_candidates = local_search_phase(X_array, F_array, uncovered_objectives, eta_percent=20, l_max=200, n_clusters=20)

        for test_case in ls_candidates:
            if eval_count >= budget:
                break

            # Evaluate and get RAW simulator outputs (not distances!)
            params = [
                test_case["car_speed"], test_case["p_x"], test_case["p_y"],
                test_case["orientation"], test_case["weather"], test_case["road_shape"]
            ]
            raw_estimates, processed_scores, reqs_satisfied = evaluate_test_case(params)

            eval_count += 1

            x_array = np.array([test_case["car_speed"], test_case["p_x"], test_case["p_y"],
                               test_case["orientation"], test_case["weather"], test_case["road_shape"]])
            database.append(test_case)
            database_X.append(x_array)
            database_F.append(raw_estimates)  # ← Constraint-mapped fitness (processed_scores)
            database_processed.append(processed_scores)  # ← Track processed scores for violations

            # Track violations per requirement
            for req_idx, req_satisfied in enumerate(reqs_satisfied):
                if not req_satisfied:
                    unsatisfied_reqs[req_idx] += 1

            if any(not r for r in reqs_satisfied):
                archive.append(test_case)

        X_array = np.array(database_X)
        F_array = np.array(database_F)  # RAW estimates
        F_processed = np.array(database_processed)  # Processed scores

        print(f"    LS: {len(ls_candidates)} candidates, {len(archive)} total violations")
        print(f"    Total evals: {eval_count}")

    # ========================================================================
    # RESULTS
    # ========================================================================

    elapsed = time.time() - start_time
    # Use PROCESSED scores to count violations (since raw estimates don't have violation info)
    min_scores_processed = np.min(F_processed, axis=0)
    objectives_covered = np.sum(min_scores_processed < 0)
    violations = np.sum(F_processed < 0)
    min_scores = min_scores_processed  # For reporting

    print("\n" + "="*80)
    print("PFES + SAMOTA RESULTS")
    print("="*80)
    print(f"Total Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Total Evaluations: {eval_count}")
    print(f"Archive Size: {len(archive)}")
    print(f"\nRequirements Breakdown (like PFES baseline):")
    for i, req_violations in enumerate(unsatisfied_reqs):
        print(f"  R{i}: {req_violations} violations")
    print(f"\nTotal Violations: {violations}")
    print(f"Objectives Covered: {objectives_covered}/{n_objectives}")
    print(f"Min Scores: {min_scores}")
    print(f"Efficiency: {violations/eval_count:.4f} violations/eval")

    return {
        'elapsed': elapsed,
        'eval_count': int(eval_count),
        'archive_size': int(len(archive)),
        'violations': int(violations),
        'unsatisfied_reqs': [int(x) for x in unsatisfied_reqs],  # ← R0, R1, R2 breakdown
        'objectives_covered': int(objectives_covered),
        'min_scores': [float(x) for x in min_scores.tolist()],
        'efficiency': float(violations / eval_count),
    }


if __name__ == "__main__":
    results = pfes_samota(max_iterations=30, budget=1800)

    print("\n✓ Test completed!")
    print(json.dumps(results, indent=2))
