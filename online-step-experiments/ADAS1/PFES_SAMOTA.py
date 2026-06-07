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
# HELPER: Build pymoo variables dynamically from config
# ============================================================================

def build_pymoo_variables():
    """Build pymoo variable definitions from config.SS_VARIABLES"""
    from pymoo.core.variable import Real, Integer
    variables = {}
    for var_name in sorted(conf.SS_VARIABLES.keys()):
        var_config = conf.SS_VARIABLES[var_name]
        bounds = tuple(var_config["range"])
        if var_config["domain"] == float:
            variables[var_name] = Real(bounds=bounds)
        else:  # int
            variables[var_name] = Integer(bounds=bounds)
    return variables

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
    Bounds are read from config.SS_VARIABLES
    """
    # Build bounds from config (respects any subject's variable ranges)
    var_names = sorted(conf.SS_VARIABLES.keys())
    lb = np.array([conf.SS_VARIABLES[var]["range"][0] for var in var_names])
    ub = np.array([conf.SS_VARIABLES[var]["range"][1] for var in var_names])

    # Generate diverse population
    pop = generate_adaptive_random_population(size, lb, ub, n_candidates=500)

    # Convert to dict format for simulator (using alphabetically sorted order)
    test_cases = []
    for x in pop:
        test_case = {var_names[i]: x[i] for i in range(len(var_names))}
        # Convert int types
        for var in var_names:
            if conf.SS_VARIABLES[var]["domain"] == int:
                test_case[var] = int(test_case[var])
        test_cases.append(test_case)

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
        variables = build_pymoo_variables()
        super().__init__(vars=variables, n_obj=1, **kwargs)
        self.obj_ensemble = obj_ensemble
        self.var_names = sorted(conf.SS_VARIABLES.keys())

    def _evaluate(self, x, out, *args, **kwargs):
        params = np.array([x[var] for var in self.var_names])
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
        variables = build_pymoo_variables()
        n_uncovered = len(uncovered_objectives)
        super().__init__(vars=variables, n_obj=n_uncovered, **kwargs)
        self.obj_ensembles_dict = obj_ensembles_dict
        self.uncovered_objectives = uncovered_objectives
        self.var_names = sorted(conf.SS_VARIABLES.keys())

    def _evaluate(self, x, out, *args, **kwargs):
        params = np.array([x[var] for var in self.var_names])

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
    obj_names = [f"V{i}" for i in range(len(conf.CONSTRAINTS))]
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

        # Handle dict (single solution), 1D array, or 2D array/list (population)
        if isinstance(res.X, dict):
            pop_X = [res.X]  # Single dict solution
        elif isinstance(res.X, np.ndarray) and res.X.ndim == 1:
            pop_X = [res.X]  # Single 1D array solution
        else:
            pop_X = res.X if isinstance(res.X, list) else list(res.X)  # Population

        for x in pop_X:
            # Extract 6 parameters: car_speed, p_x, p_y, orientation, weather, road_shape
            if isinstance(x, dict):
                # Solution is a dictionary (from mixed variables)
                params = np.array([float(x["car_speed"]), float(x["p_x"]), float(x["p_y"]),
                                   int(x["orientation"]), int(x["weather"]), int(x["road_shape"])])
            else:
                # Solution is an array
                params = np.array([float(x[0]), float(x[1]), float(x[2]),
                                   int(x[3]), int(x[4]), int(x[5])])
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
    obj_names = [f"V{i}" for i in range(len(conf.CONSTRAINTS))]
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

    # Handle dict (single solution), 1D array, or 2D array/list (population)
    if isinstance(res.X, dict):
        pop_X = [res.X]  # Single dict solution
    elif isinstance(res.X, np.ndarray) and res.X.ndim == 1:
        pop_X = [res.X]  # Single 1D array solution
    else:
        pop_X = res.X if isinstance(res.X, list) else list(res.X)  # Population

    for x in pop_X:
        # Extract 6 parameters in ALPHABETICALLY SORTED order (matches surrogate training!)
        var_names_gs = sorted(conf.SS_VARIABLES.keys())
        if isinstance(x, dict):
            # Solution is a dictionary (from mixed variables)
            params = np.array([x[var] if conf.SS_VARIABLES[var]["domain"] == float else int(x[var])
                             for var in var_names_gs])
        else:
            # Solution is an array - assume it's in alphabetically sorted order from NSGA3
            params = np.array(x)

        # For each objective, track best and most uncertain candidates
        for obj_idx in uncovered_objectives:
            if obj_idx not in obj_ensembles_dict:
                continue

            pred, unc = obj_ensembles_dict[obj_idx].predict(params.reshape(1, -1))

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

    # Convert to dict format using ALPHABETICALLY SORTED order (matches params array order!)
    var_names_convert = sorted(conf.SS_VARIABLES.keys())
    candidates = []
    for params in unique_params:
        candidates.append({
            var_name: (float(params[i]) if conf.SS_VARIABLES[var_name]["domain"] == float
                      else int(params[i]))
            for i, var_name in enumerate(var_names_convert)
        })

    return candidates


# ============================================================================
# PHASE 2: LOCAL SEARCH - RBF-based single-objective optimization per cluster
# ============================================================================

class LSProblem(ElementwiseProblem):
    """Local Search: Uses SINGLE RBF surrogate per cluster (NOT ensemble!)"""

    def __init__(self, rbf_model, **kwargs):
        variables = build_pymoo_variables()
        super().__init__(vars=variables, n_obj=1, **kwargs)
        self.rbf = rbf_model
        self.var_names = sorted(conf.SS_VARIABLES.keys())

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate using SINGLE RBF surrogate (trained on cluster data)"""
        params = np.array([x[var] for var in self.var_names])
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

            try:
                local_surrogate = RBF_Model(n_neurons=10, train_data=train_data)
            except np.linalg.LinAlgError:
                # Skip cluster if RBF training fails (singular matrix = collinear/identical points)
                continue

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

                # Handle dict-like access - NSGA3 returns alphabetically sorted variables
                var_names_ls = sorted(conf.SS_VARIABLES.keys())
                try:
                    params = np.array([best_x[var] if conf.SS_VARIABLES[var]["domain"] == float else int(best_x[var])
                                      for var in var_names_ls])
                    selected.append(params)
                except (TypeError, KeyError):
                    # If dict access fails, assume array (shouldn't happen)
                    try:
                        params = np.array(best_x)
                        selected.append(params)
                    except (TypeError, IndexError):
                        pass  # Skip if we can't extract parameters

    # Convert numpy array params to dict format using ALPHABETICALLY SORTED order
    var_names_convert_ls = sorted(conf.SS_VARIABLES.keys())
    candidates = []
    for params in selected:
        candidates.append({
            var_name: (float(params[i]) if conf.SS_VARIABLES[var_name]["domain"] == float
                      else int(params[i]))
            for i, var_name in enumerate(var_names_convert_ls)
        })

    return candidates


# ============================================================================
# MAIN PFES+SAMOTA ALGORITHM
# ============================================================================

def pfes_samota(max_iterations=1000, max_time_seconds=3600, budget=900):
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

    # Create detailed log file
    import logging
    log_file = "pfes_samota_baseline/samota_detailed.log"
    os.makedirs("pfes_samota_baseline", exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("PFES + SAMOTA DETAILED LOG")
    logger.info("="*80)
    logger.info(f"Budget: {budget} evaluations")
    logger.info(f"Max iterations: {max_iterations}")
    logger.info(f"Max time: {max_time_seconds} seconds")
    logger.info(f"Constraints: {len(conf.CONSTRAINTS)}")
    logger.info(f"MINIMAL_CONSTRAINTS objectives: {len(conf.MINIMAL_CONSTRAINTS)}")

    start_time = time.time()
    archive = []  # Test cases that violate at least one requirement
    database = []  # All evaluated test cases
    database_X = []  # Parameters
    database_F = []  # Fitness values (RAW simulator outputs - larger range!)
    database_processed = []  # Processed scores (for tracking violations)

    # Track violations per requirement (like PFES baseline: R0, R1, R2)
    unsatisfied_reqs = [0] * len(conf.CONSTRAINTS)  # [R0, R1, R2]

    eval_count = 0

    print(f"📝 Logging to: {log_file}")

    # ========================================================================
    # PHASE 1: ADAPTIVE RANDOM TESTING (ART)
    # ========================================================================

    print("\nPHASE 1: ADAPTIVE RANDOM TESTING (ART) - Maximin Sampling")
    print("-" * 80)
    logger.info("PHASE 1: ADAPTIVE RANDOM TESTING (ART)")

    art_pop, art_X = art_initial_population(size=300)
    logger.info(f"ART population generated: {len(art_pop)} samples")

    phase1_start_evals = eval_count

    for i, test_case in enumerate(art_pop):
        if time.time() - start_time > max_time_seconds or eval_count >= budget:
            logger.warning(f"PHASE 1 STOPPED at iteration {i}: eval_count={eval_count}, budget={budget}, time_elapsed={(time.time()-start_time):.1f}s")
            break

        # Evaluate via simulator - get RAW outputs (not distances!)
        # CRITICAL: Must use ALPHABETICALLY SORTED order to match helpers.create_ss_variables()
        var_names = sorted(conf.SS_VARIABLES.keys())
        params = [test_case[var] for var in var_names]
        raw_estimates, processed_scores, reqs_satisfied = evaluate_test_case(params)

        eval_count += 1

        # Store in database (using RAW estimates for surrogates!)
        # CRITICAL: Must use ALPHABETICALLY SORTED order to match helpers.create_ss_variables()
        x_array = np.array([test_case[var] for var in var_names])
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
            logger.debug(f"  ART progress: {i+1}/300, evals: {eval_count}, violations found: {len(archive)}")

    phase1_evals = eval_count - phase1_start_evals
    print(f"✓ Phase 1 complete: {phase1_evals} evaluations, {len(archive)} violations found")
    logger.info(f"✓ Phase 1 complete: {phase1_evals} ART evaluations, {len(archive)} violations found")

    X_array = np.array(database_X)
    F_array = np.array(database_F)  # RAW estimates (for surrogates)
    F_processed = np.array(database_processed)  # Processed scores (for violation checking)

    # ========================================================================
    # PHASE 2: ITERATIVE GS + LS
    # ========================================================================

    print("\nPHASE 2: ITERATIVE GS + LS")
    print("-" * 80)
    logger.info("PHASE 2: ITERATIVE GS + LS")

    n_objectives = len(database_F[0]) if database_F else len(conf.CONSTRAINTS)
    logger.info(f"Number of objectives: {n_objectives}")
    uncovered_objectives = list(range(n_objectives))

    for iteration in range(max_iterations):
        elapsed_time = time.time() - start_time

        logger.info(f"\n--- ITERATION {iteration + 1} ---")
        logger.info(f"Current eval_count: {eval_count}/{budget}")
        logger.info(f"Elapsed time: {elapsed_time:.1f}s / {max_time_seconds}s")

        if elapsed_time > max_time_seconds:
            logger.warning(f"⏱️  TIME LIMIT REACHED at iteration {iteration}")
            print(f"✓ Time limit reached at iteration {iteration}")
            break

        if eval_count >= budget:
            logger.warning(f"💰 BUDGET EXHAUSTED at iteration {iteration}: eval_count={eval_count} >= budget={budget}")
            print(f"✓ Budget exhausted at iteration {iteration}")
            break

        print(f"\nITERATION {iteration + 1}")

        # Update uncovered objectives (those without violations yet)
        # Use PROCESSED scores to check violations (< 0 = violated)
        F_processed_array = np.array(database_processed)
        min_per_obj_processed = np.min(F_processed_array, axis=0)
        covered_objectives = [i for i in range(n_objectives) if min_per_obj_processed[i] < 0]
        uncovered_objectives = [i for i in range(n_objectives) if min_per_obj_processed[i] >= 0]

        logger.info(f"Objective status (min score per objective):")
        for i in range(n_objectives):
            status = "✓ COVERED" if i in covered_objectives else "✗ UNCOVERED"
            logger.info(f"  V{i}: {min_per_obj_processed[i]:.6f} {status}")

        if len(uncovered_objectives) == 0:
            logger.info("✓ ALL OBJECTIVES COVERED - stopping iteration loop")
            print("✓ All objectives covered!")
            break

        print(f"  Uncovered objectives: {uncovered_objectives}")
        logger.info(f"  Uncovered objectives: {uncovered_objectives} ({len(uncovered_objectives)}/{n_objectives})")

        # ====================================================================
        # Global Search Phase (PER-OBJECTIVE: one surrogate+GA per uncovered objective)
        # ====================================================================
        # PER-OBJECTIVE APPROACH: Mirrors LS design
        # - Surrogates: Per-objective (specialized training, same as LS)
        # - Optimization: Single-objective NSGA3 per uncovered objective (same as LS)
        # - Result: Focused search per objective (not multi-obj trade-offs)

        print(f"  Global Search (GS) - PER-OBJECTIVE (one surrogate+GA per uncovered obj)...")
        logger.info(f"  Running GS with uncovered objectives: {uncovered_objectives}")

        gs_start_evals = eval_count
        gs_candidates = global_search_nsga3(X_array, F_array, uncovered_objectives,
                                            pop_size=30, n_gen=20)
        logger.info(f"  GS generated {len(gs_candidates)} candidates")

        gs_violations_before = len(archive)

        for gs_idx, test_case in enumerate(gs_candidates):
            if eval_count >= budget:
                logger.warning(f"  GS stopped at candidate {gs_idx}/{len(gs_candidates)}: budget exhausted")
                break

            # Evaluate and get RAW simulator outputs (not distances!)
            # CRITICAL: Must use ALPHABETICALLY SORTED order to match helpers.create_ss_variables()
            var_names_local = sorted(conf.SS_VARIABLES.keys())
            params = [test_case[var] for var in var_names_local]
            raw_estimates, processed_scores, reqs_satisfied = evaluate_test_case(params)

            eval_count += 1

            x_array = np.array([test_case[var] for var in var_names_local])
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

        gs_evals_used = eval_count - gs_start_evals
        gs_violations_found = len(archive) - gs_violations_before
        print(f"    GS: {len(gs_candidates)} candidates generated, {gs_evals_used} evaluated, {gs_violations_found} new violations")
        logger.info(f"    GS: {len(gs_candidates)} candidates, {gs_evals_used} evals, {gs_violations_found} new violations, total violations: {len(archive)}")

        # ====================================================================
        # Local Search Phase
        # ====================================================================

        print(f"  Local Search (LS)...")
        logger.info(f"  Running LS with uncovered objectives: {uncovered_objectives}")

        ls_start_evals = eval_count
        ls_candidates = local_search_phase(X_array, F_array, uncovered_objectives, eta_percent=20, l_max=200, n_clusters=20)
        logger.info(f"  LS generated {len(ls_candidates)} candidates")

        ls_violations_before = len(archive)

        for ls_idx, test_case in enumerate(ls_candidates):
            if eval_count >= budget:
                logger.warning(f"  LS stopped at candidate {ls_idx}/{len(ls_candidates)}: budget exhausted")
                break

            # Evaluate and get RAW simulator outputs (not distances!)
            # CRITICAL: Must use ALPHABETICALLY SORTED order to match helpers.create_ss_variables()
            var_names_local = sorted(conf.SS_VARIABLES.keys())
            params = [test_case[var] for var in var_names_local]
            raw_estimates, processed_scores, reqs_satisfied = evaluate_test_case(params)

            eval_count += 1

            x_array = np.array([test_case[var] for var in var_names_local])
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

        ls_evals_used = eval_count - ls_start_evals
        ls_violations_found = len(archive) - ls_violations_before
        print(f"    LS: {len(ls_candidates)} candidates generated, {ls_evals_used} evaluated, {ls_violations_found} new violations")
        logger.info(f"    LS: {len(ls_candidates)} candidates, {ls_evals_used} evals, {ls_violations_found} new violations, total violations: {len(archive)}")
        print(f"    Total evals: {eval_count}")
        logger.info(f"  ITERATION {iteration + 1} COMPLETE: evals_used={(eval_count-phase1_evals)}, total_evals={eval_count}")

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
    logger.info("="*80)
    logger.info("PFES + SAMOTA FINAL RESULTS")
    logger.info("="*80)

    print(f"Total Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Total Evaluations: {eval_count} / {budget} ({eval_count/budget*100:.1f}%)")
    print(f"Archive Size: {len(archive)}")

    logger.info(f"Total Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Total Evaluations: {eval_count} / {budget} ({eval_count/budget*100:.1f}%)")
    logger.info(f"Archive Size: {len(archive)}")

    print(f"\nRequirements Breakdown (like PFES baseline):")
    logger.info(f"\nRequirements Breakdown:")
    for i, req_violations in enumerate(unsatisfied_reqs):
        print(f"  R{i}: {req_violations} violations")
        logger.info(f"  R{i}: {req_violations} violations")

    print(f"\nObjective Coverage:")
    logger.info(f"\nObjective Coverage:")
    for i in range(n_objectives):
        min_score = min_scores_processed[i]
        status = "✓ COVERED (negative)" if min_score < 0 else "✗ UNCOVERED (non-negative)"
        print(f"  V{i}: min_score={min_score:.6f} {status}")
        logger.info(f"  V{i}: min_score={min_score:.6f} {status}")

    print(f"\nTotal Violations: {violations}")
    print(f"Objectives Covered: {objectives_covered}/{n_objectives}")
    print(f"Min Scores: {min_scores}")
    print(f"Efficiency: {violations/eval_count:.4f} violations/eval")

    logger.info(f"\nTotal Violations: {violations}")
    logger.info(f"Objectives Covered: {objectives_covered}/{n_objectives}")
    logger.info(f"Min Scores: {list(min_scores)}")
    logger.info(f"Efficiency: {violations/eval_count:.4f} violations/eval")

    # ========================================================================
    # SAVE CSV FILES
    # ========================================================================

    import os

    # Create logdir if it doesn't exist
    if not os.path.exists("pfes_samota_baseline"):
        os.makedirs("pfes_samota_baseline", exist_ok=True)

    # Save best scores (like PFES: score_NSGA3_1.csv)
    best_scores_df = pd.DataFrame({
        f'V{i}': [min_scores[i]] for i in range(len(conf.CONSTRAINTS))
    })
    best_scores_df.to_csv('pfes_samota_baseline/score_NSGA3_1.csv', index=False)
    print("\n✓ Saved: pfes_samota_baseline/score_NSGA3_1.csv")

    # Save requirements breakdown (like PFES: reqs_NSGA3_1.csv)
    # Dynamic: create one R column per constraint
    reqs_dict = {f'R{i}': [unsatisfied_reqs[i]] for i in range(len(conf.CONSTRAINTS))}
    reqs_dict['conjunction'] = [violations]
    reqs_df = pd.DataFrame(reqs_dict)
    reqs_df.to_csv('pfes_samota_baseline/reqs_NSGA3_1.csv', index=False)
    print("✓ Saved: pfes_samota_baseline/reqs_NSGA3_1.csv")

    # Save all evaluations (like PFES) - use alphabetically sorted variable order
    var_names_save = sorted(conf.SS_VARIABLES.keys())
    X_df = pd.DataFrame(
        database_X,
        columns=var_names_save
    )
    X_df.to_csv('pfes_samota_baseline/X_all_evaluations_NSGA3_0.csv', index=False)
    print("✓ Saved: pfes_samota_baseline/X_all_evaluations_NSGA3_0.csv")

    F_df = pd.DataFrame(
        database_processed,
        columns=[f'V{i}' for i in range(len(conf.CONSTRAINTS))]
    )
    F_df.to_csv('pfes_samota_baseline/F_all_evaluations_NSGA3_0.csv', index=False)
    print("✓ Saved: pfes_samota_baseline/F_all_evaluations_NSGA3_0.csv")

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
    results = pfes_samota(max_iterations=1000, budget=900)

    print("\n✓ Test completed!")
    print(json.dumps(results, indent=2))
