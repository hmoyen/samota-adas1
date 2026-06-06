# PFES + SAMOTA Integration - Formal Algorithms

**Author**: Helena Moyen
**Date**: May 20, 2026
**Objectives**: 5 (from MINIMAL_CONSTRAINTS: S0.a[2 bounds] + S2.b[3 elements] = 5 fitness values)

---

## Algorithm 1: Main PFES+SAMOTA Loop

```
Algorithm 1: PFES + SAMOTA Hybrid Framework
Input:  Search space X, budget B (900 evaluations), CONSTRAINTS (3 requirements → 5 objectives)
Output: Archive A (test cases violating constraints), Statistics

1: // PHASE 1: Adaptive Random Testing
2: P ← ART_InitialPopulation(size=300)          // Maximin sampling
3: A ← ∅                                        // Archive
4: D ← ∅                                        // Evaluation database
5: U_global ← {0, 1, 2, 3, 4}                  // All 5 optimization objectives (S0.a[2] + S2.b[3])
6: evals_phase1 ← 0

7: for each test case x ∈ P do
8:     f ← SimulatorEval(x)                    // Expensive evaluation
9:     StoreInDatabase(D, x, f)                 // Add to database
10:    evals_phase1 ← evals_phase1 + 1
11:    if ViolatesAnyConstraint(x, f) then
12:        A ← A ∪ {x}
13:    end if
14: end for
15: print("Phase 1: ", evals_phase1, "evals, ", |A|, "violations")

16: // PHASE 2: Iterative Global Search + Local Search
17: iteration ← 0
18: budget_remaining ← B - evals_phase1

19: while budget_remaining > 0 do
20:    // Update uncovered objectives (5 total)
21:    U ← {i | min(D[:, i]) < 0}              // Violated objectives (from 5)
22:    U_uncovered ← {0, 1, 2, 3, 4} \ U       // Objectives not yet violated (initially all 5)
23:
24:    if U_uncovered = ∅ then
25:        break                                // All objectives covered
26:    end if
27:
28:    print("Iteration ", iteration, ": U_uncovered = ", U_uncovered)
29:
30:    // GLOBAL SEARCH: Per-objective surrogates + Multi-objective NSGA3
31:    GS_candidates ← GlobalSearch(D, U_uncovered, pop_size=30, n_gen=20)
32:
33:    for each x ∈ GS_candidates do
34:        if budget_remaining ≤ 0 then break end if
35:        f ← SimulatorEval(x)
36:        StoreInDatabase(D, x, f)
37:        evals_phase1 ← evals_phase1 + 1
38:        budget_remaining ← budget_remaining - 1
39:
39:        if ViolatesAnyConstraint(x, f) then
40:            A ← A ∪ {x}
41:        end if
42:    end for
43:
44:    // LOCAL SEARCH: RBF per cluster + NSGA3
45:    LS_candidates ← LocalSearch(D, U_uncovered, eta=0.20, l_max=200, n_clusters=20)
46:
47:    for each x ∈ LS_candidates do
48:        if budget_remaining ≤ 0 then break end if
49:        f ← SimulatorEval(x)
50:        StoreInDatabase(D, x, f)
51:        evals_phase1 ← evals_phase1 + 1
52:        budget_remaining ← budget_remaining - 1
53:
54:        if ViolatesAnyConstraint(x, f) then
55:            A ← A ∪ {x}
56:        end if
57:    end for
58:
59:    iteration ← iteration + 1
60: end while

61: // REPORT RESULTS
62: violations_per_constraint ← [0, 0, 0]
63: for each x ∈ A do
64:     for i ← 0 to 2 do
65:         if ¬CheckConstraint(x, constraint[i]) then
66:             violations_per_constraint[i] ← violations_per_constraint[i] + 1
67:         end if
68:     end for
69: end for

70: return {
71:     archive_size: |A|,
72:     violations_R0: violations_per_constraint[0],
73:     violations_R1: violations_per_constraint[1],
74:     violations_R2: violations_per_constraint[2],
75:     total_violations: sum(violations_per_constraint),
76:     objectives_covered: |{0,1,2} \ U_uncovered|,
77:     evaluations: evals_phase1
78: }
```

---

## Algorithm 2: Global Search - Per-Objective Surrogates + Multi-Objective NSGA3

```
Algorithm 2: Global Search (GS)
Input:  Database D = {(x, f)} pairs, U_uncovered (3 or fewer objectives),
         pop_size=30, n_gen=20
Output: Set of candidate test cases (2-6 per objective)

1: // For each uncovered objective, train surrogate + use for NSGA3
2: surrogates ← {}
3:
4: for each objective u ∈ U_uncovered do
5:     S_u ← TrainPerObjectiveSurrogate(D, u)  // Train ensemble on objective u
6:                                              // Ensemble = [GP, Polynomial(deg=2), RBF(10 neurons)]
7:     surrogates[u] ← S_u
8: end for
9:
10: // Create multi-objective NSGA3 problem using ALL surrogates
11: n_obj_search ← |U_uncovered|               // 1, 2, or 3
12: ref_dirs ← GetReferenceDirections(n_objectives=n_obj_search, partitions=2)
13:
14: class GSProblem extends Problem:
15:     def evaluate(population, results):
16:         for each x ∈ population do
17:             // Evaluate all uncovered objectives on surrogates
17:             y_pred ← []
18:             for u ∈ U_uncovered do
19:                 ŷ_u ← surrogates[u].predict(x)  // Ensemble prediction
20:                 σ_u ← surrogates[u].uncertainty(x)  // Ensemble disagreement
21:                 y_pred.append(ŷ_u)
22:             end for
23:             results[x] = y_pred                 // NSGA3: Pareto-based selection
24:         end for
25:     end function
26: end class
27:
28: // Run NSGA3 for 20 generations
29: algorithm ← NSGA3(
30:     problem = GSProblem,
31:     pop_size = 30,
32:     ref_dirs = ref_dirs,
33:     eliminate_duplicates = true,
34:     mating = SBX_PolynomialMutation()
35: )
36:
37: result ← Minimize(algorithm, n_gen=20)       // 20 generations × 30 pop = 600 surr evals
38: final_population ← result.pop
39:
40: // Extract best + most uncertain per objective
41: T̂^best ← {}
42: T̂^uncertain ← {}
43:
44: for each objective u ∈ U_uncovered do
45:     x_best ← argmin(surrogates[u].predict(final_population))  // Lowest predicted fitness
46:     x_uncertain ← argmax(surrogates[u].uncertainty(final_population))  // Highest uncertainty
46:     T̂^best.add(x_best)
47:     T̂^uncertain.add(x_uncertain)
48: end for
49:
50: // Remove duplicates and return
51: candidates ← RemoveDuplicates(T̂^best ∪ T̂^uncertain)
52: return candidates                            // Expected: 2-6 candidates (2 per obj if unique)
```

---

## Algorithm 3: Local Search - RBF per Cluster + NSGA3

```
Algorithm 3: Local Search (LS)
Input:  Database D = {(x, f)} pairs, U_uncovered objectives,
         eta_percent=20 (top 20% of evals), l_max=200 (GA generations),
         n_clusters=20 (target number of clusters)
Output: Set of candidate test cases (5-30)

1: // Filter to top 20% per each uncovered objective
2: X_top ← {}
3: F_top ← {}
4:
5: for each objective u ∈ U_uncovered do
6:     // Sort by fitness for objective u
7:     sorted_indices ← SortByObjective(D, u)
8:     top_20_percent ← sorted_indices[0 : ceil(0.20 * len(D))]
9:     X_top ← X_top ∪ {D[i].x for i in top_20_percent}
10:    F_top ← F_top ∪ {D[i].f for i in top_20_percent}
11: end for
12:
13: X_top ← RemoveDuplicates(X_top)             // ~60 individuals for 300 in Phase 1
14:
15: // Cluster using HDBSCAN (min_cluster_size=5)
16: clusterer ← HDBSCAN(min_cluster_size=5)
17: cluster_labels ← clusterer.fit_predict(X_top)
17: valid_clusters ← {i | cluster_i has ≥ 5 samples}  // HDBSCAN enforces this
18:
19: candidates ← {}
20:
21: for each cluster_id ∈ valid_clusters do
22:     // Indices in this cluster
23:     cluster_mask ← (cluster_labels = cluster_id)
24:     X_cluster ← X_top[cluster_mask]
25:     F_cluster ← F_top[cluster_mask]
26:
27:     // Skip if cluster too small
28:     if len(X_cluster) < 5 then continue end if
29:
30:     // Train RBF on this cluster (single RBF, not ensemble)
31:     rbf_model ← TrainRBF(X_cluster, F_cluster, neurons=10)
32:
33:     // Create single-objective NSGA3 per objective
34:     for each objective u ∈ U_uncovered do
35:         class LSProblem extends Problem:
35:             def evaluate(population, results):
36:                 for each x ∈ population do
37:                     ŷ_u ← rbf_model.predict(x)[u]  // RBF for objective u
38:                     results[x] = [ŷ_u]             // Single objective
39:                 end for
40:             end function
41:         end class
42:
43:         // Run NSGA3 for l_max=200 generations
44:         algorithm ← NSGA3(
45:             problem = LSProblem,
45:             pop_size = 30,
46:             eliminate_duplicates = true,
47:             mating = SBX_PolynomialMutation()
48:         )
49:
50:         result ← Minimize(algorithm, n_gen=l_max)  // 200 generations
51:         best_x ← result.pop[0]  // Best from final population
52:         candidates ← candidates ∪ {best_x}
53:     end for
54: end for
55:
56: candidates ← RemoveDuplicates(candidates)
57: return candidates  // Expected: n_clusters × |U_uncovered| candidates (before dedup)
```

---

## Algorithm 4: Per-Objective Surrogate Training

```
Algorithm 4: TrainPerObjectiveSurrogate(D, objective_u)
Input:  Database D = {(x, f)} with constraint-mapped fitness values,
         objective_u (which constraint to focus on)
Output: Surrogate ensemble (GP + Polynomial + RBF)

1: // Extract data for this objective
2: X_train ← D.parameters                      // All parameter vectors
3: F_train ← D.fitness[:, u]                   // Fitness for objective u only
4:
5: // Train three models independently
6: surrogate_gp ← TrainGaussianProcess(X_train, F_train)
7: surrogate_poly ← TrainPolynomialRegression(X_train, F_train, degree=2)
8: surrogate_rbf ← TrainRBFNetwork(X_train, F_train, neurons=10)
9:
10: // Weighted ensemble (Goel weighting)
11: weights ← ComputeGoelWeights(surrogate_gp, surrogate_poly, surrogate_rbf, X_train, F_train)
12:
13: class EnsembleSurrogate:
14:     def predict(x):
15:         y_gp ← surrogate_gp.predict(x)
16:         y_poly ← surrogate_poly.predict(x)
17:         y_rbf ← surrogate_rbf.predict(x)
18:         return weights[0]*y_gp + weights[1]*y_poly + weights[2]*y_rbf
19:     end function
20:
21:     def uncertainty(x):
22:         y_gp ← surrogate_gp.predict(x)
23:         y_poly ← surrogate_poly.predict(x)
24:         y_rbf ← surrogate_rbf.predict(x)
25:         return max([y_gp, y_poly, y_rbf]) - min([y_gp, y_poly, y_rbf])
26:     end function
27: end class
28:
29: return EnsembleSurrogate()
```

---

## Key Differences: PFES vs PFES+SAMOTA

| Aspect | PFES (Baseline) | PFES+SAMOTA (Hybrid) |
|--------|-----------------|----------------------|
| **Phase 1** | ART (300 evals) | ART (300 evals) - Same |
| **Phase 2** | NSGA3 on real simulator (600 evals, expensive, 5 objectives) | Iterative GS+LS with surrogates (guides selection, 5 per-objective models) |
| **Objective Filtering** | All 5 objectives, every iteration | Only uncovered objectives (dynamic filtering from 5) |
| **Surrogates** | None (0 evals) | 5 per-objective ensembles (600 GS + 6000+ LS surr evals) |
| **Selection** | NSGA3 decides directly on simulator (15 ref dirs for 5 objectives) | Per-objective surrogates narrow down, NSGA3 selects best+uncertain per obj |
| **Budget** | 900 real evals | 300 Phase1 + ~600 Phase2 real evals + surrogate evals |
| **Expected Gain** | Baseline | 2-3× efficiency (same violations, guided search via 5 surrogates) |

---

## Constraint-Mapped Fitness Explanation

```
Raw Simulator Output: distance_to_obstacle = 0.7 meters

Constraint Threshold: 0.3 meters (desired minimum distance)

Constraint-Mapped Fitness:
    f = raw_value - threshold
    f = 0.7 - 0.3 = +0.4

Violation Detection:
    if f < 0:  Constraint violated
    else:      Constraint satisfied

Why This Works for Surrogates:
    - Symmetric around boundary (f = 0)
    - Surrogates learn patterns near violation boundary
    - Easy to identify promising regions (f → 0)
    - Better than raw values which have mixed scales
```

---

## Implementation Details: ADAS1

**Search Space** (6D):
- `car_speed`: [5.0, 50.0] (continuous)
- `p_x`: [0.0, 10.0] (continuous)
- `p_y`: [0.0, 10.0] (continuous)
- `orientation`: [-30, 30] (integer)
- `weather`: [0, 2] (integer)
- `road_shape`: [0, 2] (integer)

**Constraints & Objectives**:

Requirement Satisfaction (3 states):
- **R0**: S0.a within bounds (both 2 elements satisfied)
- **R1**: S2.b within bounds (all 3 elements satisfied)
- **R2**: Both R0 AND R1 satisfied

Optimization Objectives (5 fitness values):
- **Obj 0**: S0.a element 1 fitness
- **Obj 1**: S0.a element 2 fitness
- **Obj 2**: S2.b element 1 fitness
- **Obj 3**: S2.b element 2 fitness
- **Obj 4**: S2.b element 3 fitness

**Budget**: 900 evaluations total
- Phase 1 (ART): 300 evals
- Phase 2 (GS+LS): ~600 evals (iterative until budget exhausted)
