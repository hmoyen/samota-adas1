import click
import multiprocessing as mp
import numpy as np
import utils.helpers as helpers
import config as conf
import pandas as pd
import time
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
# from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination

from pymoo.optimize import minimize

THREADS_COUNT = 1
conf.MAX_STEPS = 10000
conf.BATCH_SIZE = 100

conf.MDP_FOLDER = "./INPUT/RescueRobot_v3"
conf.PLOT = False
conf.MAX_SAMPLES = 100

HIGH = 10.0

# Variable names in alphabetical order (matches create_ss_variables sorting)
RR_VAR_NAMES = sorted(["power", "cruise_speed", "bandwidth", "quality",
                        "illuminance", "smoke_intensity", "obstacle_size",
                        "obstacle_distance", "firm_obstacle"])

columns = ["power",
           "cruise_speed",
           "bandwidth",
           "quality",
           "illuminance",
           "smoke_intensity",
           "obstacle_size",
           "obstacle_distance",
           "firm_obstacle"] + ['req_{}'.format(i) for i in range(len(conf.CONSTRAINTS))]


class RescueRobotProblem(ElementwiseProblem):
    def __init__(self, n_objectives, n_reqs, shared_eval_count=None, shared_first_viol=None,
                 shared_all_X=None, shared_all_reqs=None, **kwargs):
        variables = {
            "power": Integer(bounds=(0, 100)),
            "cruise_speed": Real(bounds=(0, 5)),
            "bandwidth": Real(bounds=(10, 50)),
            "quality": Integer(bounds=(0, 2)),
            "illuminance": Real(bounds=(40, 120000)),
            "smoke_intensity": Integer(bounds=(0, 2)),
            "obstacle_size": Real(bounds=(0, 120)),
            "obstacle_distance": Real(bounds=(0, 10)),
            "firm_obstacle": Integer(bounds=(0, 1)),
        }
        self.unsatisfied_reqs = [0] * n_reqs
        self.min_scores = [1] * n_objectives
        self.conjunction = 0
        self.sensitivity = False
        self.var = None
        self.shared_eval_count = shared_eval_count  # [int] — mutable counter shared across instances
        self.shared_first_viol = shared_first_viol  # [None|int] per req — shared across instances
        self.shared_all_X = shared_all_X  # list — mutable, shared across instances
        self.shared_all_reqs = shared_all_reqs  # list — mutable, shared across instances
        self.reset_random_assignment()
        self.reqs_min_score = [1] * len(conf.CONSTRAINTS)
        super().__init__(vars=variables, n_obj=n_objectives, **kwargs)

    def set_sensitivity_var(self, var):
        self.var = var
        self.sensitivity = True

    def reset_random_assignment(self):
        self.rand_assignment = {
            "power": None,
            "cruise_speed": None,
            "bandwidth": None,
            "quality": None,
            "illuminance": None,
            "smoke_intensity": None,
            "obstacle_size": None,
            "obstacle_distance": None,
            "firm_obstacle": None,
        }
    
    def get_assignment(self, x, var):
        for k in x:
            if k != var:
                if self.rand_assignment[k] is None:
                    self.rand_assignment[k] = self.vars[k].sample()
                x[k] = self.rand_assignment[k]
        return x
    
    def update_reqs_min_score(self, current_min_score):
        for i in range(0, len(self.reqs_min_score)):
            if current_min_score[i] < self.reqs_min_score[i]:
                self.reqs_min_score[i] = current_min_score[i]

    def _evaluate(self, x, out, *args, **kwargs):
        # Pass variables in alphabetical order to match create_ss_variables()
        params = [x[v] for v in RR_VAR_NAMES]
        if self.sensitivity:
            x = self.get_assignment(x, self.var)
            params = [x[v] for v in RR_VAR_NAMES]
            _, scores, reqs_satisfied, reqs_min_score, conjunction = helpers.run_mdp_sensitivity(params)
        else:
            _, scores, reqs_satisfied, conjunction = helpers.run_mdp(params)

        if self.shared_all_X is not None:
            self.shared_all_X.append(params)
            self.shared_all_reqs.append(reqs_satisfied)

        if self.shared_eval_count is not None:
            self.shared_eval_count[0] += 1
        for i in range(0, len(self.min_scores)):
            if scores[i] < self.min_scores[i]:
                self.min_scores[i] = scores[i]
        for i in range(0, len(self.unsatisfied_reqs)):
            if not reqs_satisfied[i]:
                self.unsatisfied_reqs[i] += 1
                if self.shared_first_viol is not None and self.shared_first_viol[i] is None:
                    self.shared_first_viol[i] = self.shared_eval_count[0]
        self.conjunction += conjunction
        if self.sensitivity:
            self.update_reqs_min_score(reqs_min_score)
        out["F"] = scores

def log_results(n_run, unsatisfied_reqs, unsatisfied_conjunction, best_scores, duration):
    print("\nTotal Duration run {}: {} seconds".format(n_run, duration))
    print("\n\nReqs unsatisfied:")
    print(unsatisfied_reqs)
    print("\n\nReqs conjunction:")
    print(unsatisfied_conjunction)
    print("\n\nBest scores:")
    print(best_scores)

@click.command()
@click.option('--size', default=20, help='Population size.', type=int)
@click.option('--totbudget', default=900, help='Total budget.', type=int)
@click.option('--nruns', default=30, help='Runs.', type=int)
@click.option('--verbose', default=False, help='Verbose.', type=bool)
@click.option('--logdir', default="out", help='Log directory.', type=str)
@click.option('--seed', default=1, help='Base random seed (each run uses seed+run_index).', type=int)
@click.option('--equalize_budget', default=False,
              help="Truncate the focused-testing budget so sensitivity + focused evals "
                   "sum to ~totbudget, matching the fixed budget every other algorithm "
                   "uses. Default (False) preserves existing behavior, where the "
                   "sensitivity phase's evals are spent on top of a full totbudget for "
                   "focused testing.", type=bool)
def main(size, totbudget, nruns, verbose, logdir, seed, equalize_budget):
    BASE_SEED = seed
    RUNS = nruns
    SIZE = size
    BUDGET = totbudget
    VERBOSE = verbose
    LOGDIR = logdir
    
    NREQS = 6
    OBJECTIVES = 5

    uns_reqs_df = pd.DataFrame(columns=[f'R{j}' for j in range(0, NREQS)] + ["conjunction"])
    score_df = pd.DataFrame(columns=[f'V{j}' for j in range(0, OBJECTIVES)])
    timing_df = pd.DataFrame(columns=[f'R{j}_first_eval' for j in range(0, NREQS)] + ["full_coverage_eval"])

    for run in range(0, RUNS):
        SEED = BASE_SEED + run  # unique seed per run for statistical independence

        shared_eval_count = [0]
        shared_first_viol = [None] * NREQS
        shared_all_X = []
        shared_all_reqs = []
        problem = RescueRobotProblem(n_objectives=OBJECTIVES, n_reqs=NREQS,
                                     shared_eval_count=shared_eval_count,
                                     shared_first_viol=shared_first_viol,
                                     shared_all_X=shared_all_X,
                                     shared_all_reqs=shared_all_reqs)
        ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=2)

        sensitivity_budget = BUDGET // 3
        #focused_test_budget = sensitivity_budget * 2

        sensitivity_run_budget = sensitivity_budget // len(conf.SS_VARIABLES)
        sensitivity_run_iterations = max(1, sensitivity_run_budget // SIZE)
        #focused_test_run_budget = focused_test_budget // NREQS

        start_time = time.time()
        # start sensitivity analysis
        reqs_min_score = [1] * len(conf.CONSTRAINTS)
        for var in conf.SS_VARIABLES:
            problem.set_sensitivity_var(var)
            problem.reset_random_assignment()
            algorithm = NSGA3(ref_dirs=ref_dirs,
                            pop_size=SIZE,
                            sampling=MixedVariableSampling(),
                            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                            eliminate_duplicates=MixedVariableDuplicateElimination())
            
            res = minimize(problem,
                            algorithm,
                            ('n_gen', sensitivity_run_iterations),
                            seed=SEED,
                            save_history=False,
                            verbose=VERBOSE)
            for i in range(0, len(reqs_min_score)):
                if problem.reqs_min_score[i] < reqs_min_score[i]:
                    reqs_min_score[i] = problem.reqs_min_score[i]

        # start focused testing
        if equalize_budget:
            remaining_budget = max(BUDGET - shared_eval_count[0], NREQS)
            focused_test_run_budget = remaining_budget // NREQS
        else:
            focused_test_run_budget = BUDGET // NREQS
        focused_test_run_budget_iterations = max(1, focused_test_run_budget // SIZE)

        scores = np.array(reqs_min_score)
        unsatisfied_reqs_total = [0] * len(scores)
        min_scores_total = [HIGH] * len(scores)
        conjunction_total = 0
        for i in range(0, len(scores)):
            j = np.where(scores == scores.min())[0][0]
            # falsification of req_j
            algorithm = NSGA3(ref_dirs=ref_dirs,
                            pop_size=SIZE,
                            sampling=MixedVariableSampling(),
                            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                            eliminate_duplicates=MixedVariableDuplicateElimination())
            problem = RescueRobotProblem(n_objectives=OBJECTIVES, n_reqs=NREQS,
                                         shared_eval_count=shared_eval_count,
                                         shared_first_viol=shared_first_viol,
                                         shared_all_X=shared_all_X,
                                         shared_all_reqs=shared_all_reqs)
            res = minimize(problem,
                            algorithm,
                            ('n_gen', focused_test_run_budget_iterations),
                            seed=SEED,
                            save_history=False,
                            verbose=VERBOSE)
            # results for req_j
            unsatisfied_reqs_total = [sum(x) for x in zip(unsatisfied_reqs_total, problem.unsatisfied_reqs)]
            min_scores_total = [min(x) for x in zip(min_scores_total, problem.min_scores)]
            conjunction_total += problem.conjunction
            scores[j] = HIGH

        log_results(run, unsatisfied_reqs_total, conjunction_total, min_scores_total, time.time() - start_time)
        uns_reqs_df.loc[run] = unsatisfied_reqs_total + [conjunction_total]
        score_df.loc[run] = min_scores_total
        full_coverage_eval = max(shared_first_viol) if all(v is not None for v in shared_first_viol) else None
        timing_df.loc[run] = shared_first_viol + [full_coverage_eval]

        if LOGDIR is not None:
            var_names_save = sorted(conf.SS_VARIABLES.keys())
            X_df = pd.DataFrame(shared_all_X, columns=var_names_save)
            X_df.to_csv(f'{LOGDIR}/X_all_evaluations_FOC_{run}.csv', index=False)
            reqs_df = pd.DataFrame(shared_all_reqs, columns=[f'R{j}' for j in range(NREQS)])
            reqs_df.to_csv(f'{LOGDIR}/Reqs_all_evaluations_FOC_{run}.csv', index=False)

    if LOGDIR is not None:
        uns_reqs_df.to_csv(f'{LOGDIR}/reqs_FOC_{RUNS}.csv', index=False)
        score_df.to_csv(f'{LOGDIR}/score_FOC_{RUNS}.csv', index=False)
        timing_df.to_csv(f'{LOGDIR}/timing_FOC_{RUNS}.csv', index=False)

if __name__ == "__main__":
    main()
