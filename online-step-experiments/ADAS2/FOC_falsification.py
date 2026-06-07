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
conf.MAX_STEPS = 20000
conf.BATCH_SIZE = 100

conf.MDP_FOLDER = "INPUT/AutonomousDriving_v1"
conf.PLOT = False
conf.MAX_SAMPLES = 100

HIGH = 10.0

columns = ["car_speed",
           "p_x",
           "p_y",
           "orientation",
           "weather",
           "road_shape"] + ['req_{}'.format(i) for i in range(len(conf.CONSTRAINTS))]



class AutonomousDrivingProblem(ElementwiseProblem):
    def __init__(self, n_objectives, n_reqs, **kwargs):
        variables = {
            "car_speed": Real(bounds=(5.0, 50.0)),
            "p_x": Real(bounds=(0.0, 10.0)),
            "p_y": Real(bounds=(0.0, 10.0)),
            "orientation": Integer(bounds=(-30, 30)),
            "weather": Integer(bounds=(0, 2)),
            "road_shape": Integer(bounds=(0, 2)),
        }
        self.unsatisfied_reqs = [0] * n_reqs
        self.min_scores = [1] * n_objectives
        self.conjunction = 0
        self.sensitivity = False
        self.var = None
        self.reset_random_assignment()
        self.reqs_min_score = [1] * len(conf.CONSTRAINTS)
        super().__init__(vars=variables, n_obj=n_objectives, **kwargs)

    def set_sensitivity_var(self, var):
        self.var = var
        self.sensitivity = True

    def reset_random_assignment(self):
        self.rand_assignment = {
            "car_speed": None,
            "p_x": None,
            "p_y": None,
            "orientation": None,
            "weather": None,
            "road_shape": None,
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
        if self.sensitivity:
            x = self.get_assignment(x, self.var)
            _, scores, reqs_satisfied, reqs_min_score, conjunction = helpers.run_mdp_sensitivity([x["car_speed"], 
                                                        x["p_x"], 
                                                        x["p_y"], 
                                                        x["orientation"], 
                                                        x["weather"], 
                                                        x["road_shape"]])
        else:
            _, scores, reqs_satisfied, conjunction = helpers.run_mdp([x["car_speed"], 
                                                        x["p_x"], 
                                                        x["p_y"], 
                                                        x["orientation"], 
                                                        x["weather"], 
                                                        x["road_shape"]])

        for i in range(0, len(self.min_scores)):
            if scores[i] < self.min_scores[i]:
                self.min_scores[i] = scores[i]
        for i in range(0, len(self.unsatisfied_reqs)):
            if not reqs_satisfied[i]:
                self.unsatisfied_reqs[i] +=1
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
@click.option('--size', default=30, help='Population size.', type=int)
@click.option('--totbudget', default=900, help='Total budget.', type=int)
@click.option('--nruns', default=30, help='Runs.', type=int)
@click.option('--verbose', default=False, help='Verbose.', type=bool)
@click.option('--logdir', default="out", help='Log directory.', type=str)
def main(size, totbudget, nruns, verbose, logdir):
    SEED = 1
    RUNS = nruns
    SIZE = size
    BUDGET = totbudget
    VERBOSE = verbose
    LOGDIR = logdir
    
    NREQS = 6
    OBJECTIVES = 5

    uns_reqs_df = pd.DataFrame(columns=[f'R{j}' for j in range(0, NREQS)] + ["conjunction"])
    score_df = pd.DataFrame(columns=[f'V{j}' for j in range(0, OBJECTIVES)])
        
    for run in range(0, RUNS):

        problem = AutonomousDrivingProblem(n_objectives=OBJECTIVES, n_reqs=NREQS)
        ref_dirs = get_reference_directions("das-dennis", OBJECTIVES, n_partitions=2)

        sensitivity_budget = BUDGET // 3
        #focused_test_budget = sensitivity_budget * 2

        sensitivity_run_budget = sensitivity_budget // len(conf.SS_VARIABLES)
        sensitivity_run_iterations = max(1, sensitivity_run_budget // SIZE)
        #focused_test_run_budget = focused_test_budget // NREQS
        focused_test_run_budget = BUDGET // NREQS
        focused_test_run_budget_iterations = max(1, focused_test_run_budget // SIZE)

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
        scores = np.array(reqs_min_score)
        unsatisfied_reqs_total = [0] * NREQS
        min_scores_total = [HIGH] * OBJECTIVES
        conjunction_total = 0
        for i in range(0, len(scores)):
            j = np.where(scores == scores.min())[0][0]
            # falsification of req_j
            algorithm = NSGA3(ref_dirs=ref_dirs,
                            pop_size=SIZE,
                            sampling=MixedVariableSampling(),
                            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                            eliminate_duplicates=MixedVariableDuplicateElimination())
            problem = AutonomousDrivingProblem(n_objectives=OBJECTIVES, n_reqs=NREQS)
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

    if LOGDIR is not None:
        uns_reqs_df.to_csv(f'{LOGDIR}/reqs_FOC_{RUNS}.csv', index=False)
        score_df.to_csv(f'{LOGDIR}/score_FOC_{RUNS}.csv', index=False)

if __name__ == "__main__":
    main()
