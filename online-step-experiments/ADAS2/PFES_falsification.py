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
        # Track all evaluations for surrogate model training
        self.all_X = []
        self.all_F = []
        self.all_reqs = []
        super().__init__(vars=variables, n_obj=n_objectives, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        _, scores, reqs_satisfied, conjunction = helpers.run_mdp([x["car_speed"],
                                                        x["p_x"],
                                                        x["p_y"],
                                                        x["orientation"],
                                                        x["weather"],
                                                        x["road_shape"]])
        # Store all evaluations
        self.all_X.append([x["car_speed"], x["p_x"], x["p_y"], x["orientation"], x["weather"], x["road_shape"]])
        self.all_F.append(scores)
        self.all_reqs.append(reqs_satisfied)

        for i in range(0, len(self.min_scores)):
            if scores[i] < self.min_scores[i]:
                self.min_scores[i] = scores[i]
        for i in range(0, len(self.unsatisfied_reqs)):
            if not reqs_satisfied[i]:
                self.unsatisfied_reqs[i] +=1
        self.conjunction += conjunction
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
@click.option('--niterations', default=30, help='Iterations.', type=int)
@click.option('--nruns', default=1, help='Runs.', type=int)
@click.option('--optalg', default="NSGA3", help='Algorithm.', type=str)
@click.option('--verbose', default=False, help='Verbose.', type=bool)
@click.option('--logdir', default="out", help='Log directory.', type=str)
@click.option('--seed', default=1, help='Random seed.', type=int)
def main(size, niterations, nruns, optalg, verbose, logdir, seed):
    SEED = seed
    RUNS = nruns
    SIZE = size
    ITERATIONS = niterations
    OPTALG = optalg
    VERBOSE = verbose
    LOGDIR = logdir
    
    NREQS = len(conf.CONSTRAINTS)  # 3 requirements (R0, R1, R2)
    OBJECTIVES = 5  # From MINIMAL_CONSTRAINTS: S0.a (2 bounds) + S2.b (3 elements) = 5 objectives

    uns_reqs_df = pd.DataFrame(columns=[f'R{j}' for j in range(0, NREQS)] + ["conjunction"])
    score_df = pd.DataFrame(columns=[f'V{j}' for j in range(0, OBJECTIVES)])
        
    for run in range(0, RUNS):

        if OPTALG == "RANDOM":
            min_scores = [1] * OBJECTIVES
            unsatisfied_reqs = [0] * NREQS
            unsatisfied_conjunction = 0
            
            start_time = time.time()
            
            combinations = helpers.build_random_combinations(SIZE * ITERATIONS)
            pool = mp.Pool(processes=THREADS_COUNT)
            results = pool.map(helpers.run_mdp, list(combinations))
            pool.close()
            pool.join()

            for _, scores, reqs_satisfied, conjunction in results:
                for i in range(0, len(min_scores)):
                    if scores[i] < min_scores[i]:
                        min_scores[i] = scores[i]
                for i in range(0, len(reqs_satisfied)):
                    if not reqs_satisfied[i]:
                        unsatisfied_reqs[i] +=1
                unsatisfied_conjunction += conjunction
            
            log_results(run, unsatisfied_reqs, unsatisfied_conjunction, min_scores, time.time() - start_time)
            uns_reqs_df.loc[run] = unsatisfied_reqs + [unsatisfied_conjunction]
            score_df.loc[run] = min_scores
        else:
            problem = AutonomousDrivingProblem(n_objectives=OBJECTIVES, n_reqs=NREQS)
            ref_dirs = get_reference_directions("das-dennis", OBJECTIVES, n_partitions=2)

            if OPTALG == "NSGA3":
                algorithm = NSGA3(ref_dirs=ref_dirs,
                                pop_size=SIZE,
                                sampling=MixedVariableSampling(),
                                mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                                eliminate_duplicates=MixedVariableDuplicateElimination())
            elif OPTALG == "UNSGA3":
                algorithm = UNSGA3(ref_dirs=ref_dirs,
                                    pop_size=SIZE,
                                    sampling=MixedVariableSampling(),
                                    mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                                    eliminate_duplicates=MixedVariableDuplicateElimination())
            elif OPTALG == "CTAEA":
                algorithm = CTAEA(ref_dirs=ref_dirs,
                                sampling=MixedVariableSampling(),
                                mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                                eliminate_duplicates=MixedVariableDuplicateElimination())
            elif OPTALG == "RVEA":
                algorithm = RVEA(ref_dirs=ref_dirs,
                                sampling=MixedVariableSampling(),
                                mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                                eliminate_duplicates=MixedVariableDuplicateElimination())
            elif OPTALG == "SMSEMOA":
                algorithm = SMSEMOA(pop_size=SIZE,
                                sampling=MixedVariableSampling(),
                                mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                                eliminate_duplicates=MixedVariableDuplicateElimination())
            # elif OPTALG == "AGEMOEA":
            #     algorithm = AGEMOEA(pop_size=SIZE,
            #                     sampling=MixedVariableSampling(),
            #                     mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            #                     eliminate_duplicates=MixedVariableDuplicateElimination())
                
            # Record start time
            start_time = time.time()
            res = minimize(problem,
                            algorithm,
                            ('n_gen', ITERATIONS),
                            seed=SEED,
                            save_history=False,
                            verbose=VERBOSE)

            result_count = res.X.size
            #result_X = res.X
            #result_F = res.F

            log_results(run, problem.unsatisfied_reqs, problem.conjunction, problem.min_scores, time.time() - start_time)
            uns_reqs_df.loc[run] = problem.unsatisfied_reqs + [problem.conjunction]
            score_df.loc[run] = problem.min_scores

            # Save all evaluations for surrogate model training
            if LOGDIR is not None:
                # Save all X (parameters)
                X_df = pd.DataFrame(problem.all_X, columns=["car_speed", "p_x", "p_y", "orientation", "weather", "road_shape"])
                X_df.to_csv(f'{LOGDIR}/X_all_evaluations_{OPTALG}_{run}.csv', index=False)

                # Save all F (objectives/scores)
                F_df = pd.DataFrame(problem.all_F, columns=[f'V{i}' for i in range(OBJECTIVES)])
                F_df.to_csv(f'{LOGDIR}/F_all_evaluations_{OPTALG}_{run}.csv', index=False)

                # Save all requirements satisfaction data
                reqs_df = pd.DataFrame(problem.all_reqs, columns=[f'R{i}' for i in range(NREQS)])
                reqs_df.to_csv(f'{LOGDIR}/Reqs_all_evaluations_{OPTALG}_{run}.csv', index=False)

    if LOGDIR is not None:
        uns_reqs_df.to_csv(f'{LOGDIR}/reqs_{OPTALG}_{RUNS}.csv', index=False)
        score_df.to_csv(f'{LOGDIR}/score_{OPTALG}_{RUNS}.csv', index=False)

if __name__ == "__main__":
    main()
