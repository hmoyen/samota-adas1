import click
import random
import utils.helpers as helpers
import config as conf
import pandas as pd
import time
from copy import deepcopy

THREADS_COUNT = 1
conf.MAX_STEPS = 20000
conf.BATCH_SIZE = 100

conf.MDP_FOLDER = "INPUT/AutonomousDriving_v1"
conf.PLOT = False
conf.MAX_SAMPLES = 100

class QTable:
    def __init__(self, req_id, variables, n_actions, learning_rate = 0.01, discount = 0.9):
        self.req_id = req_id
        self.vars = variables
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.min_score = 0
        self.high_reward = 1000000
        self.table = {}

    def best_action(self, state):
        scores = self.table.get(str(state))
        if scores is None:
            return random.randint(0, self.n_actions-1), self.min_score
        else:
            return scores.index(max(scores)), max(scores)
    
    def max_score(self, state):
        scores = self.table.get(str(state))
        if scores is None:
            return self.min_score
        else:
            return max(scores)
    
    def score(self, state, action):
        scores = self.table.get(str(state))
        if scores is None:
            return self.min_score
        else:
            return scores[action]
        
    def update_value(self, state, action, value):
        key = str(state)
        scores = self.table.get(key)
        if scores is None:
            scores = [self.min_score] * self.n_actions
        scores[action] = value
        self.table[key] = scores
    
    def compute_score(self, state, action, new_state, req_scores):
        score = req_scores[self.req_id]
        reward = self.high_reward if score <= 0 else 1 / score
        new_score = self.score(state, action) + self.learning_rate * (reward + self.discount * self.max_score(new_state) - self.score(state, action))
        self.update_value(state, action, new_score)


# actions

ACTION_DISC = 4

def a_car_speed_up(ss_values):
    new_ss_values = deepcopy(ss_values)
    var_name = "car_speed"
    step = (conf.SS_VARIABLES[var_name]["range"][1] - conf.SS_VARIABLES[var_name]["range"][0]) / ACTION_DISC
    limit = conf.SS_VARIABLES[var_name]["range"][1]
    var_index = 0
    new_ss_values[var_index] = new_ss_values[var_index] + step
    if new_ss_values[var_index] > limit:
        new_ss_values[var_index] = limit
    return new_ss_values

def a_car_speed_down(ss_values):
    new_ss_values = deepcopy(ss_values)
    var_name = "car_speed"
    step = (conf.SS_VARIABLES[var_name]["range"][1] - conf.SS_VARIABLES[var_name]["range"][0]) / ACTION_DISC
    limit = conf.SS_VARIABLES[var_name]["range"][0]
    var_index = 0
    new_ss_values[var_index] = new_ss_values[var_index] - step
    if new_ss_values[var_index] < limit:
        new_ss_values[var_index] = limit
    return new_ss_values

def a_p_x_up(ss_values):
    new_ss_values = deepcopy(ss_values)
    var_name = "p_x"
    step = (conf.SS_VARIABLES[var_name]["range"][1] - conf.SS_VARIABLES[var_name]["range"][0]) / ACTION_DISC
    limit = conf.SS_VARIABLES[var_name]["range"][1]
    var_index = 1
    new_ss_values[var_index] = new_ss_values[var_index] + step
    if new_ss_values[var_index] > limit:
        new_ss_values[var_index] = limit
    return new_ss_values

def a_p_x_down(ss_values):
    new_ss_values = deepcopy(ss_values)
    var_name = "p_x"
    step = (conf.SS_VARIABLES[var_name]["range"][1] - conf.SS_VARIABLES[var_name]["range"][0]) / ACTION_DISC
    limit = conf.SS_VARIABLES[var_name]["range"][0]
    var_index = 1
    new_ss_values[var_index] = new_ss_values[var_index] - step
    if new_ss_values[var_index] < limit:
        new_ss_values[var_index] = limit
    return new_ss_values

def a_p_y_up(ss_values):
    new_ss_values = deepcopy(ss_values)
    var_name = "p_y"
    step = (conf.SS_VARIABLES[var_name]["range"][1] - conf.SS_VARIABLES[var_name]["range"][0]) / ACTION_DISC
    limit = conf.SS_VARIABLES[var_name]["range"][1]
    var_index = 2
    new_ss_values[var_index] = new_ss_values[var_index] + step
    if new_ss_values[var_index] > limit:
        new_ss_values[var_index] = limit
    return new_ss_values

def a_p_y_down(ss_values):
    new_ss_values = deepcopy(ss_values)
    var_name = "p_y"
    step = (conf.SS_VARIABLES[var_name]["range"][1] - conf.SS_VARIABLES[var_name]["range"][0]) / ACTION_DISC
    limit = conf.SS_VARIABLES[var_name]["range"][0]
    var_index = 2
    new_ss_values[var_index] = new_ss_values[var_index] - step
    if new_ss_values[var_index] < limit:
        new_ss_values[var_index] = limit
    return new_ss_values

def a_orientation_up(ss_values):
    new_ss_values = deepcopy(ss_values)
    var_name = "orientation"
    step = (conf.SS_VARIABLES[var_name]["range"][1] - conf.SS_VARIABLES[var_name]["range"][0]) / ACTION_DISC
    limit = conf.SS_VARIABLES[var_name]["range"][1]
    var_index = 3
    new_ss_values[var_index] = new_ss_values[var_index] + step
    if new_ss_values[var_index] > limit:
        new_ss_values[var_index] = limit
    return new_ss_values

def a_orientation_down(ss_values):
    new_ss_values = deepcopy(ss_values)
    var_name = "orientation"
    step = (conf.SS_VARIABLES[var_name]["range"][1] - conf.SS_VARIABLES[var_name]["range"][0]) / ACTION_DISC
    limit = conf.SS_VARIABLES[var_name]["range"][0]
    var_index = 3
    new_ss_values[var_index] = new_ss_values[var_index] - step
    if new_ss_values[var_index] < limit:
        new_ss_values[var_index] = limit
    return new_ss_values

def a_weather_up(ss_values):
    new_ss_values = deepcopy(ss_values)
    var_name = "weather"
    step = 1
    limit = conf.SS_VARIABLES[var_name]["range"][1]
    var_index = 4
    new_ss_values[var_index] = new_ss_values[var_index] + step
    if new_ss_values[var_index] > limit:
        new_ss_values[var_index] = limit
    return new_ss_values

def a_weather_down(ss_values):
    new_ss_values = deepcopy(ss_values)
    var_name = "weather"
    step = 1
    limit = conf.SS_VARIABLES[var_name]["range"][0]
    var_index = 4
    new_ss_values[var_index] = new_ss_values[var_index] - step
    if new_ss_values[var_index] < limit:
        new_ss_values[var_index] = limit
    return new_ss_values

def a_road_shape_up(ss_values):
    new_ss_values = deepcopy(ss_values)
    var_name = "road_shape"
    step = 1
    limit = conf.SS_VARIABLES[var_name]["range"][1]
    var_index = 5
    new_ss_values[var_index] = new_ss_values[var_index] + step
    if new_ss_values[var_index] > limit:
        new_ss_values[var_index] = limit
    return new_ss_values

def a_road_shape_down(ss_values):
    new_ss_values = deepcopy(ss_values)
    var_name = "road_shape"
    step = 1
    limit = conf.SS_VARIABLES[var_name]["range"][0]
    var_index = 5
    new_ss_values[var_index] = new_ss_values[var_index] - step
    if new_ss_values[var_index] < limit:
        new_ss_values[var_index] = limit
    return new_ss_values

# helpers

def log_results(n_run, unsatisfied_reqs, unsatisfied_conjunction, best_scores, duration):
    print("\nTotal Duration run {}: {} seconds".format(n_run, duration))
    print("\n\nReqs unsatisfied:")
    print(unsatisfied_reqs)
    print("\n\nReqs conjunction:")
    print(unsatisfied_conjunction)
    print("\n\nBest scores:")
    print(best_scores)

def random_assignment(ss_vars):
    assignment = []
    for v in ss_vars:
        lb = conf.SS_VARIABLES[v]["range"][0]
        ub = conf.SS_VARIABLES[v]["range"][1]
        if conf.SS_VARIABLES[v]["domain"] == int:
            assignment += [random.randint(lb, ub)]
        else:
            assignment += [random.uniform(lb, ub)]
    return assignment

def exploration_proba(cur_iteration, max_iterations, target_proba):
    warmup_limit = (max_iterations // 100) * 20
    if cur_iteration > warmup_limit:
        return target_proba
    return 1 - ((cur_iteration * (1 - target_proba)) / warmup_limit) 

def get_state(variables, assignment, discretization):
    state = [0] * len(variables)
    for i in range(len(state)):
        v = variables[i]
        interval = conf.SS_VARIABLES[v]["range"]
        step = (interval[1] - interval[0]) / discretization[i]
        state[i] = assignment[i] // step
    return state

def get_actions(assignment, all_actions):
    available_actions = set(range(0, len(all_actions)))
    for i in range(0, len(all_actions)):
        if all_actions[i](assignment) == assignment:
            available_actions.remove(i)
    return available_actions


@click.command()
@click.option('--nepisodes', default=900, help='Iterations.', type=int)
@click.option('--nruns', default=1, help='Runs.', type=int)
@click.option('--verbose', default=False, help='Verbose.', type=bool)
@click.option('--logdir', default="out", help='Log directory.', type=str)
def main(nepisodes, nruns, verbose, logdir):
    SEED = 1
    RUNS = nruns
    ITERATIONS = nepisodes
    VERBOSE = verbose
    LOGDIR = logdir
    TARGET_EXPLORATION_PROBA = 0.1
    
    NREQS = 6
    OBJECTIVES = 5

    uns_reqs_df = pd.DataFrame(columns=[f'R{j}' for j in range(0, NREQS)] + ["conjunction"])
    score_df = pd.DataFrame(columns=[f'V{j}' for j in range(0, OBJECTIVES)])
    
    ss_vars = ["car_speed", "p_x", "p_y", 
               "orientation", "weather", "road_shape"]
    disc_step = 8
    discretization = [disc_step, disc_step, disc_step,
                      disc_step, 2, 2]

    actions = [a_car_speed_up, a_car_speed_down,
               a_p_x_up, a_p_x_down,
               a_p_y_up, a_p_y_down,
               a_orientation_up, a_orientation_down,
               a_weather_up, a_weather_down,
               a_road_shape_up, a_road_shape_down]

    for run in range(0, RUNS):
    
        q_tables = [None] * OBJECTIVES
        for i in range(0, OBJECTIVES):
            q_tables[i] = QTable(i, ss_vars, len(actions))

        min_scores = [1] * OBJECTIVES
        unsatisfied_reqs = [0] * NREQS
        unsatisfied_conjunction = 0

        start_time = time.time()

        selected_qtab = None
        covered_objs = set()
        cur_assignment = random_assignment(ss_vars)
        for k in range(ITERATIONS):
            p_rand_action = exploration_proba(k, ITERATIONS, TARGET_EXPLORATION_PROBA)
            cur_state = get_state(ss_vars, cur_assignment, discretization)
            available_actions = get_actions(cur_assignment, actions)
            cur_action = random.choice(tuple(available_actions))
            if random.uniform(0, 1) > p_rand_action and selected_qtab is not None:
                best_score = 0
                for a in available_actions:
                    tmp_score = selected_qtab.score(cur_state, a)
                    if tmp_score > best_score:
                        cur_action = a
                        best_score = tmp_score

            new_assignment = actions[cur_action](cur_assignment)
            new_state = get_state(ss_vars, new_assignment, discretization)
            
            #print(str(new_assignment))
            
            #_, scores, reqs_satisfied, reqs_min_score, conjunction = helpers.run_mdp_sensitivity(new_assignment)
            _, scores, reqs_satisfied, conjunction = helpers.run_mdp(new_assignment)

            #print(str(scores))

            for i in range(0, len(min_scores)):
                if scores[i] < min_scores[i]:
                    min_scores[i] = scores[i]
                if scores[i] <= 0:
                    covered_objs.add(i)
            for i in range(0, len(unsatisfied_reqs)):
                if not reqs_satisfied[i]:
                    unsatisfied_reqs[i] +=1
            unsatisfied_conjunction += conjunction

            for t in q_tables:
                t.compute_score(cur_state, cur_action, new_state, scores)
            
            # select q-table base on best score
            o_index = random.randint(0, len(q_tables)-1)
            best_score = q_tables[o_index].score(cur_state, cur_action)
            for i in range(0, len(q_tables)):
                cur_score = q_tables[i].score(cur_state, cur_action)
                if cur_score > best_score and i not in covered_objs:
                    o_index = i
                    best_score = cur_score
            selected_qtab = q_tables[o_index]
            #selected_qtab = q_tables[1]
            
            cur_assignment = new_assignment
        
        log_results(run, unsatisfied_reqs, unsatisfied_conjunction, min_scores, time.time() - start_time)
        uns_reqs_df.loc[run] = unsatisfied_reqs + [unsatisfied_conjunction]
        score_df.loc[run] = min_scores

        if LOGDIR is not None:
            uns_reqs_df.to_csv(f'{LOGDIR}/reqs_MORLOT_{RUNS}.csv', index=False)
            score_df.to_csv(f'{LOGDIR}/score_MORLOT_{RUNS}.csv', index=False)

if __name__ == "__main__":
    main()
