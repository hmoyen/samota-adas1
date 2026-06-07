import numpy as np
import random
import sys
import os

# Ensure we import the parent directory's config.py
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_UTILS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Add CPS-simulator to path for mdp_simulator imports
# Path: icse2025_replication_package/online-step-experiments/ADAS1/utils
# Need to go: utils -> ADAS1 -> online-step-experiments -> icse2025_replication_package -> CPS-simulator
_ICSE_DIR = os.path.dirname(os.path.dirname(_PARENT_DIR))  # Go up to icse2025_replication_package
_CPS_SIM_DIR = os.path.join(_ICSE_DIR, "CPS-simulator")
if _CPS_SIM_DIR not in sys.path:
    sys.path.insert(0, _CPS_SIM_DIR)

import config as conf


def create_ss_variables(ss_variables: dict, input_variables: np.ndarray) -> dict:
    variables_to_exec = dict()
    for index, key in enumerate(sorted(ss_variables.keys())):
        value = ss_variables[key]
        variables_to_exec.update({key: value['domain'](input_variables[index])})

    return variables_to_exec


def build_random_combinations(expected_maximum):
    combinations = set()

    while len(combinations) < expected_maximum:
        new_combination = list()
        for key in sorted(conf.SS_VARIABLES.keys()):
            variable = conf.SS_VARIABLES[key]
            bound = variable['range']
            domain = variable['domain']
            if domain is int:
                new_value = random.randint(bound[0], bound[1])
            else:
                new_value = round(random.uniform(bound[0], bound[1]), 4)
            new_combination.append(new_value)
        combinations.add(tuple(new_combination))

    return combinations


def run_mdp(input_variables):
    from mdp_simulator import config, run, enums

    # set configuration
    if conf.PLOT:
        config.OUTPUT_FOLDER_NAME = f"{conf.RQ}_{conf.EXTRA_NAME}"
        config.PLOT_DPI = 100
    config.MAX_PARTIAL_POSTERIOR_RETRIES = conf.HISTORY_RETRIES
    config.MAX_HISTORY_PARTIAL_POSTERIOR = conf.HISTORY_LEN  # 100
    config.FOLDER_NAME = conf.MDP_FOLDER
    config.BATCH_SIZE = conf.BATCH_SIZE
    config.DEBUG_LEVEL = enums.LogTypes.ERROR
    config.MAX_STEPS = conf.MAX_STEPS

    # set_changes_config(local_duration, local_magnitude)
    override_ss_variables_starting_value = create_ss_variables(conf.SS_VARIABLES, input_variables)

    # run the mdp
    result_mdp = run(override_ss_variables_starting_value=override_ss_variables_starting_value)

    reqs_satisfied = []
    for constraint in conf.CONSTRAINTS:
        reqs_satisfied.append(check_requirement(result_mdp, constraint))

    return (input_variables, region_scores(result_mdp, conf.MINIMAL_CONSTRAINTS), reqs_satisfied, check_conjunction(result_mdp, conf.MINIMAL_CONSTRAINTS))

def run_mdp_sensitivity(input_variables):
    from mdp_simulator import config, run, enums

    # set configuration
    if conf.PLOT:
        config.OUTPUT_FOLDER_NAME = f"{conf.RQ}_{conf.EXTRA_NAME}"
        config.PLOT_DPI = 100
    config.MAX_PARTIAL_POSTERIOR_RETRIES = conf.HISTORY_RETRIES
    config.MAX_HISTORY_PARTIAL_POSTERIOR = conf.HISTORY_LEN  # 100
    config.FOLDER_NAME = conf.MDP_FOLDER
    config.BATCH_SIZE = conf.BATCH_SIZE
    config.DEBUG_LEVEL = enums.LogTypes.ERROR
    config.MAX_STEPS = conf.MAX_STEPS

    # set_changes_config(local_duration, local_magnitude)
    override_ss_variables_starting_value = create_ss_variables(conf.SS_VARIABLES, input_variables)

    # run the mdp
    result_mdp = run(override_ss_variables_starting_value=override_ss_variables_starting_value)

    reqs_min_score = []
    for constraint in conf.CONSTRAINTS:
        reqs_min_score.append(min_score_requirement(result_mdp, constraint))
    reqs_satisfied = []
    for constraint in conf.CONSTRAINTS:
        reqs_satisfied.append(check_requirement(result_mdp, constraint))

    return (input_variables, region_scores(result_mdp, conf.MINIMAL_CONSTRAINTS), reqs_satisfied, reqs_min_score, check_conjunction(result_mdp, conf.MINIMAL_CONSTRAINTS))

def run_mdp_2(input_variables):
    from mdp_simulator import config, run, enums

    # set configuration
    if conf.PLOT:
        config.OUTPUT_FOLDER_NAME = f"{conf.RQ}_{conf.EXTRA_NAME}"
        config.PLOT_DPI = 100
    config.MAX_PARTIAL_POSTERIOR_RETRIES = conf.HISTORY_RETRIES
    config.MAX_HISTORY_PARTIAL_POSTERIOR = conf.HISTORY_LEN  # 100
    config.FOLDER_NAME = conf.MDP_FOLDER
    config.BATCH_SIZE = conf.BATCH_SIZE
    config.DEBUG_LEVEL = enums.LogTypes.ERROR
    config.MAX_STEPS = conf.MAX_STEPS

    # set_changes_config(local_duration, local_magnitude)
    override_ss_variables_starting_value = create_ss_variables(conf.SS_VARIABLES, input_variables)

    # run the mdp
    result_mdp = run(override_ss_variables_starting_value=override_ss_variables_starting_value)

    reqs_satisfied = []
    for constraint in conf.CONSTRAINTS:
        reqs_satisfied.append(check_requirement(result_mdp, constraint))

    return list(input_variables) + reqs_satisfied

def check_requirement(mdp, single_constraint):
    from mdp_simulator import Action, SingleState
    states = mdp.get_states_dictionary()

    for state_id, state_constraints in single_constraint.items():
        state:SingleState = states.get(state_id)
        for action_id, bounds in state_constraints.items():
            action:Action = state.get_action(action_id)
            point_estimates = action.get_expected()

            for estimate, bound in zip(point_estimates, bounds):
                if estimate < bound[0] or estimate > bound[1]:
                    return False
    return True

def min_score_requirement(mdp, single_constraint):
    from mdp_simulator import Action, SingleState
    states = mdp.get_states_dictionary()
    min_score = 1.0

    for state_id, state_constraints in single_constraint.items():
        state:SingleState = states.get(state_id)
        for action_id, bounds in state_constraints.items():
            action:Action = state.get_action(action_id)
            point_estimates = action.get_expected()

            for estimate, bound in zip(point_estimates, bounds):
                if bound[0] <= estimate and estimate <= bound[1]:
                    if bound[0] == 0.0:
                        score = abs(bound[1]-estimate)
                    elif bound[1] == 1.0:
                        score = abs(bound[0]-estimate)
                    else:
                        score = min([abs(bound[0]-estimate), abs(bound[1]-estimate)])
                else: # bound[0] > estimate or bound[1] < estimate:
                    score = -min([abs(bound[0]-estimate), abs(bound[1]-estimate)])
                if score < min_score:
                    min_score = score
    return min_score

def check_conjunction(mdp, single_constraint):
    from mdp_simulator import Action, SingleState
    states = mdp.get_states_dictionary()
    scores = []

    for state_id, state_constraints in single_constraint.items():
        state:SingleState = states.get(state_id)
        for action_id, bounds in state_constraints.items():
            action:Action = state.get_action(action_id)
            #hdi_intervals = action.get_hdi()
            point_estimates = action.get_expected()
            for estimate, bound in zip(point_estimates, bounds):
                if bound[0] > estimate or bound[1] < estimate:
                    return 1
    return 0

def region_scores(mdp, single_constraint):
    from mdp_simulator import Action, SingleState
    states = mdp.get_states_dictionary()
    scores = []

    for state_id, state_constraints in single_constraint.items():
        state:SingleState = states.get(state_id)
        for action_id, bounds in state_constraints.items():
            action:Action = state.get_action(action_id)
            #hdi_intervals = action.get_hdi()
            point_estimates = action.get_expected()
            for estimate, bound in zip(point_estimates, bounds):
                if bound[0] <= estimate and estimate <= bound[1]:
                    if bound[0] == 0.0:
                        scores.append(abs(bound[1]-estimate))
                    elif bound[1] == 1.0:
                        scores.append(abs(bound[0]-estimate))
                    else:
                        scores.append(min([abs(bound[0]-estimate), abs(bound[1]-estimate)]))
                else: # bound[0] > estimate or bound[1] < estimate:
                    scores.append(-min([abs(bound[0]-estimate), abs(bound[1]-estimate)]))
    return scores

def raw_point_estimates(mdp, single_constraint):
    """
    Extract RAW simulator outputs (point_estimates) WITHOUT constraint mapping.

    Used for SAMOTA surrogates which learn better with natural value ranges.

    Returns: List of raw point estimates for each constraint
    """
    from mdp_simulator import Action, SingleState
    states = mdp.get_states_dictionary()
    raw_estimates = []

    for state_id, state_constraints in single_constraint.items():
        state:SingleState = states.get(state_id)
        for action_id, bounds in state_constraints.items():
            action:Action = state.get_action(action_id)
            point_estimates = action.get_expected()
            # Return raw estimates, not processed through constraint mapping
            raw_estimates.extend(point_estimates)

    return raw_estimates
