import os
import os.path
import random
import warnings
import numpy as np
import mdptoolbox.example
import importlib

from mdp_simulator.mdp.semantic_space_variable import SemanticSpaceVariable
from mdp_simulator.mdp.single_state import SingleState
from mdp_simulator.mdp.action import Action

# Ignore hdi future warning
# FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to
#   (chain, draw) for coherence with other functions
warnings.filterwarnings('ignore', message="hdi currently interprets 2d")

EPSILON = 0.25


def build_random_mdp(n_state, n_action, n_prior):
    result = None
    while result is None:
        try:
            result = __build_random_mdp(n_state, n_action, n_prior)
        except ValueError:
            pass
    return result


def __build_random_mdp(n_state, n_action, n_prior):
    p, _ = mdptoolbox.example.rand(n_state, n_action)

    all_states = []
    states_dict = dict()

    for state_index in range(n_state):
        new_state = SingleState()
        new_state.set_id(f"s{state_index}")
        new_state.set_name(f"State {state_index}")
        new_state.set_state_type("CONTROLLABLE")
        all_states.append(new_state)

    for action, action_index in zip(p, range(n_action)):
        action_id = f"{action_index}"
        for state, state_index in zip(action, range(n_state)):
            # print(f"{action_index} - {state_index} - {state}")
            for probability, next_state_index in zip(state, range(state.size)):
                next_state_id = all_states[next_state_index].get_id()
                all_states[state_index].add_action(action_id, probability, next_state_id)

    for state in all_states:
        states_dict.update({state.get_id(): state})

    prior_combinations = []

    for state in all_states:
        state_id = state.get_id()
        for action_key in state.get_actions().keys():
            prior_combinations.append([state_id, action_key])

    for _ in range(n_prior):
        if len(prior_combinations) <= 0:
            print("NO MORE COMBINATIONS")
            break
        extracted_value = random.choice(prior_combinations)
        extracted_index = prior_combinations.index(extracted_value)
        prior_combinations.pop(extracted_index)

        state_to_update: SingleState = states_dict.get(extracted_value[0])
        action_to_update: Action = state_to_update.get_action(extracted_value[1])
        probability_len = action_to_update.get_probabilities().size

        prior = np.ones(probability_len).astype(int)

        eq_constraint = []
        for probability in action_to_update.get_probabilities():
            eq_constraint.append([max(probability - EPSILON, 0), min(probability + EPSILON, 1)])
        eq_constraint = np.array(eq_constraint)

        state_to_update.add_prior(extracted_value[1], prior, eq_constraint)

    return states_dict


def random_selector(probabilities: np.ndarray) -> int:
    # Create an array with 1000 samples based on the probabilities in input
    return np.random.choice(np.arange(probabilities.size), p=probabilities)


def semantic_space_variables_importer(ss_variables_folder_path) -> list:
    if ss_variables_folder_path is None:
        return []

    assert isinstance(ss_variables_folder_path, str)

    # Load all the macro key setups from .py files in MACRO_FOLDER
    semantic_space_variables = []
    files = os.listdir(ss_variables_folder_path)
    files.sort()
    for filename in files:
        if filename.endswith('.py') and not filename.startswith('._'):
            try:
                module_name = filename[:-3]
                module_path = f"{ss_variables_folder_path}.{module_name}"
                module_path = module_path.replace("./", "")
                module_path = module_path.replace("/", ".")
                module_path = module_path.replace("\\", ".")

                module = importlib.import_module(module_path)
                for variable in module.SemanticSpaceVariable:
                    semantic_space_variables.append(SemanticSpaceVariable(variable))
            except (SyntaxError, ImportError, AttributeError, KeyError, NameError,
                    IndexError, TypeError) as err:
                print("ERROR in", filename)
                import traceback
                traceback.print_exception(err, err, err.__traceback__)

    return semantic_space_variables
