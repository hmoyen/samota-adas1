import mdptoolbox.mdp
import numpy as np

from mdp_simulator.utils.helpers import random_selector
from mdp_simulator.mdp.single_state import SingleState
from mdp_simulator.mdp.semantic_space import SemanticSpace
from mdp_simulator.mdp.action import Action
from mdp_simulator.utils.context import Context
from mdp_simulator.utils.enums import EquilibriumPolicy, Termination, StateType, Topics

CONVERSION_LOW_REWARD = -1000
POLICY_ITERATION_DISCOUNT = 0.96


class MDP:
    _startingStateId: str = None
    _currentStateId: str = None
    _statesDictionary: dict = None
    _policy = None
    _terminationCondition = None
    _terminationValue = None
    _steps: int = 0
    _changes = None
    _ss_changes = None

    # Constructor
    def __init__(self, states_dictionary: dict, termination_condition: Termination, termination_value: int,
                 policy: EquilibriumPolicy, initial_policy: list, changes: dict, semantic_space: SemanticSpace,
                 ss_changes: dict) -> None:
        self._startingStateId = list(states_dictionary.keys())[0]
        self._currentStateId = list(states_dictionary.keys())[0]
        self._statesDictionary = states_dictionary
        self._terminationCondition = termination_condition
        self._terminationValue = termination_value
        self._policy = policy
        self._changes = changes
        self._ss_changes = ss_changes
        self._semantic_space = semantic_space

        if len(initial_policy) > 0:
            self.__import_str_policy(initial_policy)

    # Methods
    def run(self) -> None:
        Context.emit(Topics.DEBUG, f"Running sanity check")
        self.__sanity_check()

        Context.emit(Topics.INFO, "Starting execution with policy {} from state {}".format(self._policy.name,
                                                                                           self._startingStateId))

        self.__compute_next_state()

    def get_starting_state_id(self) -> str:
        return self._startingStateId

    def get_current_state_id(self) -> str:
        return self._currentStateId

    def get_states_dictionary(self) -> dict:
        return self._statesDictionary

    def get_termination_condition(self) -> Termination:
        return self._terminationCondition

    def get_termination_value(self) -> int:
        return self._terminationValue

    def get_policy(self) -> EquilibriumPolicy:
        return self._policy

    def get_changes(self) -> dict:
        return self._changes

    def get_ss_changes(self) -> dict:
        return self._ss_changes

    def get_semantic_space(self) -> SemanticSpace:
        return self._semantic_space

    # Private

    def __import_str_policy(self, str_policy: list, intermediate_values=None) -> None:
        if intermediate_values is None:
            _, _, intermediate_values = self.__to_mdp_toolbox()

        if intermediate_values.get("states_count") != len(str_policy):
            raise Exception(f"Importing an inconsistent policy: {str_policy} [Different states count]")

        int_policy = []

        for state_int, action_key in enumerate(str_policy):
            try:
                state_id = intermediate_values.get("index_to_state").get(state_int)
                action_index = int(intermediate_values.get("state_action_to_index").get(state_id).get(action_key))
            except TypeError:
                # raise Exception(f"Unrecognized action key {action_key}")
                action_index = -1
            int_policy.append(action_index)

        self.__import_policy(int_policy, intermediate_values)

    def __import_policy(self, complete_policy: list, intermediate_values=None):
        if intermediate_values is None:
            _, _, intermediate_values = self.__to_mdp_toolbox()

        if intermediate_values.get("states_count") != len(complete_policy):
            raise Exception(f"Importing an inconsistent policy: {complete_policy} [Different states count]")

        # update each state with their best policy
        for i, single_policy in zip(np.arange(len(complete_policy)), complete_policy):
            state_id = intermediate_values.get("index_to_state").get(i)
            action_key = intermediate_values.get("index_to_action").get(single_policy)
            state: SingleState = self._statesDictionary.get(state_id)
            if not state.is_terminal():
                state.set_best_policy(action_key)

    def __recompute_policy(self, algorithm, *args):
        # translate to mdp toolbox & compute the policy
        Context.emit(Topics.DEBUG, "Recomputing policy...")
        p_matrix, r_matrix, intermediate_values = self.__to_mdp_toolbox()
        try:
            mdp_policy = algorithm(p_matrix, r_matrix, *args)
            mdp_policy.run()
            self.__import_policy(mdp_policy.policy, intermediate_values)
        except Exception as e:
            Context.emit(Topics.ERROR, "The policy could not be computed using MDPToolbox")

        # update each state with their best policy
        # for i, new_policy in zip(np.arange(len(mdp_policy.policy)), mdp_policy.policy):
        #     state_id = intermediate_values.get("index_to_state").get(i)
        #     action_key = intermediate_values.get("index_to_action").get(new_policy)
        #     state: SingleState = self._statesDictionary.get(state_id)
        #     if not state.is_terminal():
        #         state.set_best_policy(action_key)

        Context.emit(Topics.DEBUG, "Equilibrium Policy recomputed.")

    def __to_mdp_toolbox(self) -> (np.ndarray, np.ndarray, dict):
        # 1: count the number of states and actions
        states_count = 0
        action_count = 0

        state_to_index = dict()
        state_action_to_index = dict()
        index_to_state = dict()
        index_to_action = dict()

        for elem in self._statesDictionary.items():
            state: SingleState = elem[1]
            state_id = state.get_id()
            # update state count
            state_to_index.update({state_id: states_count})
            index_to_state.update({states_count: state_id})
            states_count += 1

            # update action count
            keys = list(state.get_actions().keys())
            for key in keys:
                if state_action_to_index.get(state_id) is None:
                    state_action_to_index.update({state_id: dict()})
                action_to_index = state_action_to_index.get(state_id)

                action_to_index.update({key: action_count})
                index_to_action.update({action_count: key})
                action_count += 1

        # 2: build the probability and the reward matrices
        probability_matrix = np.zeros((action_count, states_count, states_count))
        reward_matrix = np.zeros((action_count, states_count, states_count))

        for i in range(action_count):
            np.fill_diagonal(probability_matrix[i], 1.0)
            np.fill_diagonal(reward_matrix[i], CONVERSION_LOW_REWARD)

        # 3: build the matrices state by state
        for elem in self._statesDictionary.items():
            state: SingleState = elem[1]
            state_id = state.get_id()
            state_index = state_to_index.get(state_id)

            for key in state.get_actions().keys():
                action: Action = state.get_action(key)
                action_index = state_action_to_index.get(state_id).get(key)

                probability_matrix[action_index][state_index] = np.zeros(
                    probability_matrix[action_index][state_index].shape)
                reward_matrix[action_index][state_index] = np.zeros(reward_matrix[action_index][state_index].shape)

                rewards = action.get_rewards()
                outcomes = action.get_outcomes()
                probabilities = action.get_probabilities()
                if state.has_prior(key):
                    probabilities = state.get_action(key).get_expected()
                for outcome, probability, reward in zip(outcomes, probabilities, rewards):
                    outcome_index = state_to_index.get(outcome)
                    probability_matrix[action_index][state_index][outcome_index] = probability
                    reward_matrix[action_index][state_index][outcome_index] = reward

        # mdptoolbox.util.check(probability_matrix, reward_matrix)
        return probability_matrix, reward_matrix, {"index_to_action": index_to_action,
                                                   "index_to_state": index_to_state,
                                                   "state_to_index": state_to_index,
                                                   "state_action_to_index": state_action_to_index,
                                                   "states_count": states_count,
                                                   "action_count": action_count}

    def __sanity_check(self) -> None:
        state_ids = np.array(list(self._statesDictionary.keys()))

        for stateId in state_ids:
            state: SingleState = self._statesDictionary.get(stateId)

            if state.is_terminal():
                continue

            action_keys = np.array(list(state.get_actions().keys()))
            for actionKey in action_keys:
                outcomes = state.get_action(actionKey).get_outcomes()
                probabilities = state.get_action(actionKey).get_probabilities()

                # All probabilities are equals to 1
                if probabilities.sum() != 1.0:
                    raise Exception(
                        f"State {stateId} has an inconsistency in the outcome probabilities given the"
                        f" action {actionKey}: total probability equal to {probabilities.sum()}")

                # Outcome is not empty
                if outcomes.size == 0:
                    raise Exception(
                        f"State {stateId} has an inconsistency in the outcomes given the action {actionKey}:"
                        f" no outcomes defined")

                # Outcome size different from probabilities size
                if outcomes.size != probabilities.size:
                    raise Exception(
                        f"State {stateId} has an inconsistency in the outcomes given the action {actionKey}:"
                        f" outcome and probabilities don't have the same size")

                # All outcome state_ids exist
                for outcome in outcomes:
                    if outcome not in state_ids:
                        raise Exception(
                            f"State {stateId} has an inconsistency in the outcomes given the action {actionKey}:"
                            f" next state {outcome} doesn't exist")

    @staticmethod
    def __exit() -> None:
        Context.emit(Topics.INFO, "Execution completed")

    def __get_current_state(self) -> SingleState:
        return self._statesDictionary.get(self._currentStateId)

    def __get_status(self) -> str:
        state_id = self.__get_current_state().get_id()
        state_name = self.__get_current_state().get_name()
        return f"Current state_id -> {state_id}, Current state_name -> {state_name}"

    def __compute_next_state(self) -> None:
        while True:
            # Check termination condition
            if self._terminationCondition == Termination.STEPS:
                if self._steps >= self._terminationValue:
                    # self.__on_end_condition_satisfied()
                    Context.emit(Topics.END_CONDITION_SATISFIED, self._statesDictionary)
                    self.__exit()
                    return

            # Check for changes to apply to the action probabilities
            if self._ss_changes.get(self._steps):
                Context.emit(Topics.INFO, "Changing semantic space variables values")
                for change in self._ss_changes.get(self._steps):
                    variable_name = change.get("name")
                    variable_value = change.get("value")
                    self._semantic_space.update_variable_value(variable_name, variable_value)
                    Context.emit(Topics.DEBUG, "Variable {} -> {}".format(variable_name, variable_value))

            if self._changes.get(self._steps):
                Context.emit(Topics.INFO, "Changing probability values")
                for change in self._changes.get(self._steps):
                    state: SingleState = self._statesDictionary.get(change.get("state"))
                    action: Action = state.get_action(change.get("action_key"))
                    action.set_probabilities(change.get("probabilities"))
                    self._semantic_space.update_combination(change.get("state"), change.get("action_key"))
                    Context.emit(Topics.DEBUG, "State {} - Action {} -> {}"
                                 .format(change.get('state'), change.get('action_key'),
                                         change.get('probabilities')))

            Context.emit(Topics.DEBUG, self.__get_status())

            # Get current state obj
            current_state = self.__get_current_state()

            # Start a new step
            Context.emit(Topics.START_NEW_STEP, current_state)

            # Update step count
            self._steps += 1

            # Restart the execution if it's terminal
            if current_state.is_terminal():
                Context.emit(Topics.TERMINAL_STATE)

                Context.emit(Topics.END_STEP)

                old_state_id = self._currentStateId
                old_state_name = self.__get_current_state().get_name()
                next_state_id = self._startingStateId

                Context.emit(Topics.STAT, "ID:{}|NAME:{}|TERMINAL STATE|NEXT STATE:{}|Step N:{}"
                             .format(old_state_id, old_state_name, next_state_id, self._steps))

                self._currentStateId = next_state_id
                continue

            # Get all the possible actions
            possible_actions = current_state.get_actions()

            # Select the next action based on the policy
            # If there isn't any best policy computed choose randomly
            Context.emit(Topics.SELECT_NEXT_ACTION)

            # Check if we have the best policy ONLY if the state is controllable
            if current_state.get_best_policy() is not None and current_state.get_state_type() == StateType.CONTROLLABLE:
                selected_action_key = current_state.get_best_policy()
                Context.emit(Topics.DEBUG, f"Selected best policy: {selected_action_key}")
            else:
                probability = 1 / len(possible_actions.keys())

                probabilities = np.empty(len(possible_actions.keys()))
                probabilities.fill(probability)

                selected_action_index = random_selector(probabilities)
                selected_action_key = list(possible_actions.keys())[
                    selected_action_index]
            Context.emit(Topics.SELECTED_NEXT_ACTION, current_state, selected_action_key)

            # Select the next state based on the probabilities
            Context.emit(Topics.SELECT_NEXT_STATE)
            outcomes: Action = possible_actions.get(selected_action_key)
            selected_outcome = random_selector(outcomes.get_probabilities())
            next_state_id = outcomes.get_outcomes()[selected_outcome]

            Context.emit(Topics.SELECTED_NEXT_STATE, next_state_id)

            Context.emit(Topics.DEBUG, f"Selected action {selected_action_key} and next state {next_state_id}")

            # Check if the selected action has prior
            if current_state.has_prior(selected_action_key):
                Context.emit(Topics.PRIOR_ACTION_SELECTED, current_state, selected_action_key, selected_outcome)
                # Print posterior values only if they are new
                if current_state.has_new_values(selected_action_key):
                    prior, expected, hdi, is_hdi_exceeded = current_state.get_action(
                        selected_action_key).get_posterior()

                    hdi = np.array2string(hdi).replace("\n", "")

                    Context.emit(Topics.DEBUG, f"Action {selected_action_key}|DIR{prior}|EXP:{expected}|HDI:{hdi}")

                    Context.emit(Topics.UPDATED_PRIOR_ACTION_SELECTED,
                                 current_state,
                                 selected_action_key,
                                 selected_outcome,
                                 is_hdi_exceeded)

                # Check the equilibrium constraint against the Posterior
                if not current_state.check_constraints_satisfied(selected_action_key):
                    Context.emit(Topics.EQ_CONSTRAINT_UNSATISFIED, current_state, selected_action_key)
                    if self._policy == EquilibriumPolicy.RANDOM:
                        # Do nothing
                        pass
                    elif self._policy == EquilibriumPolicy.POLICY_ITERATION:
                        self.__recompute_policy(mdptoolbox.mdp.PolicyIteration, POLICY_ITERATION_DISCOUNT)
                    Context.emit(Topics.POLICY_UPDATED, current_state)

            # End of step
            old_state_id = self._currentStateId
            old_state_name = self.__get_current_state().get_name()

            Context.emit(Topics.END_STEP)

            Context.emit(Topics.STAT, "ID:{}|NAME:{}|AVAILABLE ACTIONS:{}|SELECTED:{}|NEXT STATE:{}|Step N:{}"
                         .format(old_state_id, old_state_name, list(possible_actions.keys()), selected_action_key,
                                 next_state_id,
                                 self._steps))

            # Update current state
            self._currentStateId = next_state_id
