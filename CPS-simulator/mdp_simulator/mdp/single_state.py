import numpy as np

from mdp_simulator.mdp.action import Action
from mdp_simulator.utils.context import Context
from mdp_simulator.utils.enums import StateType, Topics
import mdp_simulator.config as config


class SingleState:
    _id = None
    _name = None
    _actions = None
    _best_policy = None
    _state_type = None
    _is_terminal = None
    _constraint_sat_history: np.ndarray = None

    # Constructor
    def __init__(self) -> None:
        self._is_terminal = False
        self._constraint_sat_history = np.zeros(1, int)

    # Methods
    def set_best_policy(self, action_name):
        if self.get_action(action_name) is not None:
            self._best_policy = action_name

    def get_best_policy(self):
        return self._best_policy

    def add_action(self, action_name, probability, next_state_id) -> None:
        # Case 1: dictionary is not initialized
        if self._actions is None:
            if probability == 1.0 and next_state_id == self._id:
                self._is_terminal = True
            self._actions = dict(
                {action_name: Action(probability, next_state_id)})
            return

        # The dictionary is already initialized
        # (More than one possible action hence it's a non-final state)
        self._is_terminal = False
        old_actions: Action = self._actions.get(action_name)

        # Case 2: key not initialized
        if old_actions is None:
            self._actions.update(
                {action_name: Action(probability, next_state_id)})
            return

        # Case 3: key already initialized
        old_actions.add_action(probability, next_state_id)
        # self._actions.update({actionName: old_actions})

    def is_terminal(self) -> bool:
        return self._is_terminal

    def has_prior(self, action_name) -> bool:
        return self._actions.get(action_name).has_prior()

    def has_new_values(self, action_name) -> bool:
        return self._actions.get(action_name).has_new_values()

    def add_prior(self, action_name: str, prior: np.ndarray, eq_constraint: np.ndarray) -> None:
        Context.emit(Topics.DEBUG, f"Adding prior in {self._id} for {action_name} with {prior} and {eq_constraint}")

        if self._actions.get(action_name) is None or self._actions.get(action_name).has_prior():
            raise Exception(
                f"Initializing too many Prior for the same action. State: {self._id} Action:{action_name}")

        self._actions.get(action_name).add_prior(prior, eq_constraint)
        Context.emit(Topics.DEBUG, f"Prior added in {self._id} for {action_name} with {prior} and {eq_constraint}")

    def increment_prior(self, action_name, index):
        self._actions.get(action_name).increment_prior(index)

    def compute_posterior(self, action_name):
        self._actions.get(action_name).compute_posterior()

    def check_constraints_satisfied(self, action_name):
        Context.emit(Topics.DEBUG, f"Checking constraints for combination {self._id} - {action_name}")
        constraints_satisfied = self._actions.get(action_name).check_constraints_satisfied()

        self._constraint_sat_history = np.vstack(([not constraints_satisfied], self._constraint_sat_history))
        self._constraint_sat_history = self._constraint_sat_history[:config.CONSTRAINT_SAT_MAX_HISTORY]
        perc = self._constraint_sat_history.sum()

        result = perc < config.CONSTRAINT_SAT_PERC

        Context.emit(Topics.DEBUG, f"Constraints check done for combination {self._id} - {action_name} -> SAT:{result}")
        return result

    # Getters and Setters

    def set_id(self, value: str) -> None:
        self._id = value
        pass

    def get_id(self) -> str:
        return self._id

    def set_name(self, value: str) -> None:
        self._name = value
        pass

    def get_name(self) -> str:
        return self._name

    def set_state_type(self, state_type: str) -> None:
        try:
            self._state_type = StateType[state_type]
        except KeyError:
            raise Exception(f"Expecting a StateType enum not {state_type}")

    def get_state_type(self) -> StateType:
        return self._state_type

    def get_actions(self) -> dict:
        return self._actions

    def get_action(self, action_id) -> Action:
        return self._actions.get(action_id)

    # Debug
    def debug(self):
        print(
            f"id: {self._id} - name: {self._name} - actions: {len(self._actions)}")
