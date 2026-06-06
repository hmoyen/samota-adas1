from mdp_simulator.mdp.action import Action
from mdp_simulator.mdp.semantic_space_variable import SemanticSpaceVariable
from mdp_simulator.mdp.single_state import SingleState


class SemanticSpace:
    def __init__(self, all_variables: [SemanticSpaceVariable], starting_value: dict, states: dict[{SingleState}]):
        self._all_variables = all_variables
        self._variables = dict()
        self._combinations = dict()
        self._states = states

        # Populate all the dictionaries
        for variable in all_variables:
            variable: SemanticSpaceVariable
            combinations = variable.get_combinations()
            name = variable.get_name()

            if self._variables.get(name) is not None:
                raise Exception("Multiple initialization of the same SS Variable {}".format(name))

            self._variables.update({name: variable})

            for combination in combinations:
                state_id = combination.get("StateId")
                action_id = combination.get("ActionId")

                # state id not yet initialized
                if self._combinations.get(state_id) is None:
                    self._combinations.update({state_id: {action_id: [variable]}})
                else:
                    # action id not yet initialized
                    if self._combinations.get(state_id).get(action_id) is None:
                        self._combinations.get(state_id).update({action_id: [variable]})
                    else:
                        self._combinations.get(state_id).get(action_id).append(variable)

        # Set a custom starting value (if specified)
        for var_name, var_value in starting_value.items():
            if self._variables.get(var_name) is None:
                raise Exception("Trying to assign a value to an nonexistent SS Variable: {}".format(var_name))
            else:
                self._variables.get(var_name).set_value(var_value)

        # Once loaded all the variables, Update all the probabilities touched
        for state_id, action_ids in self._combinations.items():
            for action_id in action_ids:
                self.update_combination(state_id, action_id)

    def update_variable_value(self, variable_name, variable_value):
        if self._variables.get(variable_name) is None:
            raise Exception("Trying to access an nonexistent SS Variable {}".format(variable_name))

        self._variables.get(variable_name).set_value(variable_value)

        # Recompute all the combination relative to this variable
        combinations_to_update = self._variables.get(variable_name).get_combinations()

        for combination in combinations_to_update:
            state_id = combination.get("StateId")
            action_id = combination.get("ActionId")

            self.update_combination(state_id, action_id)

    def update_combination(self, state_id, action_id):
        # Only update present combinations
        if self._combinations.get(state_id) is None:
            return
        elif self._combinations.get(state_id).get(action_id) is None:
            return

        # Load the action to update
        state_to_update: SingleState = self._states.get(state_id)

        if state_to_update is None:
            raise Exception("Updating SS for a non existing state: {}".format(state_id))

        action_to_update: Action = state_to_update.get_action(action_id)

        if action_to_update is None:
            raise Exception("Updating SS for a non existing action: {} (state: {})".format(action_id, state_id))

        old_probability = action_to_update.get_starting_probabilities()
        computed_probability = old_probability.copy()

        # For each semantic space variable that can modify that probability
        for variable in self._combinations.get(state_id).get(action_id):
            variable: SemanticSpaceVariable
            computed_probability = variable.compute_probability(state_id, action_id, computed_probability)

        action_to_update.set_probabilities(computed_probability, semantic_space_update=True)

    def get_values(self):
        result = dict()
        for name, var in self._variables.items():
            var: SemanticSpaceVariable
            value = var.get_value()
            result.update({name: value})

        return result

    def get_variables(self):
        return self._variables
