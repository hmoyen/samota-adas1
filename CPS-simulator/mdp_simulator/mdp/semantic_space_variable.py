import numpy as np
import inspect
from mdp_simulator.utils.probabilities import check_and_normalize_probabilities


def expected_method(old_probability: np.ndarray, variable_value) -> np.ndarray:
    # Used only to check the signature of imported methods
    pass


class SemanticSpaceVariable:
    _name = None
    _type = None
    _domain = None
    _range = None
    _combinations = None
    _value = None

    # Constructor
    def __init__(self, ss_variable_to_import: dict) -> None:
        """
        "Name": "Test_Elem",
        "Type": "ENV",
        "Domain": int,
        "Range": [0, 5],
        "Combinations": "StateId": "s0",
                        "ActionId": "a",
                        "Method": method(old_probability, variable_value)
        """
        if not isinstance(ss_variable_to_import, dict):
            raise Exception(f"Expecting a dictionary not {type(ss_variable_to_import)}")

        self._name = ss_variable_to_import.get("Name")
        self._type = ss_variable_to_import.get("Type")
        self._domain = ss_variable_to_import.get("Domain")
        self._range = ss_variable_to_import.get("Range")
        self._combinations = ss_variable_to_import.get("Combinations")
        self._default = ss_variable_to_import.get("Default")
        self._value = None

        if not isinstance(self._name, str):
            raise Exception("Importing a semantic space variable without a consistent Name")
        if not isinstance(self._type, str):
            raise Exception("Importing a semantic space variable without a consistent Type")
        if not (self._domain == int or self._domain == float):
            raise Exception("Importing a semantic space variable without a consistent Domain")
        if not isinstance(self._range, list):
            raise Exception("Importing a semantic space variable without a consistent Range")
        if self._range[0] > self._range[1]:
            raise Exception("Importing a semantic space variable without a consistent Range")
        if not isinstance(self._combinations, list):
            raise Exception("Importing a semantic space variable without a consistent Combination")
        if self._default is None:
            raise Exception("Importing a semantic space variable without a default value")

        for combination in self._combinations:
            if not isinstance(combination.get("StateId"), str):
                raise Exception("Importing a semantic space variable with an inconsistent combination (StateID)")
            if not isinstance(combination.get("ActionId"), str):
                raise Exception("Importing a semantic space variable with an inconsistent combination (ActionID)")
            if not callable(combination.get("Method")):
                raise Exception("Importing a semantic space variable with an inconsistent combination (Method)")
            if inspect.signature(combination.get("Method")) != inspect.signature(expected_method):
                raise Exception("Importing a semantic space variable with an inconsistent Method signature. "
                                "Expected: {}".format(inspect.signature(expected_method)))

        if self._default is not None:
            self.set_value(self._default)

    def set_value(self, value) -> None:
        try:
            computed_value = self._domain(value)
        except ValueError:
            raise Exception("Assigning a value to a SS Variable not coherent with its domain: {}".format(value))
        if computed_value < self._range[0] or computed_value > self._range[1]:
            raise Exception(
                "Assigning an out of range value to a SS Variable {}. Value {} not in range {}"
                .format(self._name, value, self._range))

        self._value = computed_value

    def is_initialized(self) -> bool:
        return self._value is not None

    def compute_probability(self, state_id, action_id, old_probability: np.ndarray) -> np.ndarray:
        assert isinstance(old_probability, np.ndarray)

        method_to_exec = None

        # Make sure that the variable has a value
        if self._value is None:
            raise Exception("SS Variable {} hasn't been initialized yet".format(self._name))

        # Look for the right method to exec
        for combination in self._combinations:
            is_same_state_id = combination.get("StateId") == state_id
            is_same_action_id = combination.get("ActionId") == action_id
            if is_same_action_id and is_same_state_id:
                method_to_exec = combination.get("Method")
                break

        if method_to_exec is None:
            raise Exception("SS Variable {} doesn't have this combination {}-{}".format(
                self._name, state_id, action_id
            ))

        # A method has been found
        try:
            new_probability: np.ndarray = method_to_exec(old_probability=old_probability.copy(),
                                                         variable_value=self._value)
            new_probability = new_probability.astype(np.half)
        except TypeError:
            raise Exception(f"The method relative to {self._name} (ss variable) failed.")

        # Make sure that the returned value is a np.ndarray, and it has the same shape of the old probability
        if not isinstance(new_probability, np.ndarray):
            raise Exception(f"The semantic space method relative to {self._name} didn't return an np.ndarray")

        if new_probability.shape != old_probability.shape:
            raise Exception(f"The semantic space method relative to {self._name} didn't return a proper array")

        try:
            new_probability = check_and_normalize_probabilities(new_probability)
        except Exception:
            e_str = f"The ss variable {self._name} returned an inconsistent probability for {state_id} - {action_id}"
            raise Exception(e_str)

        return new_probability

    def get_name(self) -> str:
        return self._name

    def get_value(self):
        return self._value

    def get_combinations(self) -> list:
        all_combinations = []
        for combination in self._combinations:
            all_combinations.append({"StateId": combination.get("StateId"), "ActionId": combination.get("ActionId")})

        return all_combinations
