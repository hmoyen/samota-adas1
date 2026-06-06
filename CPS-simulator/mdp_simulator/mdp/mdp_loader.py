import os
import os.path
import numpy as np

from mdp_simulator.utils.enums import ReadState, Topics
from mdp_simulator.utils.context import Context
from mdp_simulator.mdp.single_state import SingleState


class MDPLoader:
    _states = None
    _changes = None
    _ss_variables = None
    _ss_changes = None

    # Constructor
    def __init__(self, config_path) -> None:
        self._states = dict()
        self._changes = dict()
        self._ss_changes = dict()
        self._ss_variables = dict()

        self._config_path = config_path

        states_loaded = self.__parse_states()

        if not states_loaded:
            raise Exception("A states file has to be always supplied.")
        else:
            self.__parse_priors()
            self.__parse_changes()
            self.__parse_ss_changes()
            self.__parse_semantic_space_variables()

    # Methods
    def __parse_states(self) -> bool:
        states_file_path = os.path.join(self._config_path, "states.mdp")
        file_content = self.__read_file(states_file_path)

        if file_content is None:
            return False

        file_content = self.__to_single_string(file_content)
        file_content = file_content.split(";")

        read_state = ReadState.ID
        self._states = dict({})
        total_lines = len(file_content)

        temp_state: SingleState = SingleState()
        for index, line in enumerate(file_content):
            # Check EOF
            if index == total_lines - 1:
                if read_state != ReadState.ACTIONS:
                    raise Exception(
                        "The MDP file seems inconsistent. Last state is not well defined")
                # Push Last state
                temp_id = temp_state.get_id()
                if self._states.get(temp_id) is not None:
                    raise Exception(
                        f"Multiple initialization of stateId: {temp_id}")
                self._states.update({temp_id: temp_state})
                Context.emit(Topics.DEBUG, "EOF Detected")
                continue

            # Not EOF
            line_has_comma = line.__contains__(",")

            # Check for inconsistencies
            if line_has_comma and read_state == ReadState.ID:
                raise Exception(
                    f"The MDP file seems inconsistent. Read '{line}' while expecting an ID")

            # Check for inconsistencies
            if line_has_comma and read_state == ReadState.NAME:
                raise Exception(
                    f"The MDP file seems inconsistent. Read '{line}' while expecting a NAME")

            if line_has_comma and read_state == ReadState.STATE_TYPE:
                raise Exception(
                    f"The MDP file seems inconsistent. Read '{line}' while expecting a STATE TYPE")

            # Check for new state
            if (not line_has_comma) and read_state == ReadState.ACTIONS:
                # Push previous state
                temp_id = temp_state.get_id()
                if self._states.get(temp_id) is not None:
                    raise Exception(
                        f"Multiple initialization of stateId: {temp_id}")
                self._states.update({temp_id: temp_state})
                # Reset read_state
                read_state = ReadState.ID

            # Import actual parser
            if read_state == ReadState.ID:
                temp_state = SingleState()
                temp_state.set_id(line)
                read_state = ReadState.NAME
                Context.emit(Topics.DEBUG, "ID", line)
            elif read_state == ReadState.NAME:
                temp_state.set_name(line)
                read_state = ReadState.STATE_TYPE
                Context.emit(Topics.DEBUG, "NAME", line)
            elif read_state == ReadState.STATE_TYPE:
                temp_state.set_state_type(line)
                read_state = ReadState.ACTIONS
                Context.emit(Topics.DEBUG, "STATE TYPE", line)
            elif read_state == ReadState.ACTIONS:
                divided_line = line.split(",")

                # Check line length
                if len(divided_line) != 3:
                    raise Exception(f"Inconsistent Action: {line}")

                action_name = divided_line[0]
                probability = divided_line[1]
                next_state_id = divided_line[2]

                # Check probability conversion
                try:
                    probability = float(probability)
                except ValueError:
                    raise Exception(
                        f"Inconsistent Action: {line}. Expecting a float probability, received {probability}")

                # Input ok
                temp_state.add_action(
                    action_name, probability, next_state_id)
                Context.emit(Topics.DEBUG, "ACTION", line)

        return True

    def __parse_priors(self) -> bool:
        priors_file_path = os.path.join(self._config_path, "priors.mdp")
        file_content = self.__read_file(priors_file_path)

        if file_content is None:
            return False

        file_content = self.__to_single_string(file_content)
        if file_content.__contains__("PRIORS"):
            file_content = file_content.split("PRIORS")[1]
        file_content = file_content.split(";")

        total_lines = len(file_content)
        read_state = ReadState.PRIOR_STATE
        temp_id = None
        temp_action = None

        selected_state = SingleState()
        temp_prior = None
        for index, line in enumerate(file_content):
            # Check EOF
            if index == total_lines - 1:
                if read_state != ReadState.PRIOR_STATE:
                    raise Exception(
                        "The MDP file seems inconsistent (Priors section).")
                Context.emit(Topics.DEBUG, "EOF Detected")
                continue

            # Not EOF
            line_has_comma = line.__contains__(",")

            # Check for inconsistencies
            if line_has_comma and read_state == ReadState.PRIOR_STATE:
                raise Exception(
                    f"The MDP file seems inconsistent. Read '{line}' while expecting an ID (Prior section)")

            # Check for inconsistencies
            if line_has_comma and read_state == ReadState.PRIOR_ACTION:
                raise Exception(
                    f"The MDP file seems inconsistent. Read '{line}' while expecting an ACTION NAME (Prior section)")

            # Import actual parser
            if read_state == ReadState.PRIOR_STATE:
                temp_id = line
                read_state = ReadState.PRIOR_ACTION

                # Check State ID existence
                if not self._states.get(temp_id):
                    raise Exception(
                        f"Adding a Prior to a non existing State: {temp_id}")

                Context.emit(Topics.DEBUG, "Prior ID", line)
            elif read_state == ReadState.PRIOR_ACTION:
                temp_action = line
                read_state = ReadState.PRIOR_VALUES

                # Check Action ID existence
                selected_state: SingleState = self._states.get(temp_id)
                if not selected_state.get_action(temp_action):
                    raise Exception(
                        f"Adding a Prior to a non existing Action: {temp_id} -> {temp_action}")

                Context.emit(Topics.DEBUG, "Prior ACTION NAME", line)
            elif read_state == ReadState.PRIOR_VALUES:
                divided_line = np.array(line.split(","))

                # Check line length
                if divided_line.size < 2:
                    raise Exception(f"Inconsistent Prior Values: {line}")

                # Cast to integer of all values
                try:
                    divided_line = divided_line.astype(int)
                except ValueError:
                    raise Exception(
                        f"Inconsistent Prior values: {divided_line}. Expecting only integers.")

                # Input ok
                temp_prior = divided_line
                # selected_state.add_prior(temp_action, divided_line)
                read_state = ReadState.PRIOR_EQUILIBRIUM_CONSTRAINTS
                Context.emit(Topics.DEBUG, "PRIOR VALUES", line)
            elif read_state == ReadState.PRIOR_EQUILIBRIUM_CONSTRAINTS:
                line = line.split("-")

                divided_line = np.empty(0)
                for i in range(len(line)):
                    divided_line = np.append(divided_line, np.array(line[i].split(",")))

                # Check line length
                if divided_line.size < 2:
                    raise Exception(f"Inconsistent Equilibrium constraint Values: {line}")

                # Reshape and Cast to float of all values
                try:
                    divided_line = divided_line.astype(float).reshape(-1, 2)
                except ValueError:
                    raise Exception(
                        f"Inconsistent Equilibrium constraint values: {divided_line}. Expecting only floats.")

                # Input ok
                selected_state.add_prior(temp_action, temp_prior, divided_line)
                read_state = ReadState.PRIOR_STATE
                Context.emit(Topics.DEBUG, "PRIOR VALUES", line)

        return True

    def __parse_changes(self) -> bool:
        changes_file_path = os.path.join(self._config_path, "changes.mdp")
        file_content = self.__read_file(changes_file_path)

        if file_content is None:
            return False

        file_content = self.__to_single_string(file_content)

        changes_section = file_content.split("CHANGES")
        changes_section.pop(0)
        temp_changes_section = dict()
        for elem in changes_section:
            change_index = int(elem.split(":")[0])
            changes = elem.split(":")[1].split(";")
            changes_list = []
            for index in range(0, len(changes) - 1, 3):
                probabilities = np.array(changes[index + 2].split(","))
                probabilities = probabilities.astype(float)
                if probabilities.sum() != 1.0:
                    raise Exception(f"Expecting a list of probabilities with sum equal to 1: {probabilities}")
                changes_list.append(
                    {"state": changes[index], "action_key": changes[index + 1], "probabilities": probabilities})

            temp_changes_section.update({change_index: changes_list})

        self._changes = temp_changes_section

        # Check Changes Validity
        for change_interval in self._changes.items():
            for single_change in change_interval[1]:
                if self._states.get(single_change.get("state")) is None:
                    raise Exception(f"Inconsistent change interval: non-existent state {single_change[0]}")
                if self._states.get(single_change.get("state")).get_action(single_change.get("action_key")) is None:
                    raise Exception("Inconsistent change interval: {} - {}".format(single_change.get('state'),
                                                                                   single_change.get('action_key')))

        return True

    def __parse_ss_changes(self) -> bool:
        ss_changes_file_path = os.path.join(self._config_path, "ss_changes.mdp")
        file_content = self.__read_file(ss_changes_file_path)

        if file_content is None:
            return False

        file_content = self.__to_single_string(file_content)

        ss_changes_section = file_content.split("CHANGES")
        ss_changes_section.pop(0)
        temp_ss_changes_section = dict()
        for elem in ss_changes_section:
            change_index = int(elem.split(":")[0])
            changes = elem.split(":")[1].split(";")
            changes_list = []
            for line, _ in zip(changes, range(len(changes) - 1)):
                changes_list.append({"name": line.split(",")[0], "value": line.split(",")[1]})

            temp_ss_changes_section.update({change_index: changes_list})

        self._ss_changes = temp_ss_changes_section

        # The validity of all the changes is checked at runtime:
        #  if a value is wrong the program will stop stating the error
        return True

    def __parse_semantic_space_variables(self) -> bool:
        ss_variables_file_path = os.path.join(self._config_path, "ss_variables.mdp")
        file_content = self.__read_file(ss_variables_file_path)

        if file_content is None:
            return False

        file_content = self.__to_single_string(file_content)
        if file_content.__contains__("SEMANTIC_SPACE_VARIABLES"):
            file_content = file_content.split("SEMANTIC_SPACE_VARIABLES")[1]
        file_content = file_content.split(";")
        total_lines = len(file_content) - 1

        self._ss_variables = dict()

        for variable, _ in zip(file_content, range(total_lines)):
            variable_name = variable.split(":")[0]
            variable_value = variable.split(":")[1]

            self._ss_variables.update({variable_name: variable_value})

        return True

    def get_mdp_configs(self) -> (list, dict, dict):
        return self._states, self._ss_variables, self._changes, self._ss_changes

    @staticmethod
    def __read_file(filename: str):
        # Check if the file exists
        if not os.path.isfile(filename):
            return None

        # Open the file as f.
        # The function read_lines() reads the file.
        with open(filename) as f:
            content = f.read().splitlines()

        return content

    @staticmethod
    def __to_single_string(file_content):
        output = ""
        for line in file_content:
            output = f"{output}{line}"

        # Remove useless whitespaces
        output = output.replace("; ", ";")
        output = output.replace(", ", ",")
        output = output.replace(": ", ":")

        return output
