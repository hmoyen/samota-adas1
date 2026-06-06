import mdp_simulator.config as config

# ./mdp
from mdp_simulator.mdp.mdp import MDP
from mdp_simulator.mdp.semantic_space import SemanticSpace
from mdp_simulator.mdp.mdp_loader import MDPLoader
from mdp_simulator.mdp.semantic_space_variable import SemanticSpaceVariable
from mdp_simulator.mdp.single_state import SingleState
from mdp_simulator.mdp.action import Action

# ./utils
import mdp_simulator.utils.enums as enums
import mdp_simulator.utils.helpers as helpers
import mdp_simulator.utils.logger as logger
import mdp_simulator.utils.probabilities as probabilities
from mdp_simulator.utils.pub_sub import PubSub
from mdp_simulator.utils.context import Context

import mdp_simulator.subscribers.csv_log_and_plot as _csv_logger
import mdp_simulator.subscribers.listener as _listener
import mdp_simulator.subscribers.monitor as _monitor

_states: dict
_changes: dict
_importedMDP: MDP
_ss_variables_starting_value = None
_ss_changes = None
_ss_variables = None


def __load(override_states: dict = None,
           override_changes: dict = None,
           override_ss_variables=None,
           override_ss_variables_starting_value: dict = None,
           override_ss_changes: dict = None) -> None:
    global _states, _changes, _ss_variables, _ss_variables_starting_value, _ss_changes
    global _importedMDP

    # Load cli arguments
    config.load_arguments()
    config.check_required_arguments()

    # Subscribe all the event listeners
    _listener.subscribe()
    _monitor.subscribe()
    logger.subscribe()

    # Set logger to DEBUG
    Context.emit(enums.Topics.SET_LOG_TYPE, config.DEBUG_LEVEL)

    # csv_logger.subscribe()
    if config.OUTPUT_FOLDER_NAME is not None:
        Context.emit(enums.Topics.INFO, "The log of the execution will be saved inside: {}".format(
            config.OUTPUT_FOLDER_NAME))
        _csv_logger.subscribe(config.OUTPUT_FOLDER_NAME)

    if config.RANDOM_MDP is not None:
        # Generate a random MDP
        Context.emit(enums.Topics.INFO, "Creating a random MDP config with: {}".format(config.RANDOM_MDP))
        _states = helpers.build_random_mdp(*config.RANDOM_MDP)
        _changes = None
        _ss_variables_starting_value = None
        _ss_changes = None
        _ss_variables = None

    if config.FOLDER_NAME is not None:
        # Import MDP from file
        Context.emit(enums.Topics.INFO, "Importing MDP configs from folder: {}".format(config.FOLDER_NAME))
        mdp_configs = MDPLoader(config.FOLDER_NAME)
        _states, _ss_variables_starting_value, _changes, _ss_changes = mdp_configs.get_mdp_configs()
        _ss_variables = helpers.semantic_space_variables_importer(config.FOLDER_NAME)
        Context.emit(enums.Topics.INFO, "Successfully imported the MDP configs:")
        Context.emit(enums.Topics.INFO, "[{}]\tSTATES".format(_states is not None))
        Context.emit(enums.Topics.INFO, "[{}]\tCHANGES".format(len(_changes.keys()) > 0))
        Context.emit(enums.Topics.INFO, "[{}]\tSS_VARIABLES".format(len(_ss_variables) > 0))
        Context.emit(enums.Topics.INFO, "[{}]\tSS_CHANGES".format(len(_ss_changes.keys()) > 0))

    # Load the MDP
    Context.emit(enums.Topics.INFO, "Supplying the MDP config to the simulation.")

    if override_states is not None:
        pass
    if override_changes is not None:
        _changes = override_changes
    if override_ss_variables is not None:
        pass
    if override_ss_variables_starting_value is not None:
        _ss_variables_starting_value = override_ss_variables_starting_value
    if override_ss_changes is not None:
        pass

    # Prepare the Semantic Space (can be also empty)
    semantic_space = SemanticSpace(_ss_variables, _ss_variables_starting_value, _states)

    _importedMDP = MDP(_states, config.TERMINATION_CONDITION, config.MAX_STEPS, config.EQUILIBRIUM_POLICY,
                       config.INITIAL_POLICY, _changes, semantic_space, _ss_changes)


def __start() -> MDP:
    global _states
    global _importedMDP

    # Execute
    _importedMDP.run()
    Context.emit(enums.Topics.INFO, "Exiting...")

    return _importedMDP


def __stop() -> None:
    # Actions to perform on Stop
    Context.unsubscribe_all()
    pass


def __kill() -> None:
    # Actions to perform on Kill
    pass


def run(override_states: dict = None,
        override_changes: dict = None,
        override_ss_variables=None,
        override_ss_variables_starting_value: dict = None,
        override_ss_changes: dict = None) -> MDP:
    from datetime import datetime
    import traceback

    # Used only for execution time
    start_time = datetime.now()

    # In case of graceful exit the completed mdp will contain the run mdp instance
    completed_mdp = None

    try:
        # Load routine,
        __load(override_states=override_states,
               override_changes=override_changes,
               override_ss_variables=override_ss_variables,
               override_ss_variables_starting_value=override_ss_variables_starting_value,
               override_ss_changes=override_ss_changes)
        # Start routine
        completed_mdp = __start()
        # Stop routine
        __stop()
    except (SyntaxError, ImportError, AttributeError, KeyError, NameError,
            IndexError, TypeError) as err:
        Context.emit(enums.Topics.ERROR, err)
        # Kill routine
        traceback.print_exception(err, err, err.__traceback__)
        __kill()

    except KeyboardInterrupt:
        Context.emit(enums.Topics.ERROR, "Execution killed")
        # Kill routine
        __kill()

    end_time = datetime.now()

    print(f"Execution duration: {(end_time - start_time).total_seconds()}")

    return completed_mdp
