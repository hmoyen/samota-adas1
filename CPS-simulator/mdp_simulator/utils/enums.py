from enum import IntEnum


class EquilibriumPolicy(IntEnum):
    RANDOM = 1
    POLICY_ITERATION = 2


class Termination(IntEnum):
    STEPS = 1


class ReadState(IntEnum):
    ID = 1
    NAME = 2
    STATE_TYPE = 3
    ACTIONS = 4
    PRIOR_STATE = 5
    PRIOR_ACTION = 6
    PRIOR_VALUES = 7
    PRIOR_EQUILIBRIUM_CONSTRAINTS = 8


class LogTypes(IntEnum):
    NONE = 0
    ERROR = 1
    STAT = 2
    INFO = 3
    DEBUG = 4


class StateType(IntEnum):
    OBSERVABLE = 0
    CONTROLLABLE = 1


class LogLocations(IntEnum):
    LOCAL = 0
    REMOTE = 1
    BOTH = 2


class Topics(IntEnum):
    SET_LOG_TYPE = 0
    ERROR = 1
    STAT = 2
    INFO = 3
    DEBUG = 4
    START_NEW_STEP = 5
    END_STEP = 6
    TERMINAL_STATE = 7
    SELECT_NEXT_ACTION = 8
    SELECTED_NEXT_ACTION = 9
    SELECT_NEXT_STATE = 10
    SELECTED_NEXT_STATE = 11
    PRIOR_ACTION_SELECTED = 12
    UPDATED_PRIOR_ACTION_SELECTED = 13
    EQ_CONSTRAINT_UNSATISFIED = 14
    POLICY_UPDATED = 15
    END_CONDITION_SATISFIED = 16
