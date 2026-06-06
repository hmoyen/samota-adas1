import argparse
import mdp_simulator.utils.enums as enums

# Default configuration

FOLDER_NAME = "Examples/StartingExample"

# SS_FOLDER_NAME = "Examples/StartingExample/SS_VARIABLES"

MAX_STEPS = 10000

HDI_SAMPLES = 10000
HDI_PROB = 0.8

BATCH_SIZE = 10

TERMINATION_CONDITION = enums.Termination.STEPS

EQUILIBRIUM_POLICY = enums.EquilibriumPolicy.RANDOM

DEBUG_LEVEL = enums.LogTypes.INFO

INITIAL_POLICY = []

OUTPUT_FOLDER_NAME = None

RANDOM_MDP = None  # (4, 4, 4)

MAX_HISTORY_PARTIAL_POSTERIOR = 75

MAX_PARTIAL_POSTERIOR_RETRIES = 5

# Constraint check for each state
CONSTRAINT_SAT_MAX_HISTORY = 100
CONSTRAINT_SAT_PERC = 10

PLOT_DPI = 200


def load_arguments():
    # Load arguments

    all_args = argparse.ArgumentParser()

    all_args.add_argument("-ep", "--Equilibrium-Policy", required=False, help="Set the Equilibrium Policy")
    all_args.add_argument("-n", "--N", required=False, help="Steps")
    all_args.add_argument("-b", "--Batch-Size", required=False, help="Batch size interval for HDI computation")
    all_args.add_argument("-t", "--Termination", required=False, help="Termination condition")
    all_args.add_argument("-i", "--Mpd", required=False, help="MDP input folder")
    all_args.add_argument("-d", "--Debug", required=False, help="Debug level")
    all_args.add_argument("-ip", "--Initial-Policy", required=False, help="Initial Policy")
    all_args.add_argument("-o", "--Output-Csv", required=False, help="Save the log to a csv")
    all_args.add_argument("-r", "--Random-Mdp", required=False, help="Random mdp with 'n_state, n_action, n_prior'")

    args, _ = all_args.parse_known_args()
    args = vars(args)

    # Update default values

    if args.get("Equilibrium_Policy") is not None:
        global EQUILIBRIUM_POLICY
        try:
            EQUILIBRIUM_POLICY = enums.EquilibriumPolicy[args.get("Equilibrium_Policy")]
        except KeyError:
            raise Exception(f"Expecting an Equilibrium Policy enum not {args.get('Equilibrium_Policy')}")

    if args.get("Random_Mdp") is not None:
        global RANDOM_MDP
        line = args.get("Random_Mdp")
        line = line.split(",")
        if len(line) != 3:
            raise Exception("Expecting 3 parameters divided by a comma")
        RANDOM_MDP = (int(line[0]), int(line[1]), int(line[2]))

    if args.get("Termination") is not None:
        global TERMINATION_CONDITION
        try:
            TERMINATION_CONDITION = enums.Termination[args.get("Termination")]
        except KeyError:
            raise Exception(f"Expecting a Termination enum not {args.get('Termination')}")

    if args.get("Debug") is not None:
        global DEBUG_LEVEL
        try:
            DEBUG_LEVEL = enums.LogTypes[args.get("Debug")]
        except KeyError:
            raise Exception(f"Expecting a LogType enum not {args.get('Debug')}")

    if args.get("Mpd") is not None:
        global FOLDER_NAME
        FOLDER_NAME = args.get("Mpd")

    if args.get("N") is not None:
        global MAX_STEPS
        MAX_STEPS = int(args.get("N"))

    if args.get("Batch_Size") is not None:
        global BATCH_SIZE
        BATCH_SIZE = int(args.get("Batch_Size"))

    if args.get("Output_Csv") is not None:
        global OUTPUT_FOLDER_NAME
        OUTPUT_FOLDER_NAME = args.get("Output_Csv")

    if args.get("Initial_Policy") is not None:
        global INITIAL_POLICY
        initial_policy = args.get("Initial_Policy")

        if not (initial_policy.startswith("(") and initial_policy.endswith(")")):
            raise Exception(f"Initial policy inconsistent: {initial_policy}")

        # Remove initial and final parenthesis
        initial_policy = initial_policy[1:len(initial_policy) - 1]

        # Split on ","
        initial_policy = initial_policy.split(",")

        # Check that each element is an int
        # for index in range(len(initial_policy)):
        #     initial_policy[index] = int(initial_policy[index])

        INITIAL_POLICY = initial_policy


def check_required_arguments():
    if FOLDER_NAME is None and RANDOM_MDP is None:
        raise Exception("At least one between a MDP configuration and a random mdp has to be specified.")

    if FOLDER_NAME is not None and RANDOM_MDP is not None:
        raise Exception("It can't be specified at the same time a configuration folder and a random mdp.")
