import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mdp_simulator.mdp.action import Action
from mdp_simulator.mdp.single_state import SingleState
from mdp_simulator.utils.context import Context, Topics
import mdp_simulator.config as config

from datetime import datetime
import os

COLUMN_NAMES = ["state", "action", "outcome", "probability", "hdi low", "hdi high", "full mean", "partial mean",
                "is hdi exceeded", "eq constraint low", "eq constraint high"]
VALUES = []


def new_prior_logger(current_state: SingleState, selected_action_key: str, selected_outcome: str,
                     is_hdi_exceeded: np.ndarray):
    global VALUES
    Context.emit(Topics.DEBUG, "Logging a new STEP")
    state_id = current_state.get_id()
    action: Action = current_state.get_action(selected_action_key)
    for outcome, probability, hdi, full_mean, partial_mean, hdi_exceeded, eq_cons in zip(action.get_outcomes(),
                                                                                         action.get_probabilities(),
                                                                                         action.get_hdi(),
                                                                                         action.get_expected(),
                                                                                         action.get_partial_expected(),
                                                                                         is_hdi_exceeded,
                                                                                         action.get_eq_constraints()):
        VALUES.append([state_id, selected_action_key, outcome, probability, hdi[0], hdi[1], full_mean, partial_mean,
                       hdi_exceeded, eq_cons[0], eq_cons[1]])


def execution_ended(*args):
    Context.emit(Topics.DEBUG, "Execution ended")
    save_to_file()


# Subscribe or Unsubscribe the listeners
OUTPUT_NAME: str = ""

OUTPUT_BASE_FOLDER = "OUTPUT"


def save_to_file():
    global VALUES, COLUMN_NAMES, OUTPUT_NAME, OUTPUT_BASE_FOLDER
    # Create the content of the file
    output_content = pd.DataFrame(VALUES, columns=COLUMN_NAMES)

    # Check for the output folder existence
    if not os.path.exists(OUTPUT_BASE_FOLDER):
        os.mkdir(OUTPUT_BASE_FOLDER)

    output_folder = os.path.join(OUTPUT_BASE_FOLDER, OUTPUT_NAME)
    if not os.path.exists(output_folder):
        try:
            os.mkdir(output_folder)
        except FileExistsError as e:
            pass

    # Set the output filename
    now = datetime.now()
    output_filename = f"{now.strftime('%Y-%m-%d_%H%M%S')}_{os.getpid()}.csv"
    output_path = os.path.join(output_folder, output_filename)

    # If it already exists just exit. Otherwise, save it
    if os.path.exists(output_path):
        raise Exception(f"{output_path} already exists")
    output_content.to_csv(output_path, index=False)
    plot_csv(output_path)


def plot_csv(file_path: str):
    # Read from file the csv
    df = pd.read_csv(file_path)

    # Prepare the output folder
    output_folder = file_path[:-4]
    if os.path.exists(output_folder):
        raise Exception(f"{output_folder} already exists")
    os.mkdir(output_folder)

    # Get from the csv all the combination of state - action - outcome
    state_ids = np.unique(df["state"].to_numpy())
    combinations = []
    for state_id in state_ids:
        action_keys = np.unique(df[df['state'] == state_id]["action"].to_numpy())
        for action_key in action_keys:
            outcomes = np.unique(df.loc[((df['state'] == state_id) & (df['action'] == action_key), ["outcome"])])
            for outcome in outcomes:
                combinations.append([state_id, action_key, outcome])

    # For each combination plot a graph
    for combination in combinations:
        filtered_data = df.loc[df['state'] == combination[0]].loc[df["action"] == combination[1]].loc[
            df['outcome'] == combination[2]]

        prob = filtered_data["probability"].to_numpy()
        full_mean = filtered_data["full mean"].to_numpy()
        partial_mean = filtered_data["partial mean"].to_numpy()
        hdi_low = filtered_data["hdi low"].to_numpy()
        hdi_high = filtered_data["hdi high"].to_numpy()
        eq_cons_low = filtered_data["eq constraint low"].to_numpy()
        eq_cons_high = filtered_data["eq constraint high"].to_numpy()
        is_hdi_exceeded = filtered_data["is hdi exceeded"].to_numpy()

        # Number of samples
        x = np.arange(prob.size)
        hdi_exceeded = np.where(is_hdi_exceeded)[0]
        hdi_not_exceeded = np.where(~is_hdi_exceeded)[0]

        # Split the indexes into groups of consecutive elements
        groups = np.split(hdi_exceeded, np.where(np.diff(hdi_exceeded) != 1)[0] + 1)
        # Get a list of the indexes of the groups of config.MAX_PARTIAL_POSTERIOR_RETRIES consecutive true values
        prior_reset_x = [group[-1] for group in groups if len(group) == config.MAX_PARTIAL_POSTERIOR_RETRIES]

        # Set the DPI for the image
        plt.figure(dpi=config.PLOT_DPI, figsize=(10, 5))

        # Background elements
        plt.fill_between(x, hdi_low, hdi_high, color="b", alpha=.1, label="HDI")
        plt.fill_between(x, eq_cons_low, eq_cons_high, color="g", alpha=.1, label="EQ Constraint")

        # Plot the full probability
        plt.plot(x, prob, 'r--', label="Probability")

        # Plot the full mean
        plt.plot(x,
                 full_mean,
                 '-',
                 color='b',
                 linewidth=1.5,
                 label="Full Mean")

        # Plot prior reset occurrences
        plt.vlines(prior_reset_x, 0.0, 1.0, color='c',
                   alpha=.1, label="Prior Reset Occurred")
        # Plot the partial mean completely and then partially
        plt.plot(x,
                 partial_mean,
                 '-',
                 color='black',
                 linewidth=1,
                 alpha=.4)

        # create masks for the two sections
        mask1 = np.zeros_like(partial_mean, dtype=bool)
        mask1[hdi_exceeded] = True
        mask2 = np.zeros_like(partial_mean, dtype=bool)
        mask2[hdi_not_exceeded] = True

        # # plot the two sections separately and mask the other section
        plt.plot(np.ma.masked_where(mask1, x),
                 np.ma.masked_where(mask1, partial_mean),
                 '-',
                 color='green',
                 linewidth=1,
                 label="Partial Mean (HDI Exceeded)")

        plt.plot(np.ma.masked_where(mask2, x),
                 np.ma.masked_where(mask2, partial_mean),
                 '-',
                 color='purple',
                 linewidth=1.5,
                 label="Partial Mean (HDI Not Exceeded)")

        plt.legend()

        plt.title(f'State {combination[0]} - Action {combination[1]} - Outcome {combination[2]}')
        plt.ylabel('Probability')
        plt.xlabel('Samples')
        plt.xlim(0, prob.size - 1)
        plt.ylim(0, 1.0)
        plt.savefig(f"{output_folder}/{combination[0]}-{combination[1]}-{combination[2]}.jpg")
        # plt.show()
        plt.close()


def subscribe(output_name):
    global OUTPUT_NAME, VALUES
    OUTPUT_NAME = output_name
    VALUES = []
    Context.subscribe_single(Topics.UPDATED_PRIOR_ACTION_SELECTED, new_prior_logger)
    Context.subscribe_single(Topics.END_CONDITION_SATISFIED, execution_ended)


def unsubscribe():
    Context.unsubscribe_single(Topics.UPDATED_PRIOR_ACTION_SELECTED, new_prior_logger)
    Context.unsubscribe_single(Topics.END_CONDITION_SATISFIED, execution_ended)
