import mdp_simulator.config as config
from mdp_simulator.mdp.single_state import SingleState
from mdp_simulator.utils.context import Context, Topics

pending: dict = dict()


def prior_action_handler(current_state: SingleState, selected_action_key, selected_outcome):
    global pending

    Context.emit(Topics.DEBUG,
                 "Handling PRIOR UPDATE for: {}, {}, {}"
                 .format(current_state.get_id(), selected_action_key, selected_outcome))

    # Increase the prior
    current_state.increment_prior(selected_action_key, selected_outcome)

    state_id = current_state.get_id()

    pending_state: dict = pending.get(state_id)

    # Case 1: state never updated
    if pending_state is None:
        pending.update({state_id: dict({selected_action_key: 0})})
        pending_state: dict = pending.get(state_id)

    # Case 2: state already added to the pending list
    pending_action: int = pending_state.get(selected_action_key)

    # 2.a: action never updated
    if pending_action is None:
        pending_state.update({selected_action_key: 0})
        pending_action: int = pending_state.get(selected_action_key)

    # 2.b: action already added -> increment
    pending_action += 1
    pending_state.update({selected_action_key: pending_action})

    # Check if batch size reached
    if pending_action >= config.BATCH_SIZE:
        Context.emit(Topics.DEBUG, f"Issuing PRIOR UPDATE for: {current_state.get_id()}, {selected_action_key}")
        # Compute the posterior
        current_state.compute_posterior(selected_action_key)

        # Remove the action from the pending list
        pending_state.pop(selected_action_key)
        if len(pending_state.keys()) == 0:
            pending.pop(state_id)


def update_all_pending(all_states: dict):
    global pending

    for stateId in pending.keys():
        for actionKey in pending.get(stateId).keys():
            state: SingleState = all_states.get(stateId)
            # Compute the posterior
            state.compute_posterior(actionKey)


# Subscribe or Unsubscribe the listeners


def subscribe():
    Context.subscribe_single(Topics.PRIOR_ACTION_SELECTED, prior_action_handler)
    Context.subscribe_single(Topics.END_CONDITION_SATISFIED, update_all_pending)


def unsubscribe():
    Context.unsubscribe_single(Topics.PRIOR_ACTION_SELECTED, prior_action_handler)
    Context.unsubscribe_single(Topics.END_CONDITION_SATISFIED, update_all_pending)
