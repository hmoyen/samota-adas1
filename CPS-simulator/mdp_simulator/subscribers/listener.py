from mdp_simulator.utils.context import Context, Topics


def new_step_handler(state):
    Context.emit(Topics.DEBUG, "Handling a new STEP")


def end_step_handler():
    Context.emit(Topics.DEBUG, "Handling the end of the STEP")


# Subscribe or Unsubscribe the listeners


def subscribe():
    Context.subscribe_single(Topics.START_NEW_STEP, new_step_handler)
    Context.subscribe_single(Topics.END_STEP, end_step_handler)


def unsubscribe():
    Context.unsubscribe_single(Topics.START_NEW_STEP, new_step_handler)
    Context.unsubscribe_single(Topics.END_STEP, end_step_handler)
