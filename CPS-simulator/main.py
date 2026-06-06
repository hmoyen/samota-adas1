from mdp_simulator import *

if __name__ == "__main__":
    config.MAX_STEPS = 5000
    config.FOLDER_NAME = "Examples/RescueRobotBig"
    mdp = run()
    s:SingleState = mdp.get_states_dictionary()["S0"]
    a:Action = s.get_action("a")
    print("{} {} {} {}".format(s.get_id(), "a", a.get_hdi(), a.get_eq_constraints()))
