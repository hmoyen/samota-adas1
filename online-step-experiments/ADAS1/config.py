from utils.constraints_builder import compute_constraints

SS_VARIABLES = {
    "car_speed": {"domain": float, "range": [5.0, 50.0]},
    "p_x": {"domain": float, "range": [0.0, 10.0]},
    "p_y": {"domain": float, "range": [0.0, 10.0]},
    "orientation": {"domain": int, "range": [-30, 30]},
    "weather": {"domain": int, "range": [0, 2]},
    "road_shape": {"domain": int, "range": [0, 2]},
}

RQ = "rq1"
EXTRA_NAME = "NoHis"
PLOT = False

BATCH_SIZE = 5
HISTORY_RETRIES = 10
HISTORY_LEN = 1000000
MDP_FOLDER = "INPUT/AutonomousDriving_v1"

MAX_STEPS = 100000

MAX_SAMPLES = 10000

# Constraints definition

IDEAL_SPOTS = {
    "S0": {
        "a": [0.001, 0.999]
    },
    "S2": {
        "b": [0.98, 0.015, 0.005],
    }
}

#constraints = compute_constraints([.025, .03, .04, .045], IDEAL_SPOTS)
constraints = compute_constraints([.025, .03], IDEAL_SPOTS)

CONSTRAINTS = [
    # SINGLE CONSTRAINTS
    {
        "S0": {
            "a": constraints["S0"]["a"][0]
        }
    },
    {
        "S2": {
            "b": constraints["S2"]["b"][1]
        }
    },
    {
        "S0": {
            "a": constraints["S0"]["a"][1]
        },
        "S2": {
            "b": constraints["S2"]["b"][1]
        }
    }
]

MINIMAL_CONSTRAINTS = {
    "S0": {
        "a": constraints["S0"]["a"][0]
    },
    "S2": {
        "b": constraints["S2"]["b"][1]
    }
}

# _template = {
#     "S0": {
#         "a": constraints["S0"]["a"][0]
#     },
#     "S5": {
#         "g": constraints["S5"]["g"][0]
#     },
#     "S10": {
#         "l": constraints["S10"]["l"][0],
#         "m": constraints["S10"]["m"][0]
#     }
# }
