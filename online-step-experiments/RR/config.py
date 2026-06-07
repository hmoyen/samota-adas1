from utils.constraints_builder import compute_constraints

SS_VARIABLES = {
    "power": {"domain": int, "range": [0, 100]},
    "cruise_speed": {"domain": float, "range": [0, 5]},
    "bandwidth": {"domain": float, "range": [10, 50]},
    "quality": {"domain": int, "range": [0, 2]},
    "illuminance": {"domain": float, "range": [40, 120000]},
    "smoke_intensity": {"domain": int, "range": [0, 2]},
    "obstacle_size": {"domain": float, "range": [0, 120]},
    "obstacle_distance": {"domain": float, "range": [0, 10]},
    "firm_obstacle": {"domain": int, "range": [0, 1]},
}

RQ = "rq1"
EXTRA_NAME = "NoHis"
PLOT = False

BATCH_SIZE = 5
HISTORY_RETRIES = 10
HISTORY_LEN = 1000000
MDP_FOLDER = "./INPUT/RescueRobot_v3"

MAX_STEPS = 100000

MAX_SAMPLES = 10000

# Constraints definition

IDEAL_SPOTS = {
    "S0": {
        "a": [0.05, 0.85, 0.10]
    },
    "S10": {
        "l": [0.90, .10],
    }
}

#constraints = compute_constraints([.025, .03, .04, .045], IDEAL_SPOTS)
constraints = compute_constraints([.12, .14, .16, .18], IDEAL_SPOTS)

CONSTRAINTS = [
    # SINGLE CONSTRAINTS
    {
        "S0": {
            "a": constraints["S0"]["a"][0]
        }
    },
    {
        "S10": {
            "l": constraints["S10"]["l"][0]
        }
    },
    {
        "S0": {
            "a": constraints["S0"]["a"][0]
        },
        "S10": {
            "l": constraints["S10"]["l"][0]
        }
    },
    {
        "S0": {
            "a": constraints["S0"]["a"][1]
        },
        "S10": {
            "l": constraints["S10"]["l"][1]
        }
    },
    {
        "S0": {
            "a": constraints["S0"]["a"][2]
        },
        "S10": {
            "l": constraints["S10"]["l"][2]
        }
    },
    {
        "S0": {
            "a": constraints["S0"]["a"][3]
        },
        "S10": {
            "l": constraints["S10"]["l"][3]
        }
    }
]

MINIMAL_CONSTRAINTS = {
    "S0": {
        "a": constraints["S0"]["a"][0]
    },
    "S10": {
        "l": constraints["S10"]["l"][0]
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
