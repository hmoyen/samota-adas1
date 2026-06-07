import numpy as np
from math import sin, pi, isclose


def do_nothing(old_probability: np.ndarray, variable_value) -> np.ndarray:
    return old_probability


def compute_function(budget: float, assignment: list[float], bounds: list, interval: list[float]):
    budget += .02

    # Transform into numpy arrays
    assignment = np.array(assignment, float)
    bounds = np.array(bounds, float)
    interval = np.array(interval, float) * pi

    # Pre-check
    assert 0.0 <= budget <= 1.0
    assert isclose(assignment.sum(), 0, rel_tol=1e-9, abs_tol=1e-9)
    assert bounds[0] <= bounds[1]
    assert interval[0] <= interval[1]

    def computed_function(old_probability: np.ndarray, variable_value) -> np.ndarray:
        # Run-time check
        assert bounds[0] <= variable_value <= bounds[1]
        assert assignment.shape == old_probability.shape

        # Compute the sin parameter
        perc = (variable_value - bounds[0]) / (bounds[1] - bounds[0])
        x = (interval[1] - interval[0]) * perc + interval[0]

        # Compute and assign the budget
        budget_cup = (budget * sin(x)) / 2
        budget_assignment = budget_cup * assignment

        # Compute the new probability
        new_probability = old_probability + budget_assignment

        return new_probability

    return computed_function


# compute_function(budget, assignment, bounds, interval)


SemanticSpaceVariable = [
    {
        "Name": "power",
        "Type": "SYS",
        "Domain": int,
        "Range": [0, 100],
        "Default": 0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": compute_function(0.05, [-0.5, 1.0, -0.5], [0, 100], [-0.5, 0.5])
            },
            {
                "StateId": "S5",
                "ActionId": "g",
                "Method": do_nothing
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": compute_function(0.02, [1., -1.], [0, 100], [-0.5, 0.5])
            },
            {
                "StateId": "S10",
                "ActionId": "m",
                "Method": do_nothing
            },
        ],
    },
    {
        "Name": "cruise_speed",
        "Type": "SYS",
        "Domain": float,
        "Range": [0, 5],
        "Default": 0.0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": do_nothing
            },
            {
                "StateId": "S5",
                "ActionId": "g",
                "Method": compute_function(0.07, [-.5, -.5, 1.], [0, 5], [0.5, 1.5])
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": do_nothing
            },
            {
                "StateId": "S10",
                "ActionId": "m",
                "Method": do_nothing
            },
        ],
    },
    {
        "Name": "bandwidth",
        "Type": "SYS",
        "Domain": float,
        "Range": [10, 50],
        "Default": 10.0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": do_nothing
            },
            {
                "StateId": "S5",
                "ActionId": "g",
                "Method": do_nothing
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": compute_function(0.06, [1., -1.], [10, 50], [0., 0.5])
            },
            {
                "StateId": "S10",
                "ActionId": "m",
                "Method": compute_function(0.06, [-1., 1.], [10, 50], [0., 0.5])
            },
        ],
    },
    {
        "Name": "quality",
        "Type": "SYS",
        "Domain": int,
        "Range": [0, 2],
        "Default": 0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": compute_function(0.02, [-0.5, 1.0, -0.5], [0, 2], [0, 0.5])
            },
            {
                "StateId": "S5",
                "ActionId": "g",
                "Method": do_nothing
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": do_nothing
            },
            {
                "StateId": "S10",
                "ActionId": "m",
                "Method": compute_function(0.01, [-1., 1.], [0, 2], [0, 0.5])
            },
        ],
    },
    {
        "Name": "illuminance",
        "Type": "SYS",
        "Domain": float,
        "Range": [40, 120000],
        "Default": 40.0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": compute_function(0.07, [.5, -1., .5], [40, 120000], [0.5, 1])
            },
            {
                "StateId": "S5",
                "ActionId": "g",
                "Method": do_nothing
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": compute_function(0.01, [-1., 1.], [40, 120000], [0.5, 1])
            },
            {
                "StateId": "S10",
                "ActionId": "m",
                "Method": compute_function(0.01, [1., -1.], [40, 120000], [-0.5, 0])
            },
        ],
    },
    {
        "Name": "smoke_intensity",
        "Type": "SYS",
        "Domain": int,
        "Range": [0, 2],
        "Default": 0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": compute_function(0.01, [.5, -1., .5], [0, 2], [0.5, 1.])
            },
            {
                "StateId": "S5",
                "ActionId": "g",
                "Method": compute_function(0.05, [.5, .5, -1.0], [0, 2], [0, 0.5])
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": do_nothing
            },
            {
                "StateId": "S10",
                "ActionId": "m",
                "Method": compute_function(0.01, [1., -1.0], [0, 2], [-0.5, 0.])
            },
        ],
    },
    {
        "Name": "obstacle_size",
        "Type": "SYS",
        "Domain": float,
        "Range": [0, 120],
        "Default": 0.0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": do_nothing
            },
            {
                "StateId": "S5",
                "ActionId": "g",
                "Method": compute_function(0.02, [.5, .5, -1.0], [0, 120], [0, 0.5])
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": compute_function(0.01, [-1., 1.], [0, 120], [0, 0.5])
            },
            {
                "StateId": "S10",
                "ActionId": "m",
                "Method": do_nothing
            },
        ],
    },
    {
        "Name": "obstacle_distance",
        "Type": "SYS",
        "Domain": float,
        "Range": [0, 10],
        "Default": 0.0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": compute_function(0.01, [.5, -1., .5], [0, 10], [-0.5, 0.])
            },
            {
                "StateId": "S5",
                "ActionId": "g",
                "Method": do_nothing
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": compute_function(0.04, [-1., 1.], [0, 10], [0.5, 1])
            },
            {
                "StateId": "S10",
                "ActionId": "m",
                "Method": compute_function(0.02, [1., -1.], [0, 10], [0.5, 1])
            },
        ],
    },
    {
        "Name": "firm_obstacle",
        "Type": "SYS",
        "Domain": int,
        "Range": [0, 1],
        "Default": 0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": compute_function(0.01, [.5, -1.0, .5], [0, 1], [0, 0.5])
            },
            {
                "StateId": "S5",
                "ActionId": "g",
                "Method": compute_function(0.01, [.5, .5, -1.0], [0, 1], [0.5, 1.])
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": compute_function(0.01, [-1., 1.], [0, 1], [-0.5, 0.])
            },
            {
                "StateId": "S10",
                "ActionId": "m",
                "Method": compute_function(0.02, [1., -1.], [0, 1], [0, 0.5])
            },
        ],
    },
]
