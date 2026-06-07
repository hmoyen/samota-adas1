import numpy as np


def do_nothing(old_probability: np.ndarray, variable_value) -> np.ndarray:
    return old_probability


def cruise_speed_s0_a(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 3:
        max_increment = 0.03

        offset = (variable_value - 3.) / 2.
        offset **= 2

        old_probability[0] += max_increment * offset
        old_probability[1] -= 2 * max_increment * offset
        old_probability[2] += max_increment * offset

    return old_probability


def cruise_speed_s5_g(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 3.5:
        max_increment = 0.02

        offset = (variable_value - 3.5) / 1.5
        offset **= 2

        old_probability[0] += max_increment * offset
        old_probability[1] += max_increment * offset
        old_probability[2] -= 2 * max_increment * offset

    return old_probability


def cruise_speed_s10_l(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 4.5:
        max_increment = 0.02

        offset = (variable_value - 4.5) / 0.5
        offset **= 2

        old_probability[0] -= max_increment * offset
        old_probability[1] += max_increment * offset

    return old_probability


def cruise_speed_s10_m(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 3.5:
        max_increment = 0.04

        offset = (variable_value - 3.5) / 1.5
        offset **= 2

        old_probability[0] += max_increment * offset
        old_probability[1] -= max_increment * offset

    return old_probability


def quality_s0_a(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value == 0:
        increment_0 = 0.0
        increment_1 = 0.0
        increment_2 = 0.0
    elif variable_value == 1:
        increment_0 = 0.005
        increment_1 = -0.01
        increment_2 = 0.005
    else:
        increment_0 = 0.006
        increment_1 = -0.02
        increment_2 = 0.014

    old_probability[0] += increment_0
    old_probability[1] += increment_1
    old_probability[2] += increment_2

    return old_probability


def illuminance_s0_a(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value <= 1000:
        max_increment = 0.01

        offset = variable_value / 1000.
        offset **= 3

        old_probability[0] += max_increment * offset
        old_probability[1] -= 2 * max_increment * offset
        old_probability[2] += max_increment * offset

    return old_probability


def smoke_intensity_s0_a(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value == 0:
        increment_0 = 0.0
        increment_1 = 0.0
        increment_2 = 0.0
    elif variable_value == 1:
        increment_0 = 0.007
        increment_1 = -0.014
        increment_2 = 0.007
    else:
        increment_0 = 0.01
        increment_1 = -0.02
        increment_2 = 0.01

    old_probability[0] += increment_0
    old_probability[1] += increment_1
    old_probability[2] += increment_2

    return old_probability


def obstacle_size_s0_a(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if 0 <= variable_value <= 30:
        offset = variable_value / 20.

        old_probability[0] += 0.1 * offset
        # old_probability[1] -= 0.0 * offset
        old_probability[2] -= 0.1 * offset
    if 20 < variable_value <= 40:
        offset = abs(variable_value - 30.) / 10.
        offset = 1 - offset

        # old_probability[0] += 0.1 * offset
        old_probability[1] -= 0.1 * offset
        old_probability[2] += 0.1 * offset
    if 40 < variable_value:
        offset = (variable_value - 40.) / 80.
        offset **= 2

        old_probability[0] -= 0.01 * offset
        old_probability[1] += 2 * 0.01 * offset
        old_probability[2] -= 0.01 * offset

    return old_probability


def obstacle_distance_s0_a(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value <= 1:
        offset = 1. - variable_value

        old_probability[0] += 0.01 * 2 * offset
        old_probability[1] -= 0.01 * offset
        old_probability[2] -= 0.01 * offset

    if variable_value <= 7:
        offset = variable_value / 7.
        offset = 1. - offset

        old_probability[0] -= 0.01 * offset
        old_probability[1] += 0.03 * offset
        old_probability[2] -= 0.02 * offset

    if 3 <= variable_value:
        offset = (variable_value - 3.) / 7.

        old_probability[0] -= 0.02 * offset
        old_probability[1] -= 0.01 * offset
        old_probability[2] += 0.03 * offset

    return old_probability


def obstacle_distance_s5_g(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value <= 1:
        offset = 1. - variable_value

        old_probability[0] += 0.01 * offset
        old_probability[1] += 0.01 * offset
        old_probability[2] -= 0.01 * 2 * offset
    else:
        offset = (variable_value - 1.) / 9.

        old_probability[0] -= 0.02 * offset
        old_probability[1] -= 0.01 * offset
        old_probability[2] += 0.03 * offset

    return old_probability


def firm_obstacle_s0_a(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value == 1:
        old_probability[1] += 0.015
        old_probability[2] -= 0.015

    return old_probability


def firm_obstacle_s5_g(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value == 1:
        old_probability[0] -= 0.005
        old_probability[1] -= 0.005
        old_probability[2] += 0.01

    return old_probability


def firm_obstacle_s10_l(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value == 1:
        old_probability[0] += 0.01
        old_probability[1] -= 0.01

    return old_probability


def firm_obstacle_s10_m(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value == 1:
        old_probability[0] -= 0.01
        old_probability[1] += 0.01

    return old_probability


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
        "Name": "cruise speed",
        "Type": "SYS",
        "Domain": float,
        "Range": [0, 5],
        "Default": 0.0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": cruise_speed_s0_a
            },
            {
                "StateId": "S5",
                "ActionId": "g",
                "Method": cruise_speed_s5_g
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": cruise_speed_s10_l
            },
            {
                "StateId": "S10",
                "ActionId": "m",
                "Method": cruise_speed_s10_m
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
        "Name": "quality",
        "Type": "SYS",
        "Domain": int,
        "Range": [0, 2],
        "Default": 0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": quality_s0_a
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
                "Method": do_nothing
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
                "Method": illuminance_s0_a
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
                "Method": do_nothing
            },
        ],
    },
    {
        "Name": "smoke intensity",
        "Type": "SYS",
        "Domain": int,
        "Range": [0, 2],
        "Default": 0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": smoke_intensity_s0_a
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
                "Method": do_nothing
            },
        ],
    },
    {
        "Name": "obstacle size",
        "Type": "SYS",
        "Domain": float,
        "Range": [0, 120],
        "Default": 0.0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": obstacle_size_s0_a
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
                "Method": do_nothing
            },
        ],
    },
    {
        "Name": "obstacle distance",
        "Type": "SYS",
        "Domain": float,
        "Range": [0, 10],
        "Default": 0.0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": obstacle_distance_s0_a
            },
            {
                "StateId": "S5",
                "ActionId": "g",
                "Method": obstacle_distance_s5_g
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
        "Name": "firm obstacle",
        "Type": "SYS",
        "Domain": int,
        "Range": [0, 1],
        "Default": 0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": firm_obstacle_s0_a
            },
            {
                "StateId": "S5",
                "ActionId": "g",
                "Method": firm_obstacle_s5_g
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": firm_obstacle_s10_l
            },
            {
                "StateId": "S10",
                "ActionId": "m",
                "Method": firm_obstacle_s10_m
            },
        ],
    },
]
