"""
Simulator noise study (scaffold, not run as part of this task).

mdp_simulator's model checker (action.get_expected(), used by both
region_scores() and check_requirement() in utils/helpers.py) estimates its
point estimates from a posterior over sampled histories, not a closed-form
computation. Two calls to run_mdp() with the SAME input x are not guaranteed
to return identical region_scores or reqs_satisfied. This matters for every
statistical comparison in this project: 30 "independent" runs of a
falsification algorithm confound (a) genuine search-strategy differences with
(b) simulator estimation noise, and nothing in this project currently
measures how large (b) is.

This script fixes one input x, calls run_mdp(x) N times, and reports the
variance in region_scores (per-position) and the flip rate of reqs_satisfied
(how often each requirement's pass/fail flips relative to the majority vote
across the N repeats).

This script has been written and its --help / argument parsing verified, but
NOT run against the simulator (each repeat costs one full MDP model-checking
run). To actually run it:

    cd online-step-experiments/ADAS1
    python3.11 noise_study.py --n 30

Expected output: per-region_scores-position mean/std/CV, and per-requirement
flip rate (0.0 = never flipped across all N repeats, higher = noisier).
"""
import json

import click
import numpy as np

import config as conf
import utils.helpers as helpers


def baseline_point():
    """Midpoint of each variable's declared range, in alphabetically sorted order."""
    var_names = sorted(conf.SS_VARIABLES.keys())
    point = []
    for name in var_names:
        lo, hi = conf.SS_VARIABLES[name]["range"]
        mid = (lo + hi) / 2.0
        if conf.SS_VARIABLES[name]["domain"] == int:
            mid = int(round(mid))
        point.append(mid)
    return var_names, np.array(point, dtype=float)


def parse_x(x_str, var_names):
    """--x accepts either a JSON object {"var": value, ...} or a bare
    comma-separated list of values in the alphabetically sorted variable
    order that create_ss_variables() expects."""
    x_str = x_str.strip()
    if x_str.startswith("{"):
        as_dict = json.loads(x_str)
        return np.array([as_dict[name] for name in var_names], dtype=float)
    return np.array([float(v) for v in x_str.split(",")], dtype=float)


@click.command()
@click.option("--n", default=30, help="Number of repeated run_mdp(x) calls.")
@click.option("--x", default=None,
              help="Fixed input point, as JSON {\"var\": value, ...} or a bare "
                   "comma-separated list in alphabetically-sorted variable order. "
                   "If omitted, uses the midpoint of each variable's declared range.")
def main(n, x):
    var_names, base = baseline_point()
    point = parse_x(x, var_names) if x else base

    click.echo(f"Fixed input point ({dict(zip(var_names, point))})")
    click.echo(f"Calling run_mdp() {n} times...")

    all_region_scores = []
    all_reqs_satisfied = []
    for i in range(n):
        _, region_scores, reqs_satisfied, _ = helpers.run_mdp(point)
        all_region_scores.append(region_scores)
        all_reqs_satisfied.append(reqs_satisfied)

    region_arr = np.array(all_region_scores, dtype=float)  # (n, n_regions)
    reqs_arr = np.array(all_reqs_satisfied, dtype=bool)    # (n, n_reqs)

    click.echo("\n" + "=" * 70)
    click.echo("REGION SCORE VARIANCE (per position)")
    click.echo("=" * 70)
    for j in range(region_arr.shape[1]):
        col = region_arr[:, j]
        mean, std = col.mean(), col.std()
        cv = std / abs(mean) if mean != 0 else float("inf")
        click.echo(f"  V{j}: mean={mean:.6f} std={std:.6f} cv={cv:.4f} "
                   f"range=[{col.min():.6f}, {col.max():.6f}]")

    click.echo("\n" + "=" * 70)
    click.echo("REQUIREMENT SATISFACTION FLIP RATE (relative to majority vote)")
    click.echo("=" * 70)
    for j in range(reqs_arr.shape[1]):
        col = reqs_arr[:, j]
        majority = col.mean() >= 0.5
        flip_rate = np.mean(col != majority)
        click.echo(f"  R{j}: majority={'satisfied' if majority else 'violated'} "
                   f"flip_rate={flip_rate:.4f} ({int(flip_rate * n)}/{n} repeats disagreed)")


if __name__ == "__main__":
    main()
