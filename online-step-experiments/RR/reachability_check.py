"""
RR requirement reachability check (scaffold, not run as part of this task).

Across the full results/RR dataset (6 algorithms x 30 runs), two of RR's six
requirements (R4, R5 - the check_requirement() calls built from
conf.CONSTRAINTS[4] and conf.CONSTRAINTS[5]) are never violated by anyone.
That's consistent with either (a) those requirements being genuinely
unreachable within the declared SS_VARIABLES bounds, or (b) them being
reachable but only in a tiny corner of the 9D search space that no algorithm's
900-eval budget has stumbled into yet.

check_requirement() compares MDP model-checking output (action.get_expected())
against fixed bounds - there's no closed-form way to derive reachability from
config.py alone, since the input->point_estimate mapping goes through the MDP
model checker. So this script argues about reachability the only way available
short of a full search: it evaluates a bounded, targeted set of extreme points
(and, optionally, extra random points) through the real simulator and reports
whether any of them violate the target requirement(s).

A CONFIRMED "not violated in N evaluated points" is NOT proof of
unreachability - it only narrows the question. A single found violation IS
proof of reachability.

This script has been written and its --help / argument parsing verified, but
NOT run against the simulator (each point costs one full MDP model-checking
run). To actually run it:

    cd online-step-experiments/RR
    python3.11 reachability_check.py --req_indices 4,5 --strategy corners --random_samples 50

Expected output: a per-requirement verdict line, e.g.
    R4: REACHABLE - violated by point {...} (source: one-at-a-time corner, var=quality=2)
    R5: NOT REACHED in 68 evaluated points (inconclusive, not proof of unreachability)
"""
import itertools
import json
import sys

import click
import numpy as np

import config as conf
import utils.helpers as helpers


def detect_never_violated_reqs(results_dir):
    """Read results/RR/*/out/reqs_*.csv (read-only) and return the indices of
    requirements that were never violated (sum of violation counts == 0)
    across every algorithm/run found there."""
    import glob
    import os
    import pandas as pd

    algos = ["PF", "RS", "FF", "MERLOT", "SAMOTA", "SAMOTA_SW"]
    totals = None
    for algo in algos:
        out_dir = os.path.join(results_dir, "RR", algo, "out")
        for f in glob.glob(os.path.join(out_dir, "reqs_*.csv")):
            df = pd.read_csv(f)
            req_cols = [c for c in df.columns if c.startswith("R") and c[1:].isdigit()]
            s = df[req_cols].sum(axis=0)
            totals = s if totals is None else totals.add(s, fill_value=0)
    if totals is None:
        return []
    return [int(c[1:]) for c in totals.index if totals[c] == 0]


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


def corner_points():
    """One-at-a-time extreme sweep: baseline point, then each variable pushed
    to its min and max while the rest stay at baseline. 1 + 2*n_vars points
    for RR's 9 variables = 19 points, cheap enough to always include."""
    var_names, base = baseline_point()
    points = [("baseline", dict(zip(var_names, base)), base.copy())]
    for i, name in enumerate(var_names):
        lo, hi = conf.SS_VARIABLES[name]["range"]
        for bound_name, bound_val in [("min", lo), ("max", hi)]:
            p = base.copy()
            p[i] = bound_val
            desc = dict(zip(var_names, p))
            points.append((f"{name}={bound_name}", desc, p))
    return var_names, points


def full_corner_points(max_points):
    """All 2^n_vars combinations of min/max per variable (capped at
    max_points, since 2^9=512 for RR). Use only if --strategy full_corners
    and you're willing to pay for it."""
    var_names = sorted(conf.SS_VARIABLES.keys())
    bounds = [conf.SS_VARIABLES[name]["range"] for name in var_names]
    points = []
    for combo in itertools.product(*[(lo, hi) for lo, hi in bounds]):
        p = np.array(combo, dtype=float)
        desc = dict(zip(var_names, p))
        points.append(("full_corner", desc, p))
        if len(points) >= max_points:
            break
    return var_names, points


def random_points(n, seed):
    var_names = sorted(conf.SS_VARIABLES.keys())
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(n):
        p = []
        for name in var_names:
            lo, hi = conf.SS_VARIABLES[name]["range"]
            if conf.SS_VARIABLES[name]["domain"] == int:
                p.append(rng.integers(lo, hi + 1))
            else:
                p.append(rng.uniform(lo, hi))
        p = np.array(p, dtype=float)
        desc = dict(zip(var_names, p))
        points.append(("random", desc, p))
    return var_names, points


@click.command()
@click.option("--req_indices", default=None,
              help="Comma-separated requirement indices to test (0-based, matching "
                   "conf.CONSTRAINTS order). If omitted, auto-detected from "
                   "--results_dir as the requirements with zero violations there.")
@click.option("--results_dir", default="../../results",
              help="Path to the results/ directory, used only to auto-detect "
                   "never-violated requirements when --req_indices is omitted.")
@click.option("--strategy", default="corners",
              type=click.Choice(["corners", "full_corners", "random_only"]),
              help="corners = baseline + one-at-a-time min/max sweep (19 points for RR). "
                   "full_corners = all 2^n_vars combinations, capped by --max_points. "
                   "random_only = skip the corner sweep, use only random samples.")
@click.option("--max_points", default=512, help="Cap on full_corners points.")
@click.option("--random_samples", default=0,
              help="Additional uniform-random points to test on top of the corner strategy.")
@click.option("--seed", default=0, help="Seed for --random_samples.")
def main(req_indices, results_dir, strategy, max_points, random_samples, seed):
    if req_indices is None:
        targets = detect_never_violated_reqs(results_dir)
        if not targets:
            click.echo(f"[ERROR] Could not auto-detect never-violated requirements from "
                       f"{results_dir} (no reqs_*.csv found there). Pass --req_indices explicitly.")
            sys.exit(1)
        click.echo(f"Auto-detected never-violated requirements from {results_dir}: {targets}")
    else:
        targets = [int(x) for x in req_indices.split(",")]

    if strategy == "corners":
        var_names, points = corner_points()
    elif strategy == "full_corners":
        var_names, points = full_corner_points(max_points)
    else:
        var_names, points = [], []

    if random_samples > 0:
        _, extra = random_points(random_samples, seed)
        points = points + extra
        var_names = sorted(conf.SS_VARIABLES.keys())

    click.echo(f"Testing {len(points)} points against requirement(s) {targets} "
               f"(each point = one full MDP simulator run)...")

    # For each target requirement, find its constraint definition once.
    found = {req_idx: None for req_idx in targets}

    for source, desc, arr in points:
        if all(found[r] is not None for r in targets):
            break
        _, _, reqs_satisfied, _ = helpers.run_mdp(arr)
        for req_idx in targets:
            if found[req_idx] is not None:
                continue
            # reqs_satisfied[i] is True if satisfied; violated == not satisfied
            if req_idx < len(reqs_satisfied) and not reqs_satisfied[req_idx]:
                found[req_idx] = (source, desc)

    click.echo("\n" + "=" * 70)
    click.echo("REACHABILITY VERDICT")
    click.echo("=" * 70)
    for req_idx in targets:
        if found[req_idx] is not None:
            source, desc = found[req_idx]
            click.echo(f"R{req_idx}: REACHABLE - violated by point "
                       f"(source: {source})\n       {json.dumps(desc, default=float)}")
        else:
            click.echo(f"R{req_idx}: NOT REACHED in {len(points)} evaluated points "
                       f"(inconclusive, not proof of unreachability)")


if __name__ == "__main__":
    main()
