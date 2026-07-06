"""
Slow, end-to-end smoke tests: actually invoke each falsification script against
the real simulator with a tiny budget/run count, and check it exits cleanly and
produces the expected output files. These are the tests that would have caught
the variable-order/out-of-range crashes before a full multi-hour experiment run.

Skipped by default (they take real wall-clock time to run the simulator).
Run explicitly with:

    python3.11 -m pytest tests/test_smoke_runs.py --run-slow -v
"""
import os

import pytest

from conftest import run_script

pytestmark = pytest.mark.slow

TIMEOUT = 600


def test_foc_smoke_run(python311, benchmark, benchmark_dir, tmp_path):
    args = ["--size", "4", "--totbudget", "8", "--nruns", "1",
            "--logdir", str(tmp_path), "--seed", "1"]
    result = run_script(python311, benchmark_dir, "FOC_falsification.py", args, timeout=TIMEOUT)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    for name in ("reqs_FOC_1.csv", "score_FOC_1.csv", "timing_FOC_1.csv"):
        assert (tmp_path / name).exists(), f"expected output missing: {name}"


def test_pfrl_smoke_run(python311, benchmark, benchmark_dir, tmp_path):
    args = ["--nepisodes", "3", "--nruns", "1",
            "--logdir", str(tmp_path), "--seed", "1"]
    result = run_script(python311, benchmark_dir, "PFRL_falsification.py", args, timeout=TIMEOUT)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    for name in ("reqs_MORLOT_1.csv", "score_MORLOT_1.csv", "timing_MORLOT_1.csv"):
        assert (tmp_path / name).exists(), f"expected output missing: {name}"


def test_pfes_smoke_run(python311, benchmark, benchmark_dir, tmp_path):
    args = ["--size", "4", "--niterations", "1", "--nruns", "1",
            "--logdir", str(tmp_path), "--seed", "1"]
    result = run_script(python311, benchmark_dir, "PFES_falsification.py", args, timeout=TIMEOUT)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    for name in ("reqs_NSGA3_1.csv", "score_NSGA3_1.csv"):
        assert (tmp_path / name).exists(), f"expected output missing: {name}"


def test_samota_smoke_run(python311, benchmark, benchmark_dir, tmp_path):
    args = ["--budget", "10", "--nruns", "1", "--logdir", str(tmp_path), "--seed", "1"]
    result = run_script(python311, benchmark_dir, "PFES_SAMOTA.py", args, timeout=TIMEOUT)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    for name in ("reqs_SAMOTA_1.csv", "score_SAMOTA_1.csv", "meta_SAMOTA_1.csv", "timing_SAMOTA_1.csv"):
        assert (tmp_path / name).exists(), f"expected output missing: {name}"
