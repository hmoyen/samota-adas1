"""
Shared fixtures/helpers for the online-step-experiments test suite.

Each benchmark (ADAS1, ADAS2, RR) has its own `config.py` and `utils/helpers.py`
module, both imported as bare `config` / `utils.helpers`. Importing more than one
benchmark's copy in the same process would collide via Python's sys.modules
cache, so every test that needs a benchmark's modules runs them in a fresh
`python3.11` subprocess with `cwd` set to that benchmark's directory, instead of
importing them directly into the test process.
"""
import os
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_ROOT = os.path.join(REPO_ROOT, "online-step-experiments")

BENCHMARKS = ["ADAS1", "ADAS2", "RR"]

PYTHON311 = shutil.which("python3.11") or "/usr/bin/python3.11"


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="Also run tests marked @pytest.mark.slow (real, short simulator runs).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def python311():
    if not os.path.exists(PYTHON311):
        pytest.skip(f"python3.11 not found at {PYTHON311}; required for these experiments")
    return PYTHON311


@pytest.fixture(params=BENCHMARKS)
def benchmark(request):
    """Parametrized fixture yielding each benchmark name in turn."""
    return request.param


@pytest.fixture
def benchmark_dir(benchmark):
    path = os.path.join(EXPERIMENTS_ROOT, benchmark)
    if not os.path.isdir(path):
        pytest.skip(f"benchmark directory not found: {path}")
    return path


def run_in_benchmark(python311, benchmark_dir, code, timeout=120):
    """
    Run `code` as a `python3.11 -c` subprocess with cwd set to benchmark_dir,
    so `import config` / `import utils.helpers` resolve to that benchmark's
    own copies. Returns the completed subprocess.CompletedProcess.
    """
    return subprocess.run(
        [python311, "-c", code],
        cwd=benchmark_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_script(python311, benchmark_dir, script_name, args, timeout=120):
    """Run one of the benchmark's scripts (e.g. FOC_falsification.py) with args."""
    return subprocess.run(
        [python311, script_name] + args,
        cwd=benchmark_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
