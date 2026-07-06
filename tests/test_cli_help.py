"""
Fast smoke tests: each falsification script must at least import cleanly and
parse its CLI (`--help`) without touching the simulator. This catches syntax
errors, missing imports, and broken argparse/click option definitions in a
couple of seconds per script, well before anyone kicks off a multi-hour run.
"""
import os

import pytest

from conftest import EXPERIMENTS_ROOT, run_script

CLICK_SCRIPTS = ["FOC_falsification.py", "PFRL_falsification.py", "PFES_falsification.py"]
ARGPARSE_SCRIPTS = ["PFES_SAMOTA.py"]


@pytest.mark.parametrize("script", CLICK_SCRIPTS + ARGPARSE_SCRIPTS)
def test_script_help_runs_cleanly(python311, benchmark, benchmark_dir, script):
    path = os.path.join(benchmark_dir, script)
    if not os.path.exists(path):
        pytest.skip(f"{script} does not exist for {benchmark}")

    result = run_script(python311, benchmark_dir, script, ["--help"], timeout=60)
    assert result.returncode == 0, (
        f"{benchmark}/{script} --help failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()


@pytest.mark.parametrize("script", CLICK_SCRIPTS + ARGPARSE_SCRIPTS)
def test_script_rejects_unknown_option(python311, benchmark, benchmark_dir, script):
    """A quick check that the CLI actually validates its arguments (nonzero
    exit on a bogus flag), i.e. we're not accidentally looking at a script
    that silently ignores --help/unknown flags and runs the full experiment."""
    path = os.path.join(benchmark_dir, script)
    if not os.path.exists(path):
        pytest.skip(f"{script} does not exist for {benchmark}")

    result = run_script(python311, benchmark_dir, script, ["--definitely-not-a-real-option"], timeout=60)
    assert result.returncode != 0
