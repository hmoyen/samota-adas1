"""
Sanity checks on the Python environment these experiments depend on.

These exist because on this machine `python3`/`python` resolve to 3.12, which
has an incompatible arviz (1.1.0) installed, while only python3.11 has the
pinned arviz==0.14.0 + matching numpy/scipy/pandas/pymoo versions that
CPS-simulator/mdp_simulator actually needs. Running under the wrong
interpreter doesn't fail loudly - action.py's az.hdi() call silently computes
over the wrong axis and produces garbage stats instead of crashing - so these
checks are here to catch that mistake before anyone spends hours running a
falsification experiment under the wrong interpreter.
"""
import os

from conftest import PYTHON311, run_in_benchmark


def test_python311_exists():
    assert os.path.exists(PYTHON311), (
        f"{PYTHON311} not found - these experiments require python3.11, "
        f"NOT the system python3/python (which may resolve to 3.12)."
    )


def test_arviz_version_pinned(python311, benchmark_dir):
    result = run_in_benchmark(
        python311, benchmark_dir,
        "import arviz; print(arviz.__version__)",
    )
    assert result.returncode == 0, result.stderr
    version = result.stdout.strip()
    assert version == "0.14.0", (
        f"arviz=={version} is importable under python3.11, expected 0.14.0. "
        f"arviz>=1.0 changes az.hdi()'s behavior on raw numpy input (not just "
        f"an API rename) and will silently produce wrong statistics."
    )


def test_core_dependencies_importable(python311, benchmark_dir):
    code = (
        "import numpy, scipy, pandas, pymoo, click, sklearn\n"
        "print('ok')\n"
    )
    result = run_in_benchmark(python311, benchmark_dir, code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_mdp_simulator_importable(python311, benchmark_dir):
    """
    utils/helpers.py adds CPS-simulator to sys.path relative to its own
    location before importing mdp_simulator. Verify that resolution actually
    works from each benchmark directory.
    """
    code = (
        "import utils.helpers as helpers\n"
        "from mdp_simulator import config, run, enums\n"
        "print('ok')\n"
    )
    result = run_in_benchmark(python311, benchmark_dir, code, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_config_and_simulator_input_folder_exist(python311, benchmark_dir):
    code = (
        "import os\n"
        "import config as conf\n"
        "assert os.path.isdir(conf.MDP_FOLDER), "
        "f'MDP_FOLDER does not exist: {conf.MDP_FOLDER}'\n"
        "print('ok')\n"
    )
    result = run_in_benchmark(python311, benchmark_dir, code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
