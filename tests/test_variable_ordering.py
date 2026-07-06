"""
Regression tests for the variable-order scrambling bug class.

Background: helpers.create_ss_variables() reads the input array assuming
sorted(conf.SS_VARIABLES.keys()) order. Several call sites independently built
that array (build_random_combinations, FOC_falsification.py's ADAS_VAR_NAMES /
RR_VAR_NAMES, PFRL_falsification.py's sorted_order_indices remap,
PFES_falsification.py's CSV column labels) and it was easy for one of them to
drift out of sync and silently assign a value to the wrong SS variable
(e.g. a bandwidth value landing on firm_obstacle, or car_speed's value landing
on orientation) - sometimes crashing with an out-of-range error, sometimes not
crashing at all and just producing wrong results.

These tests exercise the actual helper functions (not a hand-copied
reimplementation) from each benchmark's own utils/helpers.py, in a subprocess
with cwd set to that benchmark, so a future edit that reintroduces insertion
order instead of sorted order gets caught immediately.
"""
import ast
import os

from conftest import EXPERIMENTS_ROOT, run_in_benchmark


def test_create_ss_variables_uses_sorted_order(python311, benchmark_dir):
    code = (
        "import config as conf\n"
        "import utils.helpers as helpers\n"
        "var_names = sorted(conf.SS_VARIABLES.keys())\n"
        "n = len(var_names)\n"
        "# distinct probe values so a swap between any two positions is detectable\n"
        "probe = list(range(1, n + 1))\n"
        "result = helpers.create_ss_variables(conf.SS_VARIABLES, probe)\n"
        "for i, name in enumerate(var_names):\n"
        "    domain = conf.SS_VARIABLES[name]['domain']\n"
        "    expected = domain(probe[i])\n"
        "    assert result[name] == expected, f'{name}: expected {expected}, got {result[name]}'\n"
        "print('ok')\n"
    )
    result = run_in_benchmark(python311, benchmark_dir, code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("ok")


def test_build_random_combinations_uses_sorted_order_and_bounds(python311, benchmark_dir):
    code = (
        "import config as conf\n"
        "import utils.helpers as helpers\n"
        "var_names = sorted(conf.SS_VARIABLES.keys())\n"
        "combos = helpers.build_random_combinations(5)\n"
        "assert len(combos) == 5\n"
        "for combo in combos:\n"
        "    assert len(combo) == len(var_names)\n"
        "    for i, name in enumerate(var_names):\n"
        "        lo, hi = conf.SS_VARIABLES[name]['range']\n"
        "        assert lo <= combo[i] <= hi, f'{name}: {combo[i]} not in [{lo}, {hi}]'\n"
        "print('ok')\n"
    )
    result = run_in_benchmark(python311, benchmark_dir, code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("ok")


def test_clamp_to_bounds_uses_sorted_order_and_clamps(python311, benchmark_dir):
    code = (
        "import config as conf\n"
        "import utils.helpers as helpers\n"
        "var_names = sorted(conf.SS_VARIABLES.keys())\n"
        "# push every value far outside its own range to force clamping\n"
        "oversized = [conf.SS_VARIABLES[name]['range'][1] + 1000 for name in var_names]\n"
        "clamped = helpers.clamp_to_bounds(conf.SS_VARIABLES, oversized)\n"
        "for i, name in enumerate(var_names):\n"
        "    lo, hi = conf.SS_VARIABLES[name]['range']\n"
        "    assert clamped[i] == hi, f'{name}: expected clamp to {hi}, got {clamped[i]}'\n"
        "print('ok')\n"
    )
    result = run_in_benchmark(python311, benchmark_dir, code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("ok")


def _get_module_level_list_literal(source, target_name):
    """Statically extract a module-level `NAME = sorted([...])` or `NAME = [...]`
    list-of-strings assignment from source, without importing/executing the
    module (which has heavy pymoo/click imports)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == target_name:
            value = node.value
            # unwrap sorted(...)
            if isinstance(value, ast.Call) and getattr(value.func, "id", None) == "sorted":
                value = value.args[0]
            if isinstance(value, ast.List):
                return [elt.value for elt in value.elts]
    return None


def test_foc_var_names_match_config(benchmark):
    """FOC_falsification.py builds its own ADAS_VAR_NAMES/RR_VAR_NAMES constant
    used to order values passed into run_mdp()/run_mdp_sensitivity(). Verify it
    is still consistent with config.SS_VARIABLES, statically (importing this
    module directly would drag in pymoo/click just to check a constant)."""
    path = os.path.join(EXPERIMENTS_ROOT, benchmark, "FOC_falsification.py")
    config_path = os.path.join(EXPERIMENTS_ROOT, benchmark, "config.py")
    if not os.path.exists(path):
        import pytest
        pytest.skip(f"{path} does not exist")

    with open(path) as f:
        foc_source = f.read()
    with open(config_path) as f:
        config_source = f.read()

    ss_variables_keys = _get_ss_variables_keys(config_source)

    var_names = None
    for candidate in ("ADAS_VAR_NAMES", "RR_VAR_NAMES"):
        var_names = _get_module_level_list_literal(foc_source, candidate)
        if var_names is not None:
            break
    assert var_names is not None, (
        "FOC_falsification.py no longer defines an ADAS_VAR_NAMES/RR_VAR_NAMES "
        "constant - if it now builds the run_mdp() input array some other way, "
        "update this test to check that instead."
    )
    assert sorted(var_names) == sorted(ss_variables_keys)


def _get_ss_variables_keys(config_source):
    tree = ast.parse(config_source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "SS_VARIABLES":
            return [k.value for k in node.value.keys]
    raise AssertionError("SS_VARIABLES not found in config.py")


def test_pfrl_ss_vars_match_config(benchmark):
    """PFRL_falsification.py's `ss_vars` list (inside main()) drives its RL
    state/action space in insertion order, and is separately remapped to
    sorted order via `sorted_order_indices` right before calling run_mdp().
    Verify the ss_vars list still names exactly the current SS_VARIABLES
    (no stale/renamed/missing variable), so the sorted_order_indices remap
    stays valid."""
    path = os.path.join(EXPERIMENTS_ROOT, benchmark, "PFRL_falsification.py")
    config_path = os.path.join(EXPERIMENTS_ROOT, benchmark, "config.py")
    if not os.path.exists(path):
        import pytest
        pytest.skip(f"{path} does not exist")

    with open(path) as f:
        pfrl_source = f.read()
    with open(config_path) as f:
        config_source = f.read()

    ss_variables_keys = _get_ss_variables_keys(config_source)

    tree = ast.parse(pfrl_source)
    ss_vars = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "ss_vars" \
                and isinstance(node.value, ast.List):
            ss_vars = [elt.value for elt in node.value.elts]
            break
    assert ss_vars is not None, "ss_vars list literal not found in PFRL_falsification.py"
    assert sorted(ss_vars) == sorted(ss_variables_keys)

    assert "sorted_order_indices" in pfrl_source, (
        "PFRL_falsification.py no longer references sorted_order_indices - "
        "if run_mdp() is now called with a differently-ordered assignment, "
        "make sure it's still remapped to sorted(config.SS_VARIABLES) order."
    )


def test_pfes_falsification_csv_columns_use_sorted_order(benchmark):
    """PFES_falsification.py's X_all_evaluations_*.csv column headers must be
    built from sorted(conf.SS_VARIABLES.keys()) - either directly, or via a
    module-level *_VAR_NAMES constant that is itself `sorted([...])` (already
    checked against config.py in test_foc_var_names_match_config). Previously
    this was a hardcoded insertion-order list literal, which silently
    mislabeled the (correctly-ordered) data columns."""
    path = os.path.join(EXPERIMENTS_ROOT, benchmark, "PFES_falsification.py")
    config_path = os.path.join(EXPERIMENTS_ROOT, benchmark, "config.py")
    if not os.path.exists(path):
        import pytest
        pytest.skip(f"{path} does not exist")
    with open(path) as f:
        source = f.read()

    tree = ast.parse(source)
    columns_arg = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and getattr(node.value.func, "attr", None) == "DataFrame":
            for kw in node.value.keywords:
                if kw.arg == "columns":
                    # match the X (parameters) DataFrame, not the F/objectives one
                    target = node.targets[0]
                    if isinstance(target, ast.Name) and target.id == "X_df":
                        columns_arg = kw.value
    assert columns_arg is not None, "Could not find X_df = pd.DataFrame(..., columns=...) in PFES_falsification.py"

    if isinstance(columns_arg, ast.Call) and getattr(columns_arg.func, "id", None) == "sorted":
        return  # columns=sorted(conf.SS_VARIABLES.keys()) directly

    if isinstance(columns_arg, ast.Name):
        with open(config_path) as f:
            config_source = f.read()
        ss_variables_keys = _get_ss_variables_keys(config_source)
        var_names = _get_module_level_list_literal(source, columns_arg.id)
        assert var_names is not None and sorted(var_names) == sorted(ss_variables_keys), (
            f"columns={columns_arg.id} does not resolve to a sorted(...) literal "
            f"matching config.SS_VARIABLES"
        )
        return

    raise AssertionError(
        f"X_df columns= is neither `sorted(...)` nor a recognizable *_VAR_NAMES "
        f"constant: {ast.dump(columns_arg)}"
    )
