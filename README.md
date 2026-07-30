# ICSE 2025 Replication Package: SAMOTA + PFES

**SAMOTA implementation and baselines across three benchmark systems**

This repository contains the SAMOTA (Surrogate-Assisted Many-Objective Test case generation
Algorithm) implementation, PFES, and four other falsification baselines, evaluated on three
benchmark systems: **ADAS1** (autonomous driving, 6D/3 requirements), **ADAS2** (autonomous
driving, 6D/6 requirements), and **RR** (rescue robot, 9D/6 requirements). Each benchmark
directory under `online-step-experiments/` is self-contained (own `config.py`,
`utils/helpers.py`, algorithm implementations).

## Overview

SAMOTA combines machine learning surrogates with multi-objective optimization to efficiently
generate test cases that violate safety constraints with minimal simulator evaluations. Six
algorithms are compared per benchmark: PF (pure NSGA3), RS (random search), FF (focused
falsification), MERLOT (PFES+RL), SAMOTA, and SAMOTA+SW (sliding window).

**Key Features**:
- Full SAMOTA algorithm (Phase 1 ART + Phase 2 GS+LS) plus five baselines
- Three benchmark systems (ADAS1, ADAS2, RR)
- Comparative statistics framework (`analyze_all_results.py`): Mann-Whitney U + Vargha-Delaney
  A_12 with Holm-Bonferroni correction, bootstrap CIs, and Kruskal-Wallis omnibus tests
- Poetry or pip-based dependency management
- `pytest` test suite (`tests/`), run in CI on every push/PR

## Quick Start

### 1. Install Python 3.11

The pinned dependency versions (below) require Python 3.11 specifically — arviz 0.14.0 and
its compatible numpy/scipy/pandas versions are not available for newer Python releases.

### 2. Install Dependencies

Either via Poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
poetry env use python3.11
poetry install
poetry shell
```

or directly with pip (matches the versions pinned in CI, `.github/workflows/tests.yml`):

```bash
python3.11 -m pip install \
    numpy==1.26.4 scipy==1.10.1 pandas==1.5.3 arviz==0.14.0 pymoo==0.6.1.6 \
    scikit-learn==1.9.0 hdbscan==0.8.44 click==8.1.6 colorama==0.4.6 \
    termcolor==2.5.0 pymdptoolbox==4.0b3 reportlab==4.5.1 matplotlib==3.10.9 pytest==7.4.4
```

Then install the bundled simulator wheel:

```bash
python3.11 -m pip install CPS-simulator/dist/mdp_simulator-0.1.9-py3-none-any.whl
```

### 3. Run the test suite

```bash
python3.11 -m pytest tests/ -q
```

### 4. Run comparative experiments

```bash
cd online-step-experiments/ADAS1   # or ADAS2 / RR
python3.11 run_comparative_experiments.py --runs 5 --budget 900 --output results
```

### 5. Check Results

```
results/
├── summary.csv
├── pfes_runs.csv
├── pfes_samota_runs.csv
└── efficiency_analysis.txt
```

## Repository Structure

```
clean_repo/
├── pyproject.toml                      # Poetry configuration
├── README.md                           # This file
├── analyze_all_results.py              # Cross-algorithm statistical analysis (all benchmarks)
├── tests/                              # pytest suite (subprocess-isolated per benchmark)
├── .github/workflows/tests.yml         # CI: runs pytest on push/PR
├── CPS-simulator/
│   ├── dist/
│   │   └── mdp_simulator-0.1.9-py3-none-any.whl
│   └── README.md
└── online-step-experiments/
    ├── ADAS1/                          # 6D search space, 3 requirements
    ├── ADAS2/                          # 6D search space, 6 requirements
    └── RR/                             # 9D search space, 6 requirements
        ├── PFES_SAMOTA.py              # Main SAMOTA implementation
        ├── PFES_falsification.py       # PFES baseline
        ├── FOC_falsification.py        # FF baseline
        ├── PFRL_falsification.py       # MERLOT baseline
        ├── run_comparative_experiments.py   # Stats framework
        ├── config.py                   # Configuration
        ├── SAMOTA_ensemble.py          # Surrogate ensembles
        ├── RBF.py                      # RBF model
        ├── utils/
        │   ├── helpers.py              # Simulator wrapper
        │   └── constraints_builder.py  # Constraint utilities
        └── INPUT/                      # Simulator configuration
```

## Algorithms Implemented

### Algorithm 1: Main PFES+SAMOTA Loop
- **Phase 1**: Adaptive Random Testing (300 evals, Maximin sampling)
- **Phase 2**: Iterative Global Search + Local Search (600 evals budget)
- Dynamic uncovered objective filtering

### Algorithm 2: Global Search (GS)
- Per-objective ensemble surrogates (GP + Polynomial + RBF)
- Multi-objective NSGA3 for trade-off discovery
- Best + uncertain selection per objective
- ~2-6 candidates per iteration

### Algorithm 3: Local Search (LS)
- Top 20% filtering per objective
- HDBSCAN clustering (min 5 samples)
- Single RBF surrogate per cluster
- Single-objective NSGA3 per objective
- ~5-30 candidates per iteration

### Algorithm 4: Surrogate Ensemble
- Gaussian Process (GP)
- Polynomial Regression (degree 2)
- RBF Network (10 neurons)
- Goel-weighted ensemble

See `online-step-experiments/ADAS1/PFES_SAMOTA_ALGORITHMS_CORRECTED.md` for formal pseudocode
(same algorithm structure applies to ADAS2 and RR, with per-benchmark search spaces and
requirements).

## Objectives (ADAS1 example)

ADAS2 and RR have their own objective/requirement definitions in their respective `config.py` —
see each benchmark's `CONSTRAINTS` for details. ADAS1 has **5 optimization objectives** (not 3):

| Objective | Source | Bounds | Description |
|-----------|--------|--------|-------------|
| **Obj 0** | S0.a[0] | [lower, upper] | Autonomy bound 1 |
| **Obj 1** | S0.a[1] | [lower, upper] | Autonomy bound 2 |
| **Obj 2** | S2.b[0] | value | Behavior element 1 |
| **Obj 3** | S2.b[1] | value | Behavior element 2 |
| **Obj 4** | S2.b[2] | value | Behavior element 3 |

**3 Constraint Satisfaction States (Requirements)**:
- **R0**: S0.a satisfied (both bounds met)
- **R1**: S2.b satisfied (all 3 elements within bounds)
- **R2**: Both R0 AND R1 satisfied

## Configuration

Edit `online-step-experiments/ADAS1/config.py`:

```python
# Search space (6D)
SS_VARIABLES = {
    "car_speed": {"domain": float, "range": [5.0, 50.0]},
    "p_x": {"domain": float, "range": [0.0, 10.0]},
    "p_y": {"domain": float, "range": [0.0, 10.0]},
    "orientation": {"domain": int, "range": [-30, 30]},
    "weather": {"domain": int, "range": [0, 2]},
    "road_shape": {"domain": int, "range": [0, 2]},
}

# 3 constraints (R0, R1, R2)
CONSTRAINTS = [...]
```

See `results/<benchmark>/<algorithm>/out/` for actual violation-count and coverage CSVs from
the 30-run experiments, and run `python3.11 analyze_all_results.py` for the statistical
comparison across all six algorithms and three benchmarks.

## Usage Examples

### Run PFES+SAMOTA only

```bash
cd online-step-experiments/ADAS1   # or ADAS2 / RR
python3.11 -c "import PFES_SAMOTA; PFES_SAMOTA.run_pfes_samota(budget=900, max_iterations=30)"
```

### Run PFES baseline only

```bash
cd online-step-experiments/ADAS1
python3.11 -c "import PFES_falsification; PFES_falsification.run_pfes(max_evaluations=900)"
```

### Run all six algorithms for comparison

```bash
cd online-step-experiments/ADAS1
python3.11 run_comparative_experiments.py --runs 30 --budget 900 --output results_comparison
```

## Troubleshooting

### Python version error
Make sure you're using `python3.11` specifically (see "Install Python 3.11" above), not
whatever `python`/`python3` resolves to by default.

### mdp_simulator not found
```bash
python3.11 -m pip install CPS-simulator/dist/mdp_simulator-0.1.9-py3-none-any.whl
```
(prefix with `poetry run` if using Poetry)

## Documentation

- **online-step-experiments/ADAS1/EXPERIMENT_GUIDE.md** — Results interpretation
- **online-step-experiments/ADAS1/PFES_SAMOTA_ALGORITHMS_CORRECTED.md** — Formal algorithms

## Project Information

- **Author**: Helena Moyen
- **Framework**: ICSE 2025 Replication Package
- **License**: MIT
- **Dependencies**: Managed via Poetry, or pip with the pinned versions above
