# ICSE 2025 Replication Package: SAMOTA + PFES

**Full SAMOTA Implementation for Autonomous Driving System Testing**

This is a clean repository containing the complete implementation of the SAMOTA (Surrogate-Assisted Many-Objective Test case generation Algorithm) approach integrated with PFES for online adaptive test generation on the ADAS1 benchmark system.

## Overview

SAMOTA combines machine learning surrogates with multi-objective optimization to efficiently generate test cases that violate safety constraints with minimal simulator evaluations.

**Key Features**:
- ✅ Full SAMOTA algorithm (Phase 1 ART + Phase 2 GS+LS)
- ✅ PFES baseline for comparison
- ✅ Comparative stats framework (CSV + statistical analysis)
- ✅ Poetry dependency management
- ✅ 3 safety constraints (R0, R1, R2)

## Quick Start

### 1. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Install Dependencies

```bash
poetry install
```

### 3. Run Comparative Experiments

```bash
poetry shell
cd online-step-experiments/ADAS1
python run_comparative_experiments.py --runs 5 --budget 900 --output results
```

### 4. Check Results

```
results/
├── summary.csv
├── pfes_runs.csv
├── pfes_samota_runs.csv
└── efficiency_analysis.txt
```

**See [QUICK_START.md](QUICK_START.md) for detailed instructions.**

## Repository Structure

```
icse2025-samota-adas1/
├── pyproject.toml                      # Poetry configuration
├── README.md                           # This file
├── QUICK_START.md                      # Poetry setup guide
├── CPS-simulator/
│   ├── dist/
│   │   └── mdp_simulator-0.1.9-py3-none-any.whl
│   └── README.md
└── online-step-experiments/
    └── ADAS1/
        ├── PFES_SAMOTA.py              # Main SAMOTA implementation
        ├── PFES_falsification.py       # PFES baseline
        ├── run_comparative_experiments.py   # Stats framework
        ├── config.py                   # Configuration
        ├── SAMOTA_ensemble.py          # Surrogate ensembles
        ├── RBF.py                      # RBF model
        ├── EXPERIMENT_GUIDE.md         # Results interpretation
        ├── PFES_SAMOTA_ALGORITHMS_CORRECTED.md  # Formal algorithms
        ├── utils/
        │   ├── helpers.py              # Simulator wrapper
        │   └── constraints_builder.py  # Constraint utilities
        └── INPUT/
            └── AutonomousDriving_v1/   # Simulator configuration
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

See `online-step-experiments/ADAS1/PFES_SAMOTA_ALGORITHMS_CORRECTED.md` for formal pseudocode.

## Safety Constraints (ADAS1)

| ID | Constraint | Property | Threshold |
|----|-----------|----------|-----------|
| **R0** | S0.a[0] | Autonomy | constraints["S0"]["a"][0] |
| **R1** | S2.b[1] | Behavior | constraints["S2"]["b"][1] |
| **R2** | S0.a[1] + S2.b[1] | Combined | Both R0 and R1 |

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

## Expected Results (900 evaluations)

| Metric | PFES Baseline | PFES+SAMOTA Target |
|--------|---|---|
| **Violations** | 37 ± 8 | 35-40 |
| **R0/R1/R2** | 15 / 3 / 3 | 15-18 / 3-5 / 3-5 |
| **Objectives Covered** | 3/3 | 3/3 |
| **Time** | ~50 min | ~60-70 min |
| **Efficiency** | 0.041 v/e | 0.040-0.045 v/e |

## Usage Examples

### Run PFES+SAMOTA Only

```bash
poetry shell
cd online-step-experiments/ADAS1
python -c "import PFES_SAMOTA; PFES_SAMOTA.run_pfes_samota(budget=900, max_iterations=30)"
```

### Run PFES Baseline Only

```bash
poetry shell
cd online-step-experiments/ADAS1
python -c "import PFES_falsification; PFES_falsification.run_pfes(max_evaluations=900)"
```

### Run Multiple Parallel Experiments

```bash
poetry shell
cd online-step-experiments/ADAS1
python run_comparative_experiments.py --runs 5 --budget 900 --output results_comparison
```

## Troubleshooting

### Python Version Error
```bash
poetry env use python3.11
poetry install
```

### mdp_simulator Not Found
```bash
poetry run pip install CPS-simulator/dist/mdp_simulator-0.1.9-py3-none-any.whl
```

### Clear Cache and Reinstall
```bash
poetry env remove
poetry install --no-cache
```

See [QUICK_START.md](QUICK_START.md) for more details.

## Documentation

- **QUICK_START.md** — Poetry setup and basic commands
- **online-step-experiments/ADAS1/EXPERIMENT_GUIDE.md** — Results interpretation
- **online-step-experiments/ADAS1/PFES_SAMOTA_ALGORITHMS_CORRECTED.md** — Formal algorithms

## Project Information

- **Author**: Helena Moyen
- **Date**: June 2026
- **Framework**: ICSE 2025 Replication Package
- **License**: MIT
- **Dependencies**: Managed via Poetry

## Original Package Structure

This clean repository focuses on the SAMOTA+ADAS1 experiments. The original package included:

- `CPS-simulator`: Python simulator for autonomous driving system
- `online-step-experiments/ADAS1/ADAS2/RR/UAV`: Experiments for 4 benchmarks
- `offline-step-experiments`: MDP verification (PRISM)

This version contains only ADAS1 with the full SAMOTA implementation.

---

**Ready to run on your VM!** 🚀

For setup instructions, see [QUICK_START.md](QUICK_START.md).
