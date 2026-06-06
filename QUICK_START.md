# Quick Start Guide - Poetry Setup

## Installation

### 1. Install Poetry

If you don't have Poetry installed:

```bash
# macOS / Linux
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
export PATH="$HOME/.local/bin:$PATH"

# Or use your package manager (Ubuntu/Debian)
sudo apt-get install python3-poetry
```

### 2. Clone/Setup Repository

```bash
cd /path/to/icse2025-samota-adas1
```

### 3. Install Dependencies with Poetry

```bash
# Install project dependencies (creates virtual environment automatically)
poetry install

# This installs:
# - numpy, scipy, scikit-learn, pandas, matplotlib
# - pymoo, click, hdbscan
# - mdp_simulator (if available locally)
```

## Running Experiments

All commands should be run with `poetry run` to use the managed virtual environment.

### Run Comparative Experiments (PFES vs PFES+SAMOTA)

```bash
# Run 5 iterations with 900 evaluations per run
poetry run python online-step-experiments/ADAS1/run_comparative_experiments.py \
  --runs 5 \
  --budget 900 \
  --output results_comparison
```

### Run PFES+SAMOTA Only

```bash
poetry run python -c "
import sys
sys.path.insert(0, 'online-step-experiments/ADAS1')
import PFES_SAMOTA
result = PFES_SAMOTA.run_pfes_samota(budget=900, max_iterations=30)
print(f'Violations: {result[\"violations\"]}')
"
```

### Run PFES Baseline Only

```bash
poetry run python -c "
import sys
sys.path.insert(0, 'online-step-experiments/ADAS1')
import PFES_falsification
result = PFES_falsification.run_pfes(max_evaluations=900)
print(f'Violations: {result[\"violations\"]}')
"
```

## Virtual Environment Management

### Activate Virtual Environment

```bash
# Activate Poetry's virtual environment
poetry shell

# Now you can run Python directly (no need for 'poetry run')
python online-step-experiments/ADAS1/run_comparative_experiments.py --runs 5 --budget 900
```

### View Environment Info

```bash
# Show poetry virtual environment path
poetry env info

# Show installed packages
poetry show

# Show dependencies tree
poetry show --tree
```

### Update Dependencies

```bash
# Update to latest compatible versions
poetry update

# Update specific package
poetry update numpy
```

## Troubleshooting

### Python Version Error

If you get "Python version not supported":

```bash
# Poetry requires Python 3.10-3.11 for this project
# Check your Python version
python --version

# Use specific Python version
poetry env use python3.11
poetry install
```

### mdp_simulator Installation

The mdp_simulator wheel is included in `CPS-simulator/dist/`:

```bash
# Poetry will try to install from local wheel if configured
# If not, manually add to virtual environment:
poetry run pip install CPS-simulator/dist/mdp_simulator-0.1.9-py3-none-any.whl
```

### Clear Cache and Reinstall

```bash
# Remove virtual environment
poetry env remove

# Reinstall everything
poetry install --no-cache
```

## Project Structure

```
icse2025-samota-adas1/
├── pyproject.toml                     # Poetry configuration
├── poetry.lock                        # Lock file (dependency versions)
├── CPS-simulator/
│   ├── dist/
│   │   └── mdp_simulator-0.1.9-py3-none-any.whl  # Simulator
│   └── README.md
└── online-step-experiments/
    └── ADAS1/
        ├── PFES_SAMOTA.py             # Main implementation
        ├── PFES_falsification.py      # Baseline
        ├── run_comparative_experiments.py  # Stats framework
        ├── config.py                  # Configuration
        ├── utils/                     # Utilities
        └── INPUT/                     # Simulator data
```

## Expected Output

After running experiments:

```
results_comparison/
├── summary.csv                    # Overall comparison
├── pfes_runs.csv                 # PFES per-run results
├── pfes_samota_runs.csv          # PFES+SAMOTA per-run results
└── efficiency_analysis.txt       # Statistical analysis
```

## Next Steps

1. **Install**: `poetry install`
2. **Activate**: `poetry shell`
3. **Run**: `python online-step-experiments/ADAS1/run_comparative_experiments.py --runs 5 --budget 900 --output results`
4. **Analyze**: Check `results/efficiency_analysis.txt`

See `online-step-experiments/ADAS1/EXPERIMENT_GUIDE.md` for detailed results interpretation.
