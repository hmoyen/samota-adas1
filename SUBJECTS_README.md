# Multi-Subject Support for ICSE 2025 PFES+SAMOTA

This document describes the setup and extension of the clean repository to support multiple test subjects: **ADAS1**, **ADAS2**, and **RR** (Rescue Robot).

## Repository Structure

```
online-step-experiments/
├── ADAS1/                          # Autonomous Driving v1
│   ├── config.py                   # ADAS1 configuration
│   ├── INPUT/                      # MDP models (AutonomousDriving_v1)
│   ├── PFES_falsification.py       # Baseline PFES algorithm
│   ├── PFES_SAMOTA.py              # PFES+SAMOTA hybrid algorithm
│   ├── SAMOTA_ensemble.py          # Per-objective ensemble surrogates
│   ├── RBF.py                      # RBF surrogate model
│   ├── variable_builder.py         # Config-driven variable builder
│   └── utils/
│       ├── helpers.py              # Simulator interface
│       └── constraints_builder.py   # Constraint computation
│
├── ADAS2/                          # Autonomous Driving v2 (extended constraints)
│   ├── config.py                   # ADAS2 configuration
│   ├── INPUT/                      # Symlink to ADAS1/INPUT (same MDP)
│   ├── PFES_falsification.py       # (same code as ADAS1)
│   ├── PFES_SAMOTA.py              # (same code as ADAS1)
│   ├── [shared files...]
│   └── ...
│
└── RR/                             # Rescue Robot v3
    ├── config.py                   # RR configuration (9 variables)
    ├── INPUT/                      # MDP models (RescueRobot_v3)
    ├── PFES_falsification.py       # (same code as ADAS1)
    ├── PFES_SAMOTA.py              # (same code as ADAS1)
    ├── [shared files...]
    └── ...
```

## Key Differences Between Subjects

### ADAS1 (Baseline)
- **Variables**: 6 (car_speed, p_x, p_y, orientation, weather, road_shape)
- **Bounds**: car_speed [5, 50], p_x [0, 10], p_y [0, 10], orientation [-30, 30], weather [0, 2], road_shape [0, 2]
- **Constraints**: 3 (R0, R1, R2)
- **MDP Model**: INPUT/AutonomousDriving_v1
- **Objectives**: 5 (2 from S0.a + 3 from S2.b)

### ADAS2 (Extended)
- **Variables**: 6 (same as ADAS1)
- **Bounds**: same as ADAS1
- **Constraints**: 6 (R0-R5, extended set)
- **MDP Model**: INPUT/AutonomousDriving_v1 (same as ADAS1, symlinked)
- **Objectives**: 5 (same MINIMAL_CONSTRAINTS as ADAS1)

### RR (Rescue Robot)
- **Variables**: 9 (power, cruise_speed, bandwidth, quality, illuminance, smoke_intensity, obstacle_size, obstacle_distance, firm_obstacle)
- **Bounds**: power [0, 100], cruise_speed [0, 5], bandwidth [10, 50], quality [0, 2], illuminance [40, 120000], smoke_intensity [0, 2], obstacle_size [0, 120], obstacle_distance [0, 10], firm_obstacle [0, 1]
- **Constraints**: 6 (R0-R5)
- **MDP Model**: INPUT/RescueRobot_v3
- **Objectives**: 5 (3 from S0.a + 2 from S10.l)

## Code Refactoring: Config-Driven Design

### The Problem
The original ADAS1 code had **hardcoded variable definitions** and bounds:

```python
# Lines 165-166 (art_initial_population)
lb = np.array([5.0, 0.0, 0.0, -30, 0, 0])
ub = np.array([50.0, 10.0, 10.0, 30, 2, 2])

# Lines 196-202 (GSPerObjectiveProblem)
variables = {
    "car_speed": Real(bounds=(5.0, 50.0)),
    "p_x": Real(bounds=(0.0, 10.0)),
    "p_y": Real(bounds=(0.0, 10.0)),
    "orientation": Integer(bounds=(-30, 30)),
    "weather": Integer(bounds=(0, 2)),
    "road_shape": Integer(bounds=(0, 2)),
}

# Parameter extraction was hardcoded to specific variable names
params = np.array([x["car_speed"], x["p_x"], x["p_y"], ...])
```

**This required code changes for each new subject!**

### The Solution: `variable_builder.py`

New module that dynamically builds variables from `config.SS_VARIABLES`:

```python
# Extract bounds dynamically from config
VAR_NAMES = get_variable_names(conf.SS_VARIABLES)
LB, UB = get_bounds_arrays(conf.SS_VARIABLES)

# Build pymoo variables dynamically
variables = build_variables_dict(conf.SS_VARIABLES)

# Convert parameters using variable names
params = np.array([x[name] for name in VAR_NAMES])

# Build test case dict with proper typing
test_case = build_test_case_dict(params, VAR_NAMES, conf.SS_VARIABLES)
```

### Benefits
✅ **Single codebase** for all subjects (ADAS1, ADAS2, RR)
✅ **No hardcoded variable names or bounds**
✅ **Automatic variable type handling** (int vs float)
✅ **Works for any number of variables** (tested with 6 and 9)
✅ **Future subjects** only need a new `config.py` + `INPUT/` folder

## Running Experiments

### ADAS1 Baseline (PFES)
```bash
cd online-step-experiments/ADAS1
python3 PFES_falsification.py --size 30 --niterations 30 --nruns 1 --optalg NSGA3 --logdir out
```

### ADAS1 Hybrid (PFES+SAMOTA)
```bash
cd online-step-experiments/ADAS1
python3 PFES_SAMOTA.py  # Runs with default: budget=1800, max_iterations=30
```

### ADAS2 Baseline
```bash
cd online-step-experiments/ADAS2
python3 PFES_falsification.py --size 30 --niterations 30 --nruns 1 --optalg NSGA3 --logdir out
```

### ADAS2 Hybrid
```bash
cd online-step-experiments/ADAS2
python3 PFES_SAMOTA.py
```

### RR Baseline
```bash
cd online-step-experiments/RR
python3 PFES_falsification.py --size 30 --niterations 30 --nruns 1 --optalg NSGA3 --logdir out
```

### RR Hybrid
```bash
cd online-step-experiments/RR
python3 PFES_SAMOTA.py
```

## Configuration Files

Each subject has a `config.py` that defines:

```python
# Variable definitions with types and ranges
SS_VARIABLES = {
    "var_name": {"domain": float/int, "range": [min, max]},
    ...
}

# MDP model location
MDP_FOLDER = "INPUT/AutonomousDriving_v1"  # or "INPUT/RescueRobot_v3"

# Constraints (R0, R1, R2, ...)
CONSTRAINTS = [...]

# Minimal constraints for optimization objectives
MINIMAL_CONSTRAINTS = {...}

# Other settings
RQ = "rq1"
EXTRA_NAME = "NoHis"
BATCH_SIZE = 5
HISTORY_RETRIES = 10
...
```

## Technical Details

### Variable Name Ordering

Variables are **sorted alphabetically** for consistent ordering across subjects:

```python
# ADAS1/ADAS2 (6 variables)
VAR_NAMES = ['car_speed', 'orientation', 'p_x', 'p_y', 'road_shape', 'weather']

# RR (9 variables)
VAR_NAMES = ['bandwidth', 'cruise_speed', 'firm_obstacle', 'illuminance', 'obstacle_distance', 'obstacle_size', 'power', 'quality', 'smoke_intensity']
```

This ensures consistent parameter array ordering regardless of config key order.

### Variable Type Handling

The builder automatically applies correct types based on `config.SS_VARIABLES`:

```python
# Float variable (e.g., car_speed)
variables["car_speed"] = Real(bounds=(5.0, 50.0))
x["car_speed"]  # Returns float

# Integer variable (e.g., weather)
variables["weather"] = Integer(bounds=(0, 2))
x["weather"]  # Returns int
```

When extracting from numpy arrays, types are preserved:

```python
def build_test_case_dict(params_array, var_names, ss_variables):
    result = {}
    for i, name in enumerate(var_names):
        domain = ss_variables[name]["domain"]
        if domain == int:
            result[name] = int(params_array[i])
        else:
            result[name] = float(params_array[i])
    return result
```

### INPUT Folder Management

- **ADAS1**: Full MDP model in `INPUT/AutonomousDriving_v1/`
- **ADAS2**: Symlink to ADAS1 (`INPUT -> ../ADAS1/INPUT`) since they use the same MDP
- **RR**: Full MDP model in `INPUT/RescueRobot_v3/`

## Testing

To verify setup works correctly:

```bash
# Test ADAS1 imports
cd online-step-experiments/ADAS1
python3 -c "import config; import variable_builder; print(f'✓ {len(config.SS_VARIABLES)} variables'); print(variable_builder.get_variable_names(config.SS_VARIABLES))"

# Test ADAS2 imports
cd ../ADAS2
python3 -c "import config; import variable_builder; print(f'✓ {len(config.SS_VARIABLES)} variables')"

# Test RR imports
cd ../RR
python3 -c "import config; import variable_builder; print(f'✓ {len(config.SS_VARIABLES)} variables'); print(variable_builder.get_variable_names(config.SS_VARIABLES))"
```

## Expected Output Format

All subjects produce consistent CSV outputs:

```
score_NSGA3_1.csv     # Best fitness scores (V0, V1, V2, V3, V4)
reqs_NSGA3_1.csv      # Violations per constraint (R0, R1, ..., conjunction)
X_all_evaluations_NSGA3_0.csv       # All test cases (columns: variable names)
F_all_evaluations_NSGA3_0.csv       # All fitness scores (V0, V1, ...)
Reqs_all_evaluations_NSGA3_0.csv    # All requirement satisfactions (R0, R1, ...)
```

## Thesis Extension Points

This config-driven architecture enables:

1. **New subjects**: Add `config.py` + `INPUT/` folder, no code changes
2. **Variable count scaling**: Tested with 6 (ADAS1/2) and 9 (RR) variables
3. **Constraint count scaling**: ADAS1 (3) and ADAS2/RR (6) constraints
4. **Algorithm variants**: Can implement new algorithms using same config system
5. **Performance comparisons**: Fair testing across subjects with identical code

## Summary of Changes

| File | Changes | Status |
|------|---------|--------|
| `PFES_SAMOTA.py` | Config-driven variable bounds & definitions | ✅ Updated for all subjects |
| `PFES_falsification.py` | Config-driven variable bounds & definitions | ✅ Updated for all subjects |
| `variable_builder.py` | New module for dynamic variable extraction | ✅ Created, shared |
| `config.py` (ADAS2) | Copied from original repo | ✅ In place |
| `config.py` (RR) | Copied from original repo | ✅ In place |
| `INPUT/` (ADAS1/2/RR) | Copied/symlinked as appropriate | ✅ In place |
| `utils/` (ADAS1/2/RR) | Copied from ADAS1 | ✅ In place |
| `SAMOTA_ensemble.py`, `RBF.py` | Copied, no changes needed | ✅ In place |

