# MDP - SIMULATOR

## Install Requirements:

pip install -r requirements.txt

## Example run:

python main.py -i "Examples/MDP_CONFIG" -t "STEPS" -n 50 -d "INFO"

## CLI ARGUMENTS

- `-ep`, `--Equilibrium-Policy`: the algorithm used to recompute the policy every time there is an equilibrium
  constraint not satisfied.
    - RANDOM (default)
    - VALUE_ITERATION
- `-n`, `--N`: the number of steps to execute.
    - 500 (default)
- `-b`, `--Batch-Size`: how often to recompute the posterior values (HDI and Expected).
    - 5 (default)
- `-t`, `--Termination`: specify the termination condition.
    - STEPS (default)
- `-i`, `--Mpd`: specify the folder where all the configuration files can be found.
    - Examples/MDP_CONFIG (default)
- `-d`, `--Debug`: set the debug level.
    - STAT (default)
    - NONE
    - ERROR
    - INFO
    - DEBUG
- `-ip`, `--Initial-Policy`: specify a custom initial policy (default is `None`).
    - It's a string in the form (x0,x1,x2,x3,x4,...) which defines for each state its best action to take: the states
      are written in the same order they are defined inside the config file.
- `-o`, `--Output-Csv`: specify an output folder in which save a csv and all the plots of the execution (default
  is `None`).
    - The output folder will be created under `./OUTPUT/<folder_name>` and will contain a csv file with the
      timestamp of the execution and another folder (named after the timestamp of the execution) containing all the
      relative plots.
- `-r`, `--Random-Mdp`: specify the number of states, actions and priors to create a random mdp (default is `None`).
    - It's a string in the form `n_state, n_action, n_prior` containing the integer values used as parameters to build a
      random MDP.

## MDP CONFIGURATION

### Folder content

There are 5 different files tha can be pu under the mdp configuration folder:

- states.mdp (required)
- priors.mdp
- changes.mdp
- ss_variables.mdp
- ss_changes.mdp

#### States (states.mdp)

Each line of the states file contains the definition of a single state:
`state_id; state_label; state_type; actions`

- `state_id`: it's a string which contains an id to identify the state
- `state_label`: it's a "friendly" name used only for better understandability
- `state_type`: it specifies if a state is `CONTROLLABLE` or `OBSERVABLE`
- `actions`: it's a sequence of tuples containing three values: `action_id, probability, outcome;`
    - action_id: it's the id of the action
    - probability: once this specific action is selected, it's the probability that we select the outcome as the next
      state.
    - outcome: it's the state_id of the next state that can be reached using this action

Examples:

- `s0; start; CONTROLLABLE; a, 0.5, s1; a, 0.25, s2; a, 0.25, s3; b, 0.5, s4; b, 0.5, s5;`
- `s10; state 10; OBSERVABLE; i, 1.0, s7;`

#### Priors (priors.mdp)

Each line of the priors file contains the definition of a single prior:
`state_id; action_id; dirichlet_parameters; eq_constraints;`

- the combination `state_id` and `action_id` specifies the region of the prior
- `dirichlet_parameters` are in the form of:  `Value, Value[, Value]*`
- `eq_constraints` are in the form: `Value-Value, Value-Value[, Value-Value]*`

Both the _**dirichlet_parameters**_ and the _**eq_constraints**_ supplied here are assigned to the outcomes in the same
order they were specified in the state file:

Example:

- State definition _s0; start ;CONTROLLABLE; a, 0.5, s5; a, 0.25, s3; a, 0.25, s9;_
- Prior definition `s0; a; 10, 20, 30; 0.1-0.9, 0.1-0.6, 0.2-0.8;`
    - _**10**_ and _**0.1-0.9**_ are assigned to _**s5**_
    - _**20**_ and _**0.1-0.6**_ are assigned to _**s3**_
    - _**30**_ and _**0.2-0.8**_ are assigned to _**s9**_

#### Changes (changes.mdp)

Inside this file there are listed all the probability changes that are going to take place during the execution. These
changes are only relative to uncertain regions.

Each single change section is composed by a label that specifies the index of the step during which the changes have to
take place and by all the actual changes.

- `CHANGES step_index:`: The step index is an integer.
- `state_id; action_id; Value, Value[, Value]*;`: with the combination state_id and action_id we identify the uncertain
  region. All the subsequent values represent the probabilities of all the possible outcomes (they are assigned in the
  same order they were specified in the state config)

#### Semantic Space Variables (ss_variables.mdp)

After specifying all the variables of the semantic space, inside this file we can set a custom starting point. This
assignment is not mandatory because if a variable of the semantic space doesn't have a custom starting value, it will
assume the default one.

#### Semantic Space Variables Changes (ss_changes.mdp)

Inside this file there are listed all the value changes of the semantic space variables that are going to take place
during the execution.

Each single change section is composed by a label that specifies the index of the step during which the changes have to
take place and by all the actual changes.

- `CHANGES step_index:`: The step index is an integer.
- `semantic_space_variable_name, value;`: The value that we are assigning to the variable has to belong to the domain
  of that specific variable (int or float)

### Semantic Space Configuration

Apart from specifying `ss_variables.mdp` and `ss_changes.mdp` (that are not required), in order to define the semantic
space it's required to fill the folder `SS_VARIABES` with one or multiple files containing the definition of each
variable (these don't have to have specific naming convention as long as they end with `.py`).

The syntax of these files is python and each variable is defined as a dictionary.

```python
import numpy as np  # used for manipulating the probability array


# The semantic space will manipulate all the specified probabilities through methods like this one
def compute_probability(old_probability: np.ndarray, variable_value) -> np.ndarray:
    computed_probability = old_probability
    if variable_value % 2 == 0:
        computed_probability = np.flip(computed_probability)
    return computed_probability


SemanticSpaceVariable = [
    {
        "Name": "Test_Elem_1",  # the name of the variable
        "Type": "ENV",  # The type [ENV or SYS]
        "Domain": int,  # The domain [int or float]
        "Range": [0, 5],  # [lower_bound, upper_bound]
        "Default": 1,  # the default starting value
        "Combinations": [
            # here it can be specified the combination state_id and action_id that the variable is supposed to manipulate
            # together with the method that it's supposed to be used to exec the manipulation.
            {
                "StateId": "s0",
                "ActionId": "a",
                "Method": compute_probability
            }
        ],
    }
]
```

N.B.

- Each method has to be specified with the following signature otherwise an error will be risen:
    - `method_name(old_probability: np.ndarray, variable_value) -> np.ndarray:`
- `SemanticSpaceVariable` is an array that can contain multiple variables.
- `Combinations` is an array that can contain multiple manipulation of different actions
- Different semantic space variables that manipulate the same combination (state_id and action_id) are applied
  sequentially.

## Tests

All the tests are saved inside the _tests_ folder and are implemented using _unittest_.

To run them all and build a coverage report:

```shell
coverage run -m unittest discover
```

To print the report directly inside the shell:

```shell
python -m coverage report
```

To create an HTML report of the coverage:

```shell
coverage html
```

## Build and Install instructions

1. Install poetry: `pip install poetry`
2. Open a terminal window inside the root folder of the project
3. Build the module: `poetry build`
4. Install the newly created module with: `pip install ".\dist\mdp_simulator-X.Y.Z-py3-none-any.whl"`
   (`X.Y.Z` depends on the version you're building)












