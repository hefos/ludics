# Choosing a fitness function

There are 3 fitness functions provided in the `ludics.fitness_functions`
package. Each fitness function takes a state and returns a value for each
player in the state

## Public Goods Games

For a public goods game, the state provided must contain only 0s (defectors)
and 1s (contributors).

### Homogeneous public goods game

This fitness function requires parameters `r` and `alpha`, and provides the
fitness of players in the state playing according to a public goods game.

Use the following to calculate the fitness of each individual in the state:

```
import ludics.fitness_functions
import numpy as np

state = np.array([1,0,0,1])
r = 2
alpha = 3

ludics.fitness_functions.homogeneous_pgg_fitness_function(
    state=state,
    alpha=alpha,
    r=r
)
```

which will return

```
array([0,3,3,0])
```

### Heterogeneous public goods game

The heterogeneous public goods game requires parameters `r` and
`contribution_vector`, and provides the fitness of players in the state
according to a heterogeneous public goods game where player $i$ contributes
`contribution_vector[i]`. It works as follows:

```
import ludics.fitness_functions
import numpy as np

state = np.array([1,1,0,1])
r = 2
contribution_vector = np.array([1,2,3,4])

ludics.fitness_functions.heterogeneous_contribution_pgg_fitness_function(
    state=state,
    r=r,
    contribution_vector=contribution_vector
)
```

which returns

```
array([1.5, 1.5, 4.5, 1.5])
```

## Symbolic fitness functions

`ludics.fitness_functions` provides a general symbolic fitness function for the
state space:

```
np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
```

It takes no additional arguments

```
import ludics.fitness_functions
import numpy as np

state = np.array([0,1])
ludics.fitness_functions.general_four_state_fitness_function(state)
```

which returns:

```
array([f_1(b), f_2(b)])
```

## Defining your own fitness function

A fitness function takes a state and any other parameters and returns an array
of the fitness of each player.

For example:

```
def sample_fitness_function(state, test_parameter, **kwargs):
    return np.array([test_parameter * player_type for player_type in state])

state = np.array([1,2,3])
test_parameter = 3

sample_fitness_function(state, test_parameter)
```

which will return

```
array([3,6,9])
```

**NOTE**: A fitness function _must_ take \*\*kwargs arguments in order to be
passed to the `generate_transition_matrix` function
