# Choose a fitness function

There are 3 fitness functions provided in the `ludics.fitness_functions`
package. Each fitness function takes a state and returns a fitness value for
each player in the state.

## Public Goods Games

For a public goods game, the state must contain only 0s (defectors) and 1s
(contributors).

### Homogeneous public goods game

All players contribute the same amount `alpha`. The total pool is multiplied
by `r` and split equally.

```py
>>> import ludics.fitness_functions
>>> import numpy as np

>>> state = np.array([1, 0, 0, 1])
>>> r = 2
>>> alpha = 3

>>> ludics.fitness_functions.homogeneous_pgg_fitness_function(
...     state=state,
...     alpha=alpha,
...     r=r,
... )
array([0., 3., 3., 0.])

```

### Heterogeneous public goods game

Players contribute different amounts given by `contribution_vector`, where
player $i$ contributes `contribution_vector[i]`.

```py
>>> import ludics.fitness_functions
>>> import numpy as np

>>> state = np.array([1, 1, 0, 1])
>>> r = 2
>>> contribution_vector = np.array([1, 2, 3, 4])

>>> ludics.fitness_functions.heterogeneous_contribution_pgg_fitness_function(
...     state=state,
...     r=r,
...     contribution_vector=contribution_vector,
... )
array([ 2.5,  1.5,  3.5, -0.5])

```

## Symbolic fitness functions

`ludics.fitness_functions` provides a general symbolic fitness function for
2-player populations, covering all four states:

```
np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
```

It takes no additional arguments:

```py
>>> import ludics.fitness_functions
>>> import numpy as np

>>> state = np.array([0, 1])
>>> ludics.fitness_functions.general_four_state_fitness_function(state)
array([f_1(b), f_2(b)], dtype=object)

```
