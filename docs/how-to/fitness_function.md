# Choosing a fitness function

There are 3 fitness functions provided in the `ludics.fitness_functions`
package. Each fitness function takes a state and returns a value for each
player in the state

## Public Goods Games

For a public goods game, the state provided must contain only 0s (defectors)
and 1s (contributors). To learn more about public goods games, we recommend the
following paper:

Social dilemmas among unequals - Oliver P. Hauser, Christian Hilbe, Krishnendu Chatterjee & Martin A. Nowak

### Homogeneous public goods game

This fitness function requires parameters `r` and `alpha`, and provides the
fitness of players in the state playing according to a public goods game.

We can then
use the following to calculate the fitness of each individual in the state:

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
`contribution_vector[i]`.
