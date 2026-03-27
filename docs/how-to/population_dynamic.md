# Choose a population dynamic

A population dynamic returns the probability of transitioning between two
states given a fitness function. There are five population dynamics included in
`ludics`. They are as follows:

## The Moran process

Takes a `selection_intensity` argument in adition to standard parameters

```
import ludics
import numpy as np

def example_fitness_function(state):
    return np.array([np.sum(state) - player_action for player_action in state])

source = np.array([0,1,2])
target = np.array([1,1,2])
selection_intensity = 0.5

ludics.compute_moran_transition_probability(
    source=source,
    target=target,
    selection_intensity=selection_intensity,
    fitness_function=example_fitness_function
)
```

which will return:

```
0.1111111111111111
```

**Note:** `selection_intensity` must satisfy the following equation for all
players $i$:

$1 +$ `selection_intensity` $\cdot$ `fitness_function(state)[i]` $ > 0$

## Fermi imitation dynamics

Takes a `choice_intensity` argument in addition to standard parameters

```
import ludics
import numpy as np

def example_fitness_function(state):
    return np.array([np.sum(state) - player_action for player_action in state])

source = np.array([0,1,2])
target = np.array([1,1,2])
choice_intensity = 0.5

ludics.compute_fermi_transition_probability(
    source=source,
    target=target,
    choice_intensity=choice_intensity,
    fitness_function=example_fitness_function
)
```

which will return:

```
0.0629234447996909
```

## Introspection dynamics

Takes `choice_intensity` and `number_of_strategies` arguments in addition to standard parameters

```
import ludics
import numpy as np

def example_fitness_function(state):
    return np.array([np.sum(state) - player_action for player_action in state])

source = np.array([0,1,2])
target = np.array([1,1,2])
choice_intensity = 0.5

ludics.compute_introspection_transition_probability(
    source=source,
    target=target,
    choice_intensity=choice_intensity,
    fitness_function=example_fitness_function,
    number_of_strategies=3
)
```

which will return:

```
0.0833333333333333
```

## Aspiration dynamics

Takes `choice_intensity` and `aspiration_vector` arguments in addition to
standard parameters. State space must include exactly two actions.

```
import ludics
import numpy as np

def example_fitness_function(state):
    return np.array([np.sum(state) - player_action for player_action in state])

source = np.array([0,1,1])
target = np.array([1,1,1])
choice_intensity = 0.5
aspiration_vector = np.array([3,2,1])

ludics.compute_aspiration_transition_probability(
    source=source,
    target=target,
    choice_intensity=choice_intensity,
    fitness_function=example_fitness_function,
    aspiration_vector=aspiration_vector
)
```

which will return:

```
0.207486443733952
```

## Introspective imitation dynamics

Takes `choice_intensity` and `selection_intensity` parameters in addition to
standard parameters.

```
import ludics
import numpy as np

def example_fitness_function(state):
    return np.array([np.sum(state) - player_action for player_action in state])

source = np.array([0,1,1])
target = np.array([1,1,1])
choice_intensity = 0.5
selection_intensity = 0.5

ludics.compute_imitation_introspection_transition_probability(
    source=source,
    target=target,
    choice_intensity=choice_intensity,
    selection_intensity=selection_intensity,
    fitness_function=example_fitness_function,
)
```

which will return:

```
0.1
```

**Note:** `selection_intensity` must satisfy the following equation for all
players $i$:

$1 +$ `selection_intensity` $\cdot$ `fitness_function(state)[i]` $ > 0$

## Create your own population dynamic

A population must take two states and return the probability of transitioning
between them. It does not require a fitness function by definition.

```
import numpy as np

def example_population_dynamic(source, target, a):
    """
    Two players selected. The first accepts the strategy of the second with
    probability $\frac{1}{a}$
    """

    N = len(source)
    return (1 / N) * (1 / (N-1)) * (1 / a)

source = np.array([1,2,3])
target = np.array([2,2,3])
a = 2

example_population_dynamic(
    source=source,
    target=target,
    a=a)
```

which will return:

```
0.08333333333333333
```
