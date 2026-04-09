# Choose a population dynamic

A population dynamic returns the probability of transitioning between two
states given a fitness function. There are five population dynamics included in
`ludics`. They are as follows:

## The Moran process

Use the `compute_moran_transition_probability` function. Takes a `selection_intensity` argument in adition to standard parameters

```py
>>> import ludics.main
>>> import numpy as np

>>> def example_fitness_function(state):
...    return np.array([np.sum(state) - player_action for player_action in state])

>>> source = np.array([0,1,2])
>>> target = np.array([1,1,2])
>>> selection_intensity = 0.5

>>> ludics.main.compute_moran_transition_probability(
... source=source,
... target=target,
... selection_intensity=selection_intensity,
... fitness_function=example_fitness_function
... )
np.float64(0.1111111111111111)

```

**Note:** `selection_intensity` must satisfy the following equation for all
players $i$:

$1 +$ `selection_intensity` $\cdot$ `fitness_function(state)[i]` $\gt 0$

## Fermi imitation dynamics

Use the `compute_fermi_transition_probability` function. Takes a `choice_intensity` argument in addition to standard parameters

```py
>>> import ludics.main
>>> import numpy as np

>>> def example_fitness_function(state):
...     return np.array([np.sum(state) - player_action for player_action in state])

>>> source = np.array([0,1,2])
>>> target = np.array([1,1,2])
>>> choice_intensity = 0.5

>>> ludics.main.compute_fermi_transition_probability(
... source=source,
... target=target,
... choice_intensity=choice_intensity,
... fitness_function=example_fitness_function
... )
0.0629234447996909

```

## Introspection dynamics

Use the `compute_introspection_transition_probability` function. Takes `choice_intensity` and `number_of_strategies` arguments in addition to standard parameters

```py
>>> import ludics.main
>>> import numpy as np

>>> def example_fitness_function(state):
...     return np.array([np.sum(state) - player_action for player_action in state])

>>> source = np.array([0,1,2])
>>> target = np.array([1,1,2])
>>> choice_intensity = 0.5

>>> ludics.main.compute_introspection_transition_probability(
... source=source,
... target=target,
... choice_intensity=choice_intensity,
... fitness_function=example_fitness_function,
... number_of_strategies=3
... )
0.0833333333333333

```

## Aspiration dynamics

Use the `compute_aspiration_transition_probability` function. Takes `choice_intensity` and `aspiration_vector` arguments in addition to
standard parameters. State space must include exactly two actions.

```py
>>> import ludics.main
>>> import numpy as np

>>> def example_fitness_function(state):
...    return np.array([np.sum(state) - player_action for player_action in state])

>>> source = np.array([0,1,1])
>>> target = np.array([1,1,1])
>>> choice_intensity = 0.5
>>> aspiration_vector = np.array([3,2,1])

>>> ludics.main.compute_aspiration_transition_probability(
... source=source,
... target=target,
... choice_intensity=choice_intensity,
... fitness_function=example_fitness_function,
... aspiration_vector=aspiration_vector
... )
0.207486443733952

```

## Introspective imitation dynamics

Use the `compute_imitation_introspection_transition_probability` function. Takes `choice_intensity` and `selection_intensity` parameters in addition to
standard parameters.

```py
>>> import ludics.main
>>> import numpy as np

>>> def example_fitness_function(state):
...     return np.array([np.sum(state) - player_action for player_action in state])

>>> source = np.array([0,1,1])
>>> target = np.array([1,1,1])
>>> choice_intensity = 0.5
>>> selection_intensity = 0.5

>>> ludics.main.compute_imitation_introspection_transition_probability(
... source=source,
... target=target,
... choice_intensity=choice_intensity,
... selection_intensity=selection_intensity,
... fitness_function=example_fitness_function,
... )
0.100000000000000

```

**Note:** `selection_intensity` must satisfy the following equation for all
players $i$:

$1 +$ `selection_intensity` $\cdot$ `fitness_function(state)[i]` $ > 0$

## Create your own population dynamic

A population must take two states and return the probability of transitioning
between them. It does not require a fitness function by definition.

```py
>>> import numpy as np

>>> def example_population_dynamic(source, target, a):
...     """
...     Two players selected. The first accepts the strategy of the second
...     with probability $\frac{1}{a}$
...     """
...     N = len(source)
...     return (1 / N) * (1 / (N-1)) * (1 / a)

>>> source = np.array([1,2,3])
>>> target = np.array([2,2,3])
>>> a = 2

>>> example_population_dynamic(
... source=source,
... target=target,
... a=a)
0.08333333333333333

```
