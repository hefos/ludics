# Choose an evolutionary dynamic

A population dynamic defines how the population transitions between states.
There are four dynamics in `ludics`, falling into two categories:

**Extrinsic dynamics** (Moran, Fermi): players update
by comparing themselves to others. The resulting Markov chain is **absorbing**
and the key quantity is the fixation probability.

**Intrinsic dynamics** (introspection, aspiration): players update based on
their own payoff alone. The resulting chain is **ergodic** and the key quantity
is the stationary distribution.

See [What is a population dynamic?](../explanation/population_dynamics.md) for
the mathematical definitions of each dynamic.

## The Moran process

Use the `compute_moran_transition_probability` function. Takes a
`selection_intensity` argument in addition to standard parameters.

```py
>>> import ludics
>>> import numpy as np

>>> def example_fitness_function(state):
...    return np.array([np.sum(state) - player_action for player_action in state])

>>> source = np.array([0,1,2])
>>> target = np.array([1,1,2])
>>> selection_intensity = np.full(shape=(3,3), fill_value=0.5)

>>> ludics.compute_moran_transition_probability(
... source=source,
... target=target,
... selection_intensity=selection_intensity,
... fitness_function=example_fitness_function
... )
np.float64(0.1111111111111111)

```

**Note:** `selection_intensity` must satisfy the following equation for all
players $i$:

$1 -$ `selection_intensity` $+$ `selection_intensity` $\cdot$ `fitness_function(state)[i]` $\gt 0$

`selection_intensity` is passed as a numpy.array. If player $i$ changes, then
player $j$'s fitness is scaled by entry $(i,j)$ during selection.

## Fermi imitation dynamics

Use the `compute_fermi_transition_probability` function. Takes a
`choice_intensity` argument in addition to standard parameters.

```py
>>> import ludics
>>> import numpy as np

>>> def example_fitness_function(state):
...     return np.array([np.sum(state) - player_action for player_action in state])

>>> source = np.array([0, 1, 2])
>>> target = np.array([1, 1, 2])
>>> choice_intensity = np.full(shape=(3,3), fill_value=0.5)

>>> ludics.compute_fermi_transition_probability(
...     source=source,
...     target=target,
...     choice_intensity=choice_intensity,
...     fitness_function=example_fitness_function,
... )
0.0629234447996909

```

`choice_intensity` controls the rationality of the Fermi function. It is passed
as a `numpy.array` with shape (N,N). Like `selection_intensity`, player $i$
considers player $j$'s fitness with rationality equal to entry $(i,j)$.

## Introspection dynamics

Use the `compute_introspection_transition_probability` function. Takes
`choice_intensity` and `number_of_strategies` arguments in addition to
standard parameters.

```py
>>> import ludics
>>> import numpy as np

>>> def example_fitness_function(state):
...     return np.array([np.sum(state) - player_action for player_action in state])

>>> source = np.array([0, 1, 2])
>>> target = np.array([1, 1, 2])
>>> choice_intensity = np.full(shape=(3,3), fill_value=0.5)

>>> ludics.compute_introspection_transition_probability(
...     source=source,
...     target=target,
...     choice_intensity=choice_intensity,
...     fitness_function=example_fitness_function,
...     number_of_strategies=3,
... )
0.0833333333333333

```

In introspection dynamics, `choice_intensity` is given as an (N,K) numpy.array,
where $K$ is the number of strategies. Entry $(i,k)$ is the rationality used by
player $i$ when considering strategy $k$. For a homogeneous value, `np.full` can
be used.

## Aspiration dynamics

Use the `compute_aspiration_transition_probability` function. Takes
`choice_intensity` and `aspiration_vector` arguments in addition to standard
parameters. State space must include exactly two actions.

```py
>>> import ludics
>>> import numpy as np

>>> def example_fitness_function(state):
...     return np.array([np.sum(state) - player_action for player_action in state])

>>> source = np.array([0, 1, 1])
>>> target = np.array([1, 1, 1])
>>> choice_intensity = np.full(shape=(3,2), fill_value=0.5)
>>> aspiration_vector = np.array([3, 2, 1])

>>> ludics.compute_aspiration_transition_probability(
...     source=source,
...     target=target,
...     choice_intensity=choice_intensity,
...     fitness_function=example_fitness_function,
...     aspiration_vector=aspiration_vector,
... )
0.207486443733952

```
