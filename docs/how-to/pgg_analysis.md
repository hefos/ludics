# Analyse a Public Goods Game

A complete analysis of a homogeneous Public Goods Game (PGG) using Fermi
imitation dynamics: building the transition matrix and computing fixation
probabilities.

See [The public goods game](../explanation/public_goods_game.md) for the
payoff formula and [What is a population dynamic?](../explanation/population_dynamics.md)
for the definition of Fermi imitation dynamics.

In a homogeneous PGG with $N$ players, each contributor pays a cost `alpha`
and the pooled contributions are multiplied by `r` and shared equally. When
$r < N$, defection is individually rational, and the chain has two absorbing
states: all-defect $[0,0,0]$ and all-contribute $[1,1,1]$.

## Set up the state space

```py
>>> import ludics
>>> import ludics.fitness_functions
>>> import numpy as np

>>> state_space = ludics.get_state_space(N=N, k=2)
>>> state_space
array([[0, 0, 0],
       [0, 0, 1],
       [0, 1, 0],
       [0, 1, 1],
       [1, 0, 0],
       [1, 0, 1],
       [1, 1, 0],
       [1, 1, 1]])

```

## Build the transition matrix

```py
>>> N = 3
>>> r = 1.5
>>> alpha = 1.0
>>> choice_intensity = 1.0

>>> transition_matrix = ludics.generate_transition_matrix(
...     state_space=state_space,
...     fitness_function=ludics.fitness_functions.homogeneous_pgg_fitness_function,
...     compute_transition_probability=ludics.compute_fermi_transition_probability,
...     choice_intensity=choice_intensity,
...     r=r,
...     alpha=alpha,
... )

```

## Compute fixation probabilities

The rows of the absorption matrix correspond to the six transient (mixed)
states; the columns correspond to all-defect $[0,0,0]$ and all-contribute
$[1,1,1]$.

```py
>>> ludics.approximate_absorption_matrix(transition_matrix)
array([[0.90996943, 0.09003057],
       [0.90996943, 0.09003057],
       [0.66524096, 0.33475904],
       [0.90996943, 0.09003057],
       [0.66524096, 0.33475904],
       [0.66524096, 0.33475904]])

```

With $r=1.5 < N=3$, states with a single contributor fix on defection with
probability ~91%, and states with two contributors still favour defection (~67%).
