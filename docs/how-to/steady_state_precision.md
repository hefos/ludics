# Control how the stationary distribution is found

`compute_steady_state` solves for the stationary distribution directly. It
writes $\pi T = \pi$ as $(T^{\mathsf{T}} - I)\pi^{\mathsf{T}} = 0$, replaces the
last row with the condition that the entries sum to one, and hands the system to
scipy. The result is exact to within floating point, so there is no precision to
tune in the ordinary case.

```py
>>> import ludics
>>> import numpy as np

>>> transition_matrix = np.array([
...     [0.3, 0.3, 0.3, 0.1],
...     [0.0, 0.3, 0.2, 0.5],
...     [0.1, 0.1, 0.7, 0.1],
...     [0.1, 0.0, 0.0, 0.9],
... ])

>>> ludics.compute_steady_state(transition_matrix)
array([0.11585366, 0.07317073, 0.16463415, 0.64634146])

```

## When the chain has more than one stationary distribution

A chain with two absorbing states has a family of stationary distributions
rather than one, and which of them the chain settles on depends on where it
starts. The direct system is singular in that case, so `compute_steady_state`
iterates $\pi \leftarrow \pi T$ from `initial_dist` instead, and `tolerance`
decides when to stop. Passing an `initial_dist` always takes this route, since a
caller who names a starting distribution is asking about that starting point.

```py
>>> transition_matrix = np.array([
...     [1.00, 0.00, 0.00, 0.00],
...     [0.25, 0.25, 0.25, 0.25],
...     [0.25, 0.25, 0.25, 0.25],
...     [0.00, 0.00, 0.00, 1.00],
... ])

>>> ludics.compute_steady_state(
...     transition_matrix, initial_dist=np.array([1, 0, 0, 0])
... )
array([1, 0, 0, 0])

>>> ludics.compute_steady_state(
...     transition_matrix, initial_dist=np.array([0, 0, 0, 1])
... )
array([0, 0, 0, 1])

```

We note that `tolerance` bounds the change between successive steps, not the
distance to the answer. A chain which mixes slowly can sit well inside the
tolerance while still being some way from its stationary distribution, which is
one reason to prefer the direct solve wherever it applies.
