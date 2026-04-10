# Increase the precision of the approximate stationary distribution

`compute_steady_state` iterates $\pi \leftarrow \pi T$ until successive
distributions differ by less than `tolerance` (default $10^{-6}$). Tightening
this threshold produces a more accurate result at the cost of more iterations.

## Default precision

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
array([0.11585355, 0.07317128, 0.16463682, 0.64633834])

```

## Higher precision

Pass a smaller `tolerance` to require a tighter convergence criterion:

```py
>>> ludics.compute_steady_state(transition_matrix, tolerance=1e-12)
array([0.11585366, 0.07317073, 0.16463415, 0.64634146])

```
