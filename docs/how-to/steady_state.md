# Compute the stationary distribution of a Markov chain

Both `approximate_steady_state` and `calculate_steady_state` work on any
ergodic (irreducible, aperiodic) Markov chain — not just ones built from
evolutionary dynamics. You can pass any transition matrix directly.

## Numeric transition matrices

Use `approximate_steady_state`:

```py
>>> import ludics
>>> import numpy as np

>>> transition_matrix = np.array([
...     [0.3, 0.3, 0.3, 0.1],
...     [0.0, 0.3, 0.2, 0.5],
...     [0.1, 0.1, 0.7, 0.1],
...     [0.1, 0.0, 0.0, 0.9],
... ])

>>> ludics.approximate_steady_state(transition_matrix)
array([0.11585355, 0.07317128, 0.16463682, 0.64633834])

```

## Symbolic transition matrices

Use `calculate_steady_state` when entries contain symbolic parameters:

```py
>>> import ludics
>>> import sympy as sym
>>> import numpy as np

>>> x = sym.Symbol('x')
>>> y = sym.Symbol('y')
>>> transition_matrix = np.array([
...     [1 - y, 0, y, 0],
...     [y, 1 - y, 0, 0],
...     [0, 0, 1 - x, x],
...     [0, x, 0, 1 - x],
... ])

>>> ludics.calculate_steady_state(transition_matrix)
array([x/(2*(x + y)), x/(2*(x + y)), y/(2*(x + y)), y/(2*(x + y))],
      dtype=object)

```

The result is always a `numpy.array`. For symbolic matrices, at least one entry
per row must be expressed as $(1 - \sum_{\text{other entries}})$ to ensure the
transition matrix has a valid eigenvector for all parameter values.
