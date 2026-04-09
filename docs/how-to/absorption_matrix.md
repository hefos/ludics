# Compute absorption probabilities

The absorption matrix gives the **fixation probabilities** of the chain: entry
$B_{ij}$ is the probability that, starting from transient state $i$, the
population eventually fixes in absorbing state $j$.

## Numeric transition matrix

Use the `approximate_absorption_matrix` function:

```py
>>> import numpy as np
>>> import ludics

>>> transition_matrix = np.array([
...     [1, 0, 0, 0],
...     [0.2, 0.3, 0.2, 0.3],
...     [0.1, 0.4, 0.3, 0.2],
...     [0, 0, 0, 1],
... ])
>>> ludics.approximate_absorption_matrix(transition_matrix)
array([[0.3902439 , 0.6097561 ],
       [0.36585366, 0.63414634]])

```

**Note:** the transition matrix must have at least one absorbing state (a row
with a 1 on the diagonal and 0 elsewhere).

## Symbolic transition matrix

Use the `calculate_absorption_matrix` function for matrices with symbolic
entries:

```py
>>> import ludics
>>> import sympy as sym
>>> import numpy as np

>>> x = sym.Symbol('x')
>>> y = sym.Symbol('y')
>>> transition_matrix = np.array([
...     [1, 0, 0, 0],
...     [x, 1 - x - y, y, 0],
...     [0, y, 1 - x - y, x],
...     [0, 0, 0, 1],
... ])
>>> ludics.calculate_absorption_matrix(transition_matrix)
Matrix([
[x*(x + y)/(x**2 + 2*x*y),       x*y/(x**2 + 2*x*y)],
[      x*y/(x**2 + 2*x*y), x*(x + y)/(x**2 + 2*x*y)]])

```
