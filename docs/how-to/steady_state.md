# Calculate a steady state

## Numeric transition matrices

Use the `approximate_steady_state` function

```
import ludics
import numpy as np

transition_matrix = np.array([
    [0.3,0.3,0.3,0.1],
    [0,0.3,0.2,0.5],
    [0.1,0.1,0.7,0.1],
    [0.1,0,0,0.9]
])

ludics.approximate_steady_state(transition_matrix)
```

which will return:

```
array([0.11585355, 0.07317128, 0.16463682, 0.64633834])
```

## Symbolic transition matrices

Use the `calculate_steady_state` function

```
import ludics
import sympy as sym
import numpy as np

x = sym.Symbol('x')
y = sym.Symbol('y')
z = sym.Symbol('z')
transition_matrix = np.array([
    [1-y, 0, y, 0],
    [y, 1-y, 0, 0],
    [0, 0, 1-x, x],
    [0, x, 0, 1-x]
])

ludics.calculate_steady_state(transition_matrix)
```

which will give:

```
array([x/(2*(x + y)), x/(2*(x + y)), y/(2*(x + y)), y/(2*(x + y))])
```

The steady states calculated are always returned as numpy.array objects. If
a symbolic transition matrix is explicity defined, it is important to ensure an
entry has a $(1-\sum_{E \in row} E)$ term to ensure the existence of an
eigenvector for all values of your symbolic terms.
