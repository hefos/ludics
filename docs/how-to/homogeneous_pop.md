# Model a Homogeneous Population

As a default, `ludics` models heterogeneous populations. However, many models
use a homogeneous population instead. We show the mathematics behind this in our
explanation of modelling evolutionary games as Markov chains.

The probability functions, fitness functions, and `generate_state_space`, are
built to handle a heterogeneous population. However,
`generate_transition_matrix` will accept a homogeneous population, provided one
writes a suitable probability function and fitness function.

Imagine we wish to model a 3 player homogeneous public goods game using the
Moran process. Then we can use the following model:
```py
>>> import ludics
>>> import numpy as np

>>> def homogeneous_pgg(number_of_cooperators, alpha, r, **kwargs):
...     sum_contributions = r * number_of_cooperators * alpha
...     contributor_return = sum_contributions - alpha
...     return np.array([contributor_return, sum_contributions])

>>> def homogeneous_moran_process(source, target, fitness_function, N, epsilon, **kwargs):
...     source = source[0]
...     target = target[0]
...     difference = source - target
...     if np.abs(difference) > 1:
...         return 0
...     elif np.abs(difference) == 0:
...         return None
...     fitness_C, fitness_D = fitness_function(number_of_cooperators=source, **kwargs)
...     if difference == 1:
...         return (1 / N) * ((N - source) * (1 + epsilon * fitness_D)) / (
...             ((N - source) * (1 + epsilon * fitness_D))
...             + (source * (1 + epsilon * fitness_C))
...         )
...     elif difference == -1:
...         return (1 / N) * (
...             (source * (1 + epsilon * fitness_C))
...             / (
...                 ((N - source) * (1 + epsilon * fitness_D))
...                 + (source * (1 + epsilon * fitness_C))
...             )
...         )

>>> state_space = np.array([[0], [1], [2], [3]])
>>> alpha = 2
>>> r = 2
>>> epsilon = 0.1

>>> ludics.generate_transition_matrix(
...     state_space=state_space,
...     fitness_function=homogeneous_pgg,
...     compute_transition_probability=homogeneous_moran_process,
...     alpha=alpha,
...     r=r,
...     N=3,
...     epsilon=epsilon,
... )
array([[1.        , 0.        , 0.        , 0.        ],
       [0.23333333, 0.66666667, 0.1       , 0.        ],
       [0.        , 0.12      , 0.66666667, 0.21333333],
       [0.        , 0.        , 0.        , 1.        ]])

```

Note that while the above works for `generate_transition_matrix`, the function
`simulate_markov_chain` does not support being passed a homogeneous state space.
Also note that the state space *must* remain in the form
`np.array([[a],[b],...])`, or else the `generate_transition_matrix` function
will fail. Mutation is also not currently supported for this, but may be added
to the custom `compute_transition_probability` function.