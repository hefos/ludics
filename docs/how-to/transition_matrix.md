# Assemble the underlying Markov chain

Use the `generate_transition_matrix` function:

```py
>>> import ludics
>>> import ludics.fitness_functions

>>> r = 1.5
>>> alpha = 5
>>> selection_intensity = 0.2
>>> state_space = ludics.get_state_space(N=2, k=2)
>>> ludics.generate_transition_matrix(
...     state_space=state_space,
...     fitness_function=ludics.fitness_functions.homogeneous_pgg_fitness_function,
...     compute_transition_probability=ludics.compute_moran_transition_probability,
...     selection_intensity=selection_intensity,
...     alpha=alpha,
...     r=r,
... )
array([[1.  , 0.  , 0.  , 0.  ],
       [0.35, 0.5 , 0.  , 0.15],
       [0.35, 0.  , 0.5 , 0.15],
       [0.  , 0.  , 0.  , 1.  ]])

```

Most fitness functions and population dynamics in `ludics` will require
\*\*kwargs, which must be passed into the `generate_transition_matrix` function.

## Apply mutation

Pass the `individual_to_action_mutation_probability` argument:

```py
>>> import ludics
>>> import ludics.fitness_functions
>>> import numpy as np

>>> r = 1.5
>>> alpha = 5
>>> selection_intensity = 0.2
>>> state_space = ludics.get_state_space(N=3, k=2)
>>> mutation_probabilities = np.array([
...     [0.1, 0.2],
...     [0.2, 0.15],
...     [0.15, 0.1],
... ])

>>> ludics.generate_transition_matrix(
...     state_space=state_space,
...     fitness_function=ludics.fitness_functions.homogeneous_pgg_fitness_function,
...     compute_transition_probability=ludics.compute_moran_transition_probability,
...     selection_intensity=selection_intensity,
...     alpha=alpha,
...     r=r,
...     individual_to_action_mutation_probability=mutation_probabilities,
... )
array([[0.85      , 0.03333333, 0.05      , 0.        , 0.06666667,
        0.        , 0.        , 0.        ],
       [0.26428571, 0.5547619 , 0.        , 0.08095238, 0.        ,
        0.1       , 0.        , 0.        ],
       [0.25238095, 0.        , 0.57857143, 0.06904762, 0.        ,
        0.        , 0.1       , 0.        ],
       [0.        , 0.175     , 0.175     , 0.46666667, 0.        ,
        0.        , 0.        , 0.18333333],
       [0.23333333, 0.        , 0.        , 0.        , 0.61666667,
        0.06904762, 0.08095238, 0.        ],
       [0.        , 0.15      , 0.        , 0.        , 0.175     ,
        0.51666667, 0.        , 0.15833333],
       [0.        , 0.        , 0.15      , 0.        , 0.175     ,
        0.        , 0.51666667, 0.15833333],
       [0.        , 0.        , 0.        , 0.03333333, 0.        ,
        0.06666667, 0.05      , 0.85      ]])

```
