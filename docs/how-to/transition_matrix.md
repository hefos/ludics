# Assemble the underlying Markov chain

Use the `generate_transition_matrix` function:

```py
>>> import ludics
>>> import ludics.fitness_functions
>>> import numpy as np

>>> r = 1.5
>>> alpha = 5
>>> selection_intensity = 0.2
>>> state_space = ludics.get_state_space(N=2, k=2)
>>> ludics.generate_transition_matrix(
...     state_space=state_space,
...     fitness_function=ludics.fitness_functions.public_goods_game_fitness_function,
...     compute_transition_probability=ludics.compute_moran_transition_probability,
...     selection_intensity=selection_intensity,
...     fitness_map=ludics.linear_fitness_map,
...     alpha=alpha,
...     r=r,
... )
array([[1.        , 0.        , 0.        , 0.        ],
       [0.36904762, 0.5       , 0.        , 0.13095238],
       [0.36904762, 0.        , 0.5       , 0.13095238],
       [0.        , 0.        , 0.        , 1.        ]])

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
...     fitness_function=ludics.fitness_functions.public_goods_game_fitness_function,
...     compute_transition_probability=ludics.compute_moran_transition_probability,
...     selection_intensity=selection_intensity,
...     fitness_map=ludics.linear_fitness_map,
...     alpha=alpha,
...     r=r,
...     individual_to_action_mutation_probability=mutation_probabilities,
... )
array([[0.85      , 0.03333333, 0.05      , 0.        , 0.06666667,
        0.        , 0.        , 0.        ],
       [0.27413793, 0.56264368, 0.        , 0.07241379, 0.        ,
        0.0908046 , 0.        , 0.        ],
       [0.26091954, 0.        , 0.58908046, 0.0591954 , 0.        ,
        0.        , 0.0908046 , 0.        ],
       [0.        , 0.18137255, 0.18235294, 0.45980392, 0.        ,
        0.        , 0.        , 0.17647059],
       [0.24252874, 0.        , 0.        , 0.        , 0.62586207,
        0.0591954 , 0.07241379, 0.        ],
       [0.        , 0.15686275, 0.        , 0.        , 0.18235294,
        0.50882353, 0.        , 0.15196078],
       [0.        , 0.        , 0.15686275, 0.        , 0.18137255,
        0.        , 0.51078431, 0.15098039],
       [0.        , 0.        , 0.        , 0.03333333, 0.        ,
        0.06666667, 0.05      , 0.85      ]])

```
