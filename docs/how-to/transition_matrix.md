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

## Build the matrix sparsely for a large population

A dynamic which changes one individual at a time can only reach the states which
differ from the source at a single position, so most of the matrix is zero. The
array those entries are written into is the square of the number of states
whatever the dynamic does with it: a population of 6561 states takes 344MB
densely and under 2MB sparsely, and one of 19683 states takes 3.1GB, which is
where a dense build stops rather than where it gets slow.

Pass `return_sparse_array=True` to keep only the entries which are not zero:

```py
>>> import ludics
>>> import ludics.fitness_functions
>>> import numpy as np

>>> state_space = ludics.get_state_space(N=3, k=2)
>>> arguments = {
...     "state_space": state_space,
...     "fitness_function": (
...         ludics.fitness_functions.public_goods_game_fitness_function
...     ),
...     "compute_transition_probability": (
...         ludics.compute_moran_transition_probability
...     ),
...     "selection_intensity": 0.2,
...     "fitness_map": ludics.linear_fitness_map,
...     "alpha": 5,
...     "r": 1.5,
... }

>>> transition_matrix = ludics.generate_transition_matrix(
...     **arguments, return_sparse_array=True
... )
>>> transition_matrix.shape
(8, 8)
>>> transition_matrix.nnz
26

```

It is the same chain, so it gives the same stationary distribution:

```py
>>> dense_matrix = ludics.generate_transition_matrix(**arguments)
>>> bool(np.allclose(transition_matrix.toarray(), dense_matrix))
True

>>> bool(np.allclose(
...     ludics.compute_steady_state(transition_matrix),
...     ludics.compute_steady_state(dense_matrix),
... ))
True

```

We note two limits. A dynamic which is not marked with `changes_one_individual`
reaches every state, so its matrix has no zeros to leave out and asking for a
sparse array raises a `ValueError`. A symbolic chain comes back as an array of
objects, since a sparse array holds floats.

A sparse array is read by `compute_steady_state`,
`solve_stationary_distribution` and `compute_absorption_matrix`, so both the
stationary distribution and the absorption probabilities of a large chain can
be found from one. `simulate_markov_chain` still expects an ordinary array.
