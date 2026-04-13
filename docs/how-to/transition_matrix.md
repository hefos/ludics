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

## Consider heterogeneous population dynamics

Pass `compute_transition_probability` as a _numpy.array_ of population dynamics

```py
>>> import ludics.main
>>> import numpy as np
>>> import ludics.fitness_functions

>>> source = np.array([1, 1, 1])
>>> target = np.array([1, 1, 0])

>>> population_dynamic_array = np.array([
...     ludics.compute_moran_transition_probability,
...     ludics.compute_fermi_transition_probability,
...     ludics.compute_introspection_transition_probability
... ])

>>> N = 3
>>> number_of_strategies = 2
>>> state_space = ludics.get_state_space(N=N, k=number_of_strategies)

>>> r = 2
>>> contribution_vector = np.array([1, 2, 3])
>>> choice_intensity = 1
>>> selection_intensity = 0.1

>>> ludics.generate_transition_matrix(
...     state_space=state_space,
...     fitness_function=ludics.fitness_functions.heterogeneous_contribution_pgg_fitness_function,
...     compute_transition_probability=population_dynamic_array,
...     r=r,
...     contribution_vector=contribution_vector,
...     choice_intensity=choice_intensity,
...     selection_intensity=selection_intensity,
...     number_of_strategies=number_of_strategies
... )
array([[0.91035286, 0.08964714, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.24368619, 0.6575004 , 0.        , 0.00790431, 0.        ,
        0.09090909, 0.        , 0.        ],
       [0.29359903, 0.        , 0.51953161, 0.08964714, 0.        ,
        0.        , 0.09722222, 0.        ],
       [0.        , 0.14679951, 0.24368619, 0.40316509, 0.        ,
        0.        , 0.        , 0.20634921],
       [0.22939068, 0.        , 0.        , 0.        , 0.63613861,
        0.08964714, 0.04482357, 0.        ],
       [0.        , 0.12418301, 0.        , 0.        , 0.24368619,
        0.57940292, 0.        , 0.05272788],
       [0.        , 0.        , 0.12121212, 0.        , 0.14679951,
        0.        , 0.64234123, 0.08964714],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.24368619, 0.75631381]])

```
