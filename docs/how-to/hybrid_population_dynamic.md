# Build a hybrid population dynamic

Use the `build_hybrid_population_dynamic` function:

```py
>>> import ludics
>>> import numpy as np

>>> population_dynamic_array = np.array([
...     ludics.compute_moran_transition_probability,
...     ludics.compute_fermi_transition_probability,
...     ludics.compute_introspection_transition_probability
... ])

>>> ludics.build_hybrid_population_dynamic(population_dynamic_array)
<function ...>

```

This can be passed directly into `generate_transition_matrix`

```py
>>> import ludics
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
>>> hybrid_population_dynamic = ludics.build_hybrid_population_dynamic(population_dynamic_array)
>>> r = 2
>>> contribution_vector = np.array([1, 2, 3])
>>> choice_intensity = 1
>>> selection_intensity = 0.1

>>> ludics.generate_transition_matrix(
...     state_space=state_space,
...     fitness_function=ludics.fitness_functions.heterogeneous_contribution_pgg_fitness_function,
...     compute_transition_probability=hybrid_population_dynamic,
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
