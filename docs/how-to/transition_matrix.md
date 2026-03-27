# Generate a transition matrix

Given a state space, a fitness function, and a population dynamic, our
transition matrix is given by:

```
import ludics
import ludics.fitness_functions

r=1.5
alpha=5
selection_intensity=0.2
state_space = ludics.get_state_space(N=2, k=2)
ludics.generate_transition_matrix(
    state_space=state_space,
    fitness_function=ludics.fitness_functions.homogeneous_pgg_fitness_function,
    compute_transition_probability=ludics.compute_moran_transition_probability
    selection_intensity=selection_intensity,
    alpha=alpha,
    r=r
)
```

which gives the following:

```
array([[1.  , 0.  , 0.  , 0.  ],
       [0.35, 0.5 , 0.  , 0.15],
       [0.35, 0.  , 0.5 , 0.15],
       [0.  , 0.  , 0.  , 1.  ]])
```

Most fitness functions and population dynamics in `ludics` will require
\*\*kwargs, which must be passed into the `generate_transition_matrix` function.
