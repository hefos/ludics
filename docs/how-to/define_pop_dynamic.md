# Define an evolutionary dynamic

A population dynamic can be defined by creating 
a function satisfying the following:

- Takes two states (`source` and `target`) and a `fitness_function`.
- Returns `None` if the states are the same.
- Returns a value in $(0,1)$ otherwise.

This can then be passed to `generate_transition_matrix` as `compute_transition_probability`

Most dynamics change one individual at a time, so that a state can only reach
the states which differ from it at a single position. Such a dynamic returns 0
whenever the two states differ at more than one position, and we mark it with
`changes_one_individual` so that `generate_transition_matrix` visits only the
neighbourhood of each state rather than every ordered pair of states. The mark
is an assertion about the dynamic and not a change to the chain, so the matrix
is the same either way.

A dynamic which does not change one individual at a time, such as a
Wright-Fisher process in which the whole population is replaced at once, is
left unmarked and every ordered pair of states is given to it. We note that an
unmarked dynamic is always built correctly, so the only cost of forgetting the
mark is the time the longer loop takes.

An example of this would be the definition of the `peer pressure` population
dynamic, where players accept a new strategy based on the total fitness of that
strategy in the population. We would define this using the following function:

```py
>>> import ludics
>>> import numpy as np

>>> @ludics.changes_one_individual
... def compute_peer_pressure_probability(source, target, beta, fitness_function, **kwargs):
...    different_indices = np.where(source != target)
...    if len(different_indices[0]) > 1:
...        return 0
...    if len(different_indices[0]) == 0:
...        return None
...    fitness = fitness_function(source, **kwargs)
...    fitness_current = np.sum(fitness[source == source[different_indices]])
...    fitness_other = np.sum(fitness[source == target[different_indices]])
...    delta = fitness_current - fitness_other
...    return ludics.fermi_imitation_function(delta=delta, choice_intensity=beta)/len(source)

>>> def trivial_fitness_function(state):
...     return np.array([1 for _ in state])

>>> source = np.array([1,0,1,1])
>>> target = np.array([0,0,1,1])
>>> compute_peer_pressure_probability(
... source=source,
... target=target,
... beta=0.5,
... fitness_function=trivial_fitness_function
... )
np.float64(0.06723535534249878)

```

We can then pass this function to compute_transition_matrix 
in order to model a Markov chain under this population dynamic.

```py
>>> import ludics
>>> import numpy as np

>>> @ludics.changes_one_individual
... def compute_peer_pressure_probability(source, target, beta, fitness_function, **kwargs):
...    different_indices = np.where(source != target)
...    if len(different_indices[0]) > 1:
...        return 0
...    if len(different_indices[0]) == 0:
...        return None
...    fitness = fitness_function(source, **kwargs)
...    fitness_current = np.sum(fitness[source == source[different_indices]])
...    fitness_other = np.sum(fitness[source == target[different_indices]])
...    delta = fitness_current - fitness_other
...    return ludics.fermi_imitation_function(delta=delta, choice_intensity=beta)/len(source)

>>> def trivial_fitness_function(state):
...     return np.array([1 for _ in state])

>>> state_space = ludics.get_state_space(N=2, k=2)
>>> ludics.generate_transition_matrix(
... state_space=state_space,
... compute_transition_probability=compute_peer_pressure_probability,
... fitness_function=trivial_fitness_function,
... beta=1
... )
array([[0.88079708, 0.05960146, 0.05960146, 0.        ],
       [0.25      , 0.5       , 0.        , 0.25      ],
       [0.25      , 0.        , 0.5       , 0.25      ],
       [0.        , 0.05960146, 0.05960146, 0.88079708]])

```