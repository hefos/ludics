# Define a fitness function

A fitness function takes a state and any additional parameters and returns an
array of fitness values, one per player.

```py
>>> import numpy as np

>>> def sample_fitness_function(state, test_parameter, **kwargs):
...     return np.array([test_parameter * player_type for player_type in state])

>>> state = np.array([1, 2, 3])
>>> test_parameter = 3

>>> sample_fitness_function(state, test_parameter)
array([3, 6, 9])

```

**NOTE**: A fitness function _must_ accept `**kwargs` in order to be passed to
the `generate_transition_matrix` function.
