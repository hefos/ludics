# Simulate dynamics for large populations

Use the `simulate_markov_chain` function to approximate the long-run behaviour
of the chain via forward simulation. This is useful when the state space is too
large for exact computation of the transition or absorption matrix.

```py
>>> import ludics
>>> import numpy as np

>>> def example_fitness_function(state):
...     return np.array([np.sum(state) - player_action for player_action in state])

>>> initial_state = np.array([1, 1, 0])
>>> number_of_strategies = 2
>>> seed = 4
>>> warmup = 2
>>> iterations = 8
>>> choice_intensity = 2
>>> dynamic = ludics.compute_introspection_transition_probability

>>> visited_states, visit_counts = ludics.simulate_markov_chain(
...     initial_state=initial_state,
...     number_of_strategies=number_of_strategies,
...     fitness_function=example_fitness_function,
...     compute_transition_probability=dynamic,
...     iterations=iterations,
...     warmup=warmup,
...     seed=seed,
...     choice_intensity=choice_intensity,
... )

>>> visited_states
[(1, 1, 0), (1, 1, 0), (1, 1, 0), (1, 1, 0), (1, 0, 0), (1, 0, 0)]

>>> visit_counts
Counter({(1, 1, 0): 4, (1, 0, 0): 2})

```

**NOTE**: Only 6 states are recorded because of :code:`warmup=2` which ensures
the first 2 states are ignored.
