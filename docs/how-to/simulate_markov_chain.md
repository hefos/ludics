# Simulate a Markov chain

Use the `simulate_markov_chain` function

```
import ludics
import numpy as np

def example_fitness_function(state):
    return np.array([np.sum(state) - player_action for player_action in state])

initial_state = np.array([1,1,0])
number_of_strategies = 2
seed = 4
warmup = 2
iterations = 8
choice_intensity = 2

ludics.main.simulate_markov_chain(
    initial_state=initial_state,
    number_of_strategies=number_of_strategies,
    fitness_function=example_fitness_function,
    compute_transition_probability=ludics.main.
    compute_introspection_transition_probability,
    iterations=iterations,
    warmup=warmup,
    seed=seed,
    choice_intensity=choice_intensity
)
```

which will return:

```
([(1, 1, 0), (1, 1, 0), (1, 1, 0), (1, 1, 0), (1, 0, 0), (1, 0, 0)],
 Counter({(1, 1, 0): 4, (1, 0, 0): 2}))
```
