# generate_transition_matrix

```
ludics.main.generate_transition_matrix(state_space, fitness_function, compute_transition_probability, individual_to_action_mutation_probability=None,
**kwargs)
```

Builds the transition matrix for a Markov chain

### Parameters:

- `state_space`: _numpy.array_ - an array of possible states
- `fitness_function`: _func_ - takes a numpy.array and returns an
  `array of floats with the same shape
- `compute_transition_probability`: _func_ - takes two states and returns the
  probability of transitioning between them
- `individual_to_action_mutation_probability`: _numpy.array_ - probability that
  player (row) mutates to action type (column)

### Returns:

- _numpy.array_ - a transition matrix
