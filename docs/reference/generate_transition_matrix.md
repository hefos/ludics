# generate_transition_matrix

```
ludics.generate_transition_matrix(state_space, fitness_function, compute_transition_probability, individual_to_action_mutation_probability=None,
return_sparse_array=False, **kwargs)
```

Builds the transition matrix for a Markov chain

### Parameters:

- `state_space`: _numpy.array_ - an array of possible states
- `fitness_function`: _func_ - takes a numpy.array and returns an
  `array of floats with the same shape
- `compute_transition_probability`: _func_ - takes two states and returns the
  probability of transitioning between them. May pass a _numpy.array_ of
  functions instead, if wishing to consider heterogeneous population dynamics.
- `individual_to_action_mutation_probability`: _numpy.array_ - probability that
  player (row) mutates to action type (column)
- `return_sparse_array`: _bool_ - whether to keep only the entries which are
  not zero, rather than writing them into an array of zeros. Available for a
  dynamic marked with [changes_one_individual](changes_one_individual.md),
  which is the case where most of the matrix is zero

### Returns:

- _numpy.array_ or _scipy.sparse.csr_array_ - a transition matrix

### Raises:

- _ValueError_ - when a sparse array is asked for and the dynamic is not
  marked with `changes_one_individual`, since it reaches every state and so
  its matrix has no zeros to leave out
