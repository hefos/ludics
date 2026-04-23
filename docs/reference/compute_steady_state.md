# compute_steady_state

```
ludics.compute_steady_state(transition_matrix, tolerance=10**-6, initial_dist=None)
```

Approximates the steady state of a Markov chain with a numeric transition matrix by iterating through state
distributions

### Parameters:

- `transition_matrix`: _numpy.array_ - a square matrix of numeric transition
  probabilities
- `tolerance`: _float_ - how close a state distribution must be to the
  previous distribution to be accepted
- `initial_dist`: _numpy.array_ - the first state distribution to be considered

### Returns:

- _numpy.array_ - the steady state of the Markov chain
