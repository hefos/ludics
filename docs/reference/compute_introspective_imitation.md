# compute_introspective_imitation_transition_probability

```
ludics.compute_introspective_imitation_transition_probability(source, target,
fitness_function, choice_intensity, selection_intensity,
fitness_map=linear_fitness_map, **kwargs)
```

Returns the probability of moving from one state to another under introspective
imitation dynamics, in which an individual compares its own fitness with that of
a peer it has chosen, and adopts the peer's strategy with a probability given by
the Fermi function.

The dynamic changes one individual at a time, so it is marked with
[changes_one_individual](changes_one_individual.md). A source and target which
differ at more than one position have probability zero, and a source and target
which are equal return `None`, since the diagonal is set from the row sums.

### Parameters:

- `source`: _numpy.array_ - the starting state
- `target`: _numpy.array_ - the state which the source transitions to
- `fitness_function`: _func_ - maps a state to a numpy.array of fitnesses
- `choice_intensity`: _numpy.array_ - entry $ik$ is the rationality with which
  player $i$ considers strategy $k$. Shape must be $(N, K)$
- `selection_intensity`: _numpy.array_ - entry $ij$ is the selection intensity
  when player $i$ copies player $j$. Shape must be $(N, N)$. Entries may be
  floats or sympy symbols
- `fitness_map`: _func_ - a positive mapping applied to the fitness, linear by
  default

### Returns:

- _float_ or _None_ - the transition probability from source to target, or
  `None` when the states are equal
