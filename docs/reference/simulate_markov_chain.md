# simulate_markov_chain

```
ludics.simulate_markov_chain(initial_state, number_of_strategies, fitness_function, compute_transition_probability, seed, individual_to_action_mutation_probability=None, warmup=0, iterations=10000,
**kwargs)
```

Simulates a Markov chain across a specified number of iterations

### Parameters:

- `initial_state`: _numpy.array_ - the state that the Markov chain begins in
- `number_of_strategies`: _int_ - the number of strategies which players can
  play
- `fitness_function`: _func_ - takes a state and returns a _numpy.array_ of
  floats with the same shape
- `compute_transition_probability`: _func_ - takes two states and returns the
  probability of transitioning between them.
- `seed`: _int_ - the seed for _numpy.random.seed_
- `individual_to_action_mutation_probability`: _numpy.array_ - the probability
  that a player (row) mutates to an action (column) when chosen. Set to 0 for
  all mutations by default.
- `warmup`: _int_ - the number of iterations to take place before recording the state
  distribution
- `iterations`: _int_ - the number of iterations to simulate

### Returns:

- _tuple_ containing:
  1. _numpy.array_ - the states as they were reached over time in the
     simulation
  2. _dict_ - state : number of times the state was visited in the simulation
