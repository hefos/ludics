# apply_mutation_probability

```
ludics.main.apply_mutation_probability(
    source, target, individual_to_action_mutation_probability, transition_probability)
```

Takes an existing transition probability and applies mutation

### Parameters:

- `source`: _numpy.array_ - the current state
- `target`: _numpy.array_ - the state being transitioned to
- `individual_to_action_mutation_probability`: _numpy.array_ - probability that
  player (row) mutates to action type (column)
- `transition_probability`: _float_ - the transition probability without
  mutation applied

### Returns:

- _float_ - the transition probability with mutation
