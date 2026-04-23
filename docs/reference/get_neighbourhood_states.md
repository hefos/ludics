# get_neighbourhood_states

```
ludics.get_neighbourhood_states(state, number_of_strategies)
```

Finds all states that differ from a state in exactly one position

### Parameters:

- `state`: _numpy.array_ - the focal state
- `number_of_strategies`: _int_ - how many strategies the players in the state
  can play

### Returns:

- _np.array_ - an array of states adjacent to the focal state
