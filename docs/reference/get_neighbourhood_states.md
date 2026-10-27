# get_neighbourhood_states

```
ludics.get_neighbourhood_states(state, number_of_strategies=None, action_set=None)
```

Finds all states that differ from a state in exactly one position

The actions are given either as a count or as the set of actions themselves.
A state space whose labels are not 0 to k - 1 needs the second form, since the
actions it uses cannot be recovered from a count.

### Parameters:

- `state`: _numpy.array_ - the focal state
- `number_of_strategies`: _int_ - how many strategies the players in the state
  can play, taken to be 0 to `number_of_strategies` - 1. Used when `action_set`
  is not given
- `action_set`: _numpy.array_ - the actions the players in the state can play

### Returns:

- _np.array_ - an array of states adjacent to the focal state
