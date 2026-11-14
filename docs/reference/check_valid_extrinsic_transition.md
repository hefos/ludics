# check_valid_extrinsic_transition

```
ludics.check_valid_extrinsic_transition(source, target)
```

Returns whether a transition is reachable under an extrinsic dynamic, in which
an individual may only adopt a strategy which some other member of the
population already uses. A move to a strategy nobody holds cannot be imitated,
so it is not a valid extrinsic transition.

### Parameters:

- `source`: _numpy.array_ - the starting state
- `target`: _numpy.array_ - the state which the source transitions to

### Returns:

- _bool_ - whether the transition is reachable under an extrinsic dynamic
