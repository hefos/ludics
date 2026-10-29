# get_different_indices

```
ludics.get_different_indices(source, target)
```

Returns the positions at which two states differ. The population dynamics in
`ludics` use this to find which individual has changed strategy, since a dynamic
which changes one individual at a time has exactly one such position.

### Parameters:

- `source`: _numpy.array_ - the starting state
- `target`: _numpy.array_ - the state which the source transitions to

### Returns:

- _numpy.array_ - the positions at which the two states differ
