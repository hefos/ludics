# compute_moran_transition_probability

```
ludics.main.compute_moran_transition_probablity(source, target, fitness_function, selection_intensity, **kwargs)
```

Calculates the probability of transitioning from `source` to `target` in the
Moran process

### Parameters:

- `source`: _numpy.array_ - the current state
- `target`: _numpy.array_ - the state being transitioned to
- `fitness_function`: _func_ - takes a numpy.array and returns an
  `array of floats with the same shape
- `selection_intensity`: _float_ - the selection intensity of the process

### Returns:

- _float_ if `source` != `target`, else _None_
