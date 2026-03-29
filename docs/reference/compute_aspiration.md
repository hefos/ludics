# compute_aspiration_transition_probability

```
ludics.main.compute_aspiration_transition_probability(source, target, fitness_function, choice_intensity, aspiration_vector, **kwargs)
```

Calculates the probability of transitioning from `source` to `target` with
aspiration dynamics

### Parameters:

- `source`: _numpy.array_ - the current state
- `target`: _numpy.array_ - the state being transitioned to
- `fitness_function`: _func_ - takes a numpy.array and returns an
  `array of floats with the same shape
- `choice_intensity`: _float_ - the choice intensity of the process
- `aspiration_vector`: _numpy.array_ - the aspiration of each player in the
  state

### Returns:

- _float_ if `source` != `target`, else _None_
