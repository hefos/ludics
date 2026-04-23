# compute_fermi_transition_probability

```
ludics.compute_fermi_transition_probability(source, target, fitness_function, choice_intensity, **kwargs)
```

Calculates the probability of transitioning from `source` to `target` with
Fermi imitation dynamics

### Parameters:

- `source`: _numpy.array_ - the current state
- `target`: _numpy.array_ - the state being transitioned to
- `fitness_function`: _func_ - takes a numpy.array and returns an
  `array of floats with the same shape
- `choice_intensity`: _float_ - the choice intensity of the process

### Returns:

- _float_ if `source` != `target`, else _None_
