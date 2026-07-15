# exponential_fitness_map

```
ludics.exponential_fitness_map(fitness, selection_intensity, **kwargs)
```

Returns fitness mapped by the function $e^{\epsilon\pi_i(\mathbf{a})}$

### Parameters

fitness: _numpy.array_ - the fitness of the players in the state. May also take a
float for a single player's fitness. Currently, this function only supports a
numerical fitness values

selection_intensity: _numpy.array_ - the selection intensity of the system

### Returns

_numpy.array_ - the scaled fitness of the state