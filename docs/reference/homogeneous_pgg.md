# homogeneous_pgg_fitness_function

```
ludics.fitness_functions.homogeneous_pgg_fitness_function(state, alpha, r, **kwargs)
```

Calculates each player's payoff in a homogeneous public goods game in a given
state

### Parameters:

- `state`: _numpy.array_ - the state of each player's action type
- `alpha`: _float_ - the amount that each player in the game contributes
- `r`: _float_ - the ratio by which the contributions are multiplied

### Returns:

- _numpy.array_: each player's fitness in the state when playing a homogeneous public goods game
