# public_goods_game_fitness_function

```
ludics.fitness_functions.public_goods_game_fitness_function(state, r, alpha, **kwargs)
```

Calculates each player's payoff in a public goods game with heterogeneous
contributions and returns in a given state

### Parameters:

- `state`: _numpy.array_ - the state of each player's action type
- `r`: _float_ or _numpy.array_ - the ratio by which the contributions are
  multiplied. Heterogeneous if provided a _float_, heterogeneous if passed a _numpy.array_
- `contribution_vector`: _float_ or _numpy.array_ - the amount that each player
  contributes. Heterogeneous if provided a _float_, heterogeneous if passed a _numpy.array_

### Returns:

- _numpy.array_: each player's fitness in the state when playing a public goods
  game with heterogeneous contributions and returns
