# heterogeneous_contribution_pgg_fitness_function

```
ludics.fitness_functions.fully_heterogeneous_pgg_fitness_function(state, r, contribution_vector, **kwargs)
```

Calculates each player's payoff in a public goods game with heterogeneous contributions in a given
state

### Parameters:

- `state`: _numpy.array_ - the state of each player's action type
- `r_vector`: _numpy.array_ - array of each player's rate of return
- `contribution_vector`: _numpy.array_ - the amount that each player contributes

### Returns:

- _numpy.array_: each player's fitness in the state when playing a public goods
  game with heterogeneous contributions and returns
