# build_hybrid_population_dynamic

```
ludics.build_hybrid_population_dynamic(dynamic_array)
```

Builds a hybrid population dynamic where each player updates based on a
different rule.

### Parameters:

- `dynamic_array`: _numpy.array_ - an array of population dynamic functions

### Returns:

- _func_: points to the correct population dynamic to be used when
  transitioning between states
