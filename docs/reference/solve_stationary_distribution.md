# solve_stationary_distribution

```
ludics.solve_stationary_distribution(transition_matrix, residual_tolerance=10**-8,
density_threshold=0.25)
```

Solves for the unique stationary distribution of a transition matrix, or returns
`None` when the chain does not have exactly one.

Writing $\pi T = \pi$ as $(T^{\mathsf{T}} - I)\pi^{\mathsf{T}} = 0$ and replacing
the last row with the condition that the entries sum to one gives a linear
system whose solution is the stationary distribution. The system is factorised
sparsely when few of its entries are non zero and densely when most of them are.

A chain with two or more absorbing states has a family of stationary
distributions rather than one, which makes the system singular. There is then no
single answer to return, so `None` is returned and the caller may iterate
instead. This is what [compute_steady_state](compute_steady_state.md) does.

### Parameters:

- `transition_matrix`: _numpy.array_ or _scipy.sparse array_ - a transition
  matrix. A sparse array is read without being converted
- `residual_tolerance`: _float_ - how far $\pi T$ may sit from $\pi$ for the
  solution to be accepted
- `density_threshold`: _float_ - the proportion of non zero entries at which the
  system stops being factorised sparsely and is factorised densely instead

### Returns:

- _numpy.array_ or _None_ - the stationary distribution, or `None` when the
  chain does not have exactly one
