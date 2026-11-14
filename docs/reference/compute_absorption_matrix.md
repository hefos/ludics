# compute_absorption_matrix

```
ludics.compute_absorption_matrix(transition_matrix)
```

Computes the absorption matrix for a numeric transition matrix

$B$ solves $(I - Q)B = R$, where $Q$ holds the transitions between transient
states and $R$ the transitions from transient states to absorbing ones. The
system is factorised the way the chain was given: densely for an ordinary
array, which is the faster of the two, and sparsely for a sparse array, which
is what lets a large chain be held at all. For 6558 transient states the dense
factorisation takes 1.4 seconds and 387MB, and the sparse one 3.5 seconds and
under 6MB.

### Parameters:

- `transition_matrix`: _numpy.array_ or _scipy.sparse array_ - a square matrix
  of numeric transition probabilities

### Returns:

- _numpy.array_ - the absorption matrix of the system, giving the probability of
  being absorbed in each absorbing state (column) from each transient state
  (row)
