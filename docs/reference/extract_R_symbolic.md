# extract_R_symbolic

```
ludics.extract_R_symbolic(transition_matrix)
```

Extracts the submatrix of transitions from transitive states to absorbing
states for a transition matrix with symbolic values

### Parameters

- `transition_matrix`: _numpy.array_ - A square matrix of numeric transition
  probabilities. May contain symbolic values

### Returns

- _numpy.array_ - the submatrix of transitions from transitive states to
  absorbing states
