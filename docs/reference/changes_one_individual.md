# changes_one_individual

```
ludics.changes_one_individual(compute_transition_probability)
```

Marks a population dynamic in which one individual changes type per step

A dynamic carrying this mark asserts that it returns 0 whenever the source and
the target differ at more than one position. `generate_transition_matrix` then
visits only the neighbourhood of each state rather than every ordered pair of
states, which is the difference between work in the square of the number of
states and work in the number of states times the size of a neighbourhood.

A dynamic which is not marked is given every state as a candidate target, since
we cannot know which of them it reaches. We therefore ask each dynamic to make
the claim rather than assuming it, so that a dynamic in which the whole
population is replaced at once, such as a Wright-Fisher process, is handled
correctly by a caller who has never heard of this mark. The only cost of
forgetting the mark is the time the longer loop takes.

Every dynamic in `ludics` carries the mark, as does any dynamic built by
`build_hybrid_population_dynamic`.

### Parameters:

- `compute_transition_probability`: _func_ - the population dynamic to mark

### Returns:

- _func_ - the same population dynamic, marked

### Example

```py
>>> import ludics
>>> import numpy as np

>>> @ludics.changes_one_individual
... def compute_lazy_probability(source, target, fitness_function, **kwargs):
...     different_indices = ludics.get_different_indices(source=source, target=target)
...     if len(different_indices) > 1:
...         return 0
...     if len(different_indices) == 0:
...         return None
...     return 0.25

>>> compute_lazy_probability.changes_one_individual
True

```
