# Choose or Define a Fitness Map

When using population dynamics with fitness proportional selection, for example
the Moran process, payoffs must be mapped to strictly positive values. There are
a number of methods of doing this, with the most common included in `ludics`.

These maps are functions $f(\pi, \epsilon)$, which are non-negative, have strictly positive sums, and
satisfy:

$$\frac{\partial f}{\partial \pi} > 0$$
$$\text{sign}(\frac{\partial f}{\partial \epsilon}) = \text{sign}(\pi) $$
 in the payoffs $\pi$, and either increasing or decreasing in the

### Linear Fitness Map

This map corresponds to the map $f(\pi, \epsilon) = 1 - \epsilon + \epsilon\pi$.
It is commonly used even when payoffs are all positive, and is useful for
systems where the payoffs have a small absolute value. In this case, $\epsilon$
must be chosen such that all payoffs are strictly positive. Setting $\epsilon =
1$ gives a fitness equal to payoff. In `ludics`, this is implemented using
`ludics.linear_fitness_map`:

```py
>>> import numpy as np
>>> import ludics

>>> fitness = np.array([1,2,3])
>>> selection_intensity = 0.2

>>> ludics.linear_fitness_map(fitness=fitness, selection_intensity=selection_intensity)
array([1. , 1.2, 1.4])

```

### Exponential Fitness Map

The exponential fitness map corresponds to the mapping $f(\pi, \epsilon) =
e^{\epsilon\pi}$. Such a mapping is useful when a function has large negative
payoffs, as it will map all payoffs to a positive value regardless of the value
of $\epsilon$. In `ludics`, this is done using `exponential_fitness_map`:

```py
>>> import numpy as np
>>> import ludics
>>> fitness = np.array([0,1,2])
>>> selection_intensity = 0.2

>>> ludics.exponential_fitness_map(fitness=fitness, selection_intensity=selection_intensity)
array([1.        , 1.22140276, 1.4918247 ])

```

Currently, the exponential fitness map only supports numeric entries

### Defining A Fitness Map

You can define your own fitness map in `ludics`. A fitness map must have the
following properties:

1. Takes arguments `fitness` and `selection_intensity`
2. Takes `**kwargs`
3. Returns strictly positive values

An example of this is shown below:

```py
>>> import numpy as np
>>> import ludics

>>> def example_fitness_map(fitness, selection_intensity, **kwargs):
...     return (1 + np.tanh(fitness * selection_intensity))/2
>>> fitness = np.array([0,1,2])
>>> selection_intensity = 0.2
>>> example_fitness_map(fitness=fitness, selection_intensity=selection_intensity)
array([0.5       , 0.59868766, 0.68997448])

```