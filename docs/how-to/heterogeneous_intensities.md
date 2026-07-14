# Implement Heterogeneous Intensities

This section shows how to implement heterogeneous choice and selection
intensities into each of the population dynamics included in `ludics`. For more
information about heterogeneous selection and choice intensities, see the
[explanation page](../explanation/intensities.md).

### The Moran Process

In the Moran process, `selection_intensity` can be passed as a _float_, or as a
_numpy.array_ with shape $(N,N)$, where entry $i,j$ is the selection intensity
when player $i$ copies player $j$

```py
>>> import ludics
>>> import numpy as np

>>> def trivial_fitness_function(state, **kwargs):
...         return np.array([2 for i in state])
    
>>> source = np.array([1,0,1,0])
>>> target = np.array([0,0,1,0])

>>> selection_intensity = np.array([
...         [0.1,0.2,0.3,0.4],
...         [0.05, 0.01, 0.02, 0.03],
...         [0.06, 0.07, 0.08, 0.09],
...         [0.5,0.6,0.7,0.8]
... ])

>>> ludics.compute_moran_transition_probability(
...     source=source,
...     target=target,
...     fitness_function=trivial_fitness_function,
...     selection_intensity=selection_intensity,
... )
np.float64(0.13)

```

### Fermi Imitation Dynamics

Fermi imitation dynamics takes `choice_intensity` as a parameter. This can be
passed as a _float_, or as a _numpy.array_, with shape $(N,N)$, where entry
$i,j$ is the choice intensity when player $i$ copies player $j$.

```py
>>> import ludics
>>> import numpy as np

>>> def trivial_fitness_function(state, **kwargs):
...         return np.array([i+1 for i,_ in enumerate(state)])
    
>>> source = np.array([1,0,1,0])
>>> target = np.array([1,0,0,0])

>>> choice_intensity = np.array([
...         [0.1,0.2,0.3,0.4],
...         [0.05, 0.01, 0.02, 0.03],
...         [0.06, 0.07, 0.08, 0.09],
...         [0.5,0.6,0.7,0.8]
... ])

>>> ludics.compute_fermi_transition_probability(
...     source=source,
...     target=target,
...     fitness_function=trivial_fitness_function,
...     choice_intensity=choice_intensity,
... )
0.0837493305937842

```

### Introspection Dynamics

Introspection dynamics takes a `choice_intensity` parameter. This can be passed
as a _float_, or as a _numpy.array_ with shape $(N,K)$, where $K$ is the number
of strategies in the population. Entry $i,j$ is the choice intensity when player
$i$ considers strategy $j$, as represented by numeric states (for more
information on state representation, see
[here](../explanation/state_representation.md))

```py
>>> import ludics
>>> import numpy as np

>>> def trivial_fitness_function(state, **kwargs):
...         return np.array([i for _,i in enumerate(state)])
    
>>> source = np.array([1,0,0])
>>> target = np.array([0,0,0])

>>> choice_intensity = np.array([
...     [0.1,0.2],
...     [0.05, 0.12],
...     [0.06, 0.07],
... ])

>>> ludics.compute_introspection_transition_probability(
...     source=source,
...     target=target,
...     fitness_function=trivial_fitness_function,
...     choice_intensity=choice_intensity,
...     number_of_strategies=2
... )
0.158340270840353

```

### Aspiration Dynamics

Aspiration dynamics takes a `choice_intensity` parameter. This can be passed
as a _float_, or as a _numpy.array_ with shape $(N,2)$. Entry $i,j$ is the
choice intensity when player $i$ considers changing away from strategy
$j$, as represented by numeric states.

```py
>>> import ludics
>>> import numpy as np

>>> def trivial_fitness_function(state, **kwargs):
...         return np.array([i for _,i in enumerate(state)])
    
>>> source = np.array([0,1,1,0])
>>> target = np.array([0,1,0,0])

>>> choice_intensity = np.array([
...     [0.1,0.2],
...     [0.05, 0.12],
...     [0.06, 0.07],
...     [0.8, 0.2],
... ])

>>> aspiration_vector = np.array([1,3,4,2])

>>> ludics.compute_aspiration_transition_probability(
...     source=source,
...     target=target,
...     fitness_function=trivial_fitness_function,
...     choice_intensity=choice_intensity,
...     aspiration_vector=aspiration_vector
... )
0.138076977393581

```