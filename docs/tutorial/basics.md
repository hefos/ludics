# The Basics

In this tutorial, you'll learn how to install `ludics`, create a state space,
choose a fitness function and population dynamic,
and define a simple transition matrix.

## Installing Ludics

To install ludics:

```
uv add ludics
```

## Create a state space

First define a state space. This is the set of possible populations that our
model can take.

```py
>>> import ludics.main

>>> N = 2
>>> number_of_strategies = 2

>>> ludics.main.get_state_space(N=N, k=number_of_strategies)
array([[0, 0],
       [0, 1],
       [1, 0],
       [1, 1]])

```

## Choose a fitness function

Now choose a fitness function. The ludics.fitness_functions module provides a
selection of common fitness functions for evolutionary game theory.

```py
>>> import ludics.fitness_functions
>>> import numpy as np

>>> ludics.fitness_functions.homogeneous_pgg_fitness_function(
... state=np.array([0,1]),
... alpha=2,
... r=1.5
... )
array([ 1.5, -0.5])

```

## Choose a population dynamic

Population dynamics calculate the probability of transitioning
between two states. There are 5 population dynamics built into ludics.main. We use
them as follows:

```py
>>> import ludics.main
>>> import ludics.fitness_functions
>>> import numpy as np

>>> source = np.array([0,1])
>>> target = np.array([0,0])
>>> r = 1.5
>>> alpha = 2
>>> selection_intensity = 0.5

>>> ludics.main.compute_moran_transition_probability(
... source=source,
... target=target,
... selection_intensity=0.5,
... fitness_function=ludics.fitness_functions.homogeneous_pgg_fitness_function,
... alpha=2,
... r=1.5
... )
np.float64(0.35)

```

## Create a Transition Matrix

Use the `generate_transition_matrix` function.

```py
>>> import ludics.main
>>> import ludics.fitness_functions
>>> import numpy as np

>>> state_space = ludics.main.get_state_space(N=2,k=2)
>>> selection_intensity=0.5
>>> r=1.5
>>> alpha=2

>>> ludics.main.generate_transition_matrix(
... state_space=state_space,
... compute_transition_probability=ludics.main.compute_moran_transition_probability,
... fitness_function=ludics.fitness_functions.homogeneous_pgg_fitness_function,
... selection_intensity=0.5,
... alpha=2,
... r=1.5
... )
array([[1.  , 0.  , 0.  , 0.  ],
       [0.35, 0.5 , 0.  , 0.15],
       [0.35, 0.  , 0.5 , 0.15],
       [0.  , 0.  , 0.  , 1.  ]])

```
