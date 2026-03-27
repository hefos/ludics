# The Basics

In this tutorial, you'll learn how to install `ludics`, create a state space,
choose a fitness function and population dynamic,
and define a simple transition matrix. For more information about modelling evolutionary
games as Markov chains, we suggest Evolutionary dynamics, Exploring The
Equations of Life - Martin A. Nowak and Evolutionary Games and Population
Dynamics - Josef Hofbauer and Karl Sigmund.

## Installing Ludics

To install ludics with uv:

```
uv add ludics
```

or using pip:

```
python -m pip install ludics
```

## Create a state space

We first define a state space. This is the set of possible populations that our
model can take.

```
import ludics

N = 2
number_of_strategies = 2

ludics.get_state_space(N=N, k=number_of_strategies)
```

This will return:

```
array([[0, 0],
       [0, 1],
       [1, 0],
       [1, 1]])
```

We denote different action types by the values 0,1,2,...

## Choose a fitness function

We then choose a fitness function. This shows how well a player performs in a
given state. The ludics.fitness_functions module provides a selection of common
fitness functions for evolutionary game theory. We see an example for the
homogeneous public goods game. To learn more about the public goods game, see
the previously mentioned textbook by Martin A. Nowak:

```
import ludics.fitness_functions

ludics.fitness_functions.homogeneous_pgg_fitness_function(
    state=state_space[1],
    alpha=2,
    r=1.5
)
```

Which will return:

```
array([ 1.5, -0.5])
```

## Choose a population dynamic

We use population dynamics to calculate the probability of transitioning
between two states. There are 5 population dynamics built into ludics. We use
them as follows:

```
ludics.compute_moran_transition_probability(
    source=state_space[1],
    target=state_space[0],
    selection_intensity=0.5,
    fitness_function=ludics.fitness_functions.homogeneous_pgg_fitness_function,
    alpha=2,
    r=1.5
)
```

Then this returns:

```
np.float64(0.35)
```

## Create a Transition Matrix

We now have the three ingredients to create a transition matrix. We do this
using the `generate_transition_matrix` function.

```
ludics.generate_transition_matrix(
    state_space=state_space,
    compute_transition_probability=ludics.compute_moran_transition_probability,
    fitness_function=ludics.fitness_functions.homogeneous_pgg_fitness_function,
    selection_intensity=0.5,
    alpha=2,
    r=1.5
)
```

This returns a transition matrix:

```
array([[1.  , 0.  , 0.  , 0.  ],
       [0.35, 0.5 , 0.  , 0.15],
       [0.35, 0.  , 0.5 , 0.15],
       [0.  , 0.  , 0.  , 1.  ]])
```
