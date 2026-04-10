# Analyse a Public Goods Game

In this tutorial you will build a complete evolutionary game theory analysis
using `ludics`: from defining a population, through constructing the Markov
chain, to computing fixation probabilities.

The code blocks use Python's interactive prompt format. Lines starting with
`>>>` are code you run; lines without are the output.

## Installing ludics

```
$ pip install ludics
```

## The state space

A population of $N$ players each playing one of $k$ strategies is represented
as an ordered vector $\mathbf{a} = (a_1, \ldots, a_N)$, where $a_i \in \{0,
1, \ldots, k-1\}$ is the strategy of player $i$. The full state space is the
set of all $k^N$ such vectors.

See [How states are represented](../explanation/state_representation.md) for
a detailed explanation of this convention.

```py
>>> import ludics

>>> N = 2
>>> number_of_strategies = 2

>>> ludics.get_state_space(N=N, k=number_of_strategies)
array([[0, 0],
       [0, 1],
       [1, 0],
       [1, 1]])

```

With $N=2$ players and $k=2$ strategies (0 = defect, 1 = cooperate), there are
four states. `[0, 0]` (all-defect) and `[1, 1]` (all-cooperate) are absorbing
under most evolutionary dynamics: once the population reaches either state, it
stays there.

## The fitness function

The fitness function maps a state to a payoff for each player. Here we use a
homogeneous Public Goods Game with multiplication factor $r = 1.5$ and
contribution $\alpha = 2$. Since $r < N = 2$, defection is individually
rational.

```py
>>> import ludics.fitness_functions
>>> import numpy as np

>>> ludics.fitness_functions.homogeneous_pgg_fitness_function(
...     state=np.array([0, 1]),
...     alpha=2,
...     r=1.5,
... )
array([ 1.5, -0.5])

```

In state `[0, 1]`, player 0 defects and player 1 cooperates. The defector
earns 1.5 and the cooperator earns -0.5, as expected when $r < N$.

## The evolutionary dynamic

An evolutionary dynamic determines how the population transitions between
states. Here we use the **Moran process**: fitter players are more likely to
reproduce and pass on their strategy. The `selection_intensity` parameter
controls how strongly fitness differences influence the outcome (0 = neutral
drift, larger values = stronger selection).

`compute_moran_transition_probability` returns the probability of moving from
one state to a neighbouring state in a single step:

```py
>>> import ludics
>>> import ludics.fitness_functions
>>> import numpy as np

>>> source = np.array([0, 1])
>>> target = np.array([0, 0])

>>> ludics.compute_moran_transition_probability(
...     source=source,
...     target=target,
...     selection_intensity=0.5,
...     fitness_function=ludics.fitness_functions.homogeneous_pgg_fitness_function,
...     alpha=2,
...     r=1.5,
... )
np.float64(0.35)

```

From state `[0, 1]`, the probability of moving to `[0, 0]` in one step is 0.35.

## The transition matrix

`generate_transition_matrix` assembles transition probabilities for the entire
state space into a single matrix $T$, where $T_{ij}$ is the probability of
moving from state $i$ to state $j$:

```py
>>> import ludics
>>> import ludics.fitness_functions
>>> import numpy as np

>>> state_space = ludics.get_state_space(N=2, k=2)

>>> ludics.generate_transition_matrix(
...     state_space=state_space,
...     compute_transition_probability=ludics.compute_moran_transition_probability,
...     fitness_function=ludics.fitness_functions.homogeneous_pgg_fitness_function,
...     selection_intensity=0.5,
...     alpha=2,
...     r=1.5,
... )
array([[1.  , 0.  , 0.  , 0.  ],
       [0.35, 0.5 , 0.  , 0.15],
       [0.35, 0.  , 0.5 , 0.15],
       [0.  , 0.  , 0.  , 1.  ]])

```

Rows and columns correspond to states `[0,0]`, `[0,1]`, `[1,0]`, `[1,1]`. The
first and last rows confirm that `[0,0]` and `[1,1]` are absorbing.

## Fixation probabilities

The **fixation probability** is the probability that, starting from a mixed
state, the population eventually fixes on a particular strategy. It is given by
the absorption matrix, whose entry $B_{ij}$ is the probability of fixing in
absorbing state $j$ when starting from transient state $i$:

```py
>>> transition_matrix = ludics.generate_transition_matrix(
...     state_space=state_space,
...     compute_transition_probability=ludics.compute_moran_transition_probability,
...     fitness_function=ludics.fitness_functions.homogeneous_pgg_fitness_function,
...     selection_intensity=0.5,
...     alpha=2,
...     r=1.5,
... )

>>> ludics.approximate_absorption_matrix(transition_matrix)
array([[0.7, 0.3],
       [0.7, 0.3]])

```

The rows correspond to the two transient states `[0,1]` and `[1,0]`; the
columns to `[0,0]` (all-defect) and `[1,1]` (all-cooperate). From either mixed
state, the population fixes on defection with probability 0.7 and on
cooperation with probability 0.3, as expected when $r < N$.

## Further reading

For the theory behind these methods, see the [bibliography](../reference/bibliography.md).
Nowak (2006) and Hofbauer and Sigmund (1998) both cover evolutionary dynamics
and fixation probabilities in depth.
