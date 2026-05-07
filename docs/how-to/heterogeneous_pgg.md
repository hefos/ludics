# Model a Public Goods Game

`public_goods_game_fitness_function` models a PGG where each
player has a distinct contribution level and rate of return.

See [The public goods game](../explanation/public_goods_game.md) for the
payoff formula.

To model a Homogeneous public goods game, pass a _float_ value of `r` and
`alpha`:

```py
>>> import ludics.fitness_functions
>>> import numpy as np

>>> state = np.array([1,1,1])
>>> alpha=1
>>> r=2

>>> ludics.fitness_functions.public_goods_game_fitness_function(state=state, alpha=alpha, r=r)
array([1., 1., 1.])

```

To model a population with heterogeneous $r$ and $\alpha$, pass a _numpy.array_
value of `r` and `alpha`.

```py
>>> import ludics.fitness_functions
>>> import numpy as np

>>> contribution_vector = np.array([2,3,4])
>>> r = np.array([1.5,2,2.5])
>>> state = np.array([1,1,1])
>>> ludics.fitness_functions.public_goods_game_fitness_function(state=state,
... alpha=contribution_vector, r=r)
array([2.5, 3. , 3.5])

```

This can then be passed to `generate_transition_matrix` to model the game as a
Markov chain. Here player 0 contributes 1 unit, player 1 contributes 2 units, and
player 2 contributes 3 units, and `r` is homogeneous between all players.

```py
>>> import ludics
>>> import ludics.fitness_functions
>>> import numpy as np

>>> N = 3
>>> r = 1.5
>>> contribution_vector = np.array([1.0, 2.0, 3.0])
>>> choice_intensity = 1.0

>>> state_space = ludics.get_state_space(N=N, k=2)
>>> transition_matrix = ludics.generate_transition_matrix(
...     state_space=state_space,
...     fitness_function=ludics.fitness_functions.public_goods_game_fitness_function,
...     compute_transition_probability=ludics.compute_fermi_transition_probability,
...     choice_intensity=choice_intensity,
...     r=r,
...     alpha=contribution_vector,
... )
>>> ludics.compute_absorption_matrix(transition_matrix)
array([[0.99330284, 0.00669716],
       [0.9808759 , 0.0191241 ],
       [0.90462762, 0.09537238],
       [0.94452438, 0.05547562],
       [0.81294603, 0.18705397],
       [0.77450603, 0.22549397]])

```

High contributors bear a greater individual cost relative to their share of
the pool (since $r < N$). States where the high contributor is the lone
contributor fix on defection with the highest probability (~99%).
