# Model heterogeneous contributions for the Public Goods Game

Players often differ in how much they can contribute to a public good.
`heterogeneous_contribution_pgg_fitness_function` models a PGG where each
player has a distinct contribution level.

See [The public goods game](../explanation/public_goods_game.md) for the
payoff formula.

Here player 0 contributes 1 unit, player 1 contributes 2 units, and player 2
contributes 3 units.

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
...     fitness_function=ludics.fitness_functions.heterogeneous_contribution_pgg_fitness_function,
...     compute_transition_probability=ludics.compute_fermi_transition_probability,
...     choice_intensity=choice_intensity,
...     r=r,
...     contribution_vector=contribution_vector,
... )
>>> ludics.approximate_absorption_matrix(transition_matrix)
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
