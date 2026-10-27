# Compare evolutionary dynamics on a Public Goods Game

The same game can produce qualitatively different outcomes under different
evolutionary dynamics. This guide compares the Moran process, Fermi imitation,
and introspection dynamics on a homogeneous PGG.

See [What is a population dynamic?](../explanation/population_dynamics.md) for
the mathematical definitions and [How states are represented](../explanation/state_representation.md)
for the state convention used below.

## When to use each dynamic

**Moran process**: a classical, well-studied baseline. Results are easily
interpretable due to the simple model.

**Fermi imitation**: models noisy social learning, where players sometimes copy
worse-performing neighbours. The `choice_intensity` parameter ($\beta$)
controls rationality: $\beta \to 0$ is neutral drift, $\beta \to \infty$ is
strict imitation of better-performing players.

**Introspection**: players deliberate privately about whether to switch,
without comparing themselves to others. The chain is ergodic with no absorbing
states; use the stationary distribution rather than fixation probabilities.

**Aspiration**: players are not concerned about reaching their maximum payoff,
but instead wish to reach a specific target payoff.

Moran and Fermi produce **absorbing** chains. Introspection and aspiration
produce **ergodic** chains.

```py
>>> import ludics
>>> import ludics.fitness_functions
>>> import numpy as np

>>> N = 3
>>> r = 1.5
>>> alpha = 1
>>> state_space = ludics.get_state_space(N=N, k=2)

```

## Moran process (absorbing chain)

```py
>>> tm_moran = ludics.generate_transition_matrix(
...     state_space=state_space,
...     fitness_function=ludics.fitness_functions.public_goods_game_fitness_function,
...     compute_transition_probability=ludics.compute_moran_transition_probability,
...     selection_intensity=0.5,
...     r=r,
...     alpha=alpha,
...     fitness_map=ludics.linear_fitness_map
... )
>>> ludics.compute_absorption_matrix(tm_moran)
array([[0.9, 0.1],
       [0.9, 0.1],
       [0.6, 0.4],
       [0.9, 0.1],
       [0.6, 0.4],
       [0.6, 0.4]])

```

## Fermi imitation dynamics (absorbing chain)

```py
>>> tm_fermi = ludics.generate_transition_matrix(
...     state_space=state_space,
...     fitness_function=ludics.fitness_functions.public_goods_game_fitness_function,
...     compute_transition_probability=ludics.compute_fermi_transition_probability,
...     choice_intensity=1,
...     r=r,
...     alpha=alpha,
... )
>>> ludics.compute_absorption_matrix(tm_fermi)
array([[0.90996943, 0.09003057],
       [0.90996943, 0.09003057],
       [0.66524096, 0.33475904],
       [0.90996943, 0.09003057],
       [0.66524096, 0.33475904],
       [0.66524096, 0.33475904]])

```

## Introspection dynamics (ergodic chain)

Under introspection, a player can always reconsider even when all others play
the same strategy, so the chain has no absorbing states.

```py
>>> tm_intro = ludics.generate_transition_matrix(
...     state_space=state_space,
...     fitness_function=ludics.fitness_functions.public_goods_game_fitness_function,
...     compute_transition_probability=ludics.compute_introspection_transition_probability,
...     choice_intensity=1,
...     number_of_strategies=2,
...     r=r,
...     alpha=alpha,
... )
>>> ludics.compute_steady_state(tm_intro)
array([0.24117286, 0.14628008, 0.14628008, 0.08872416, 0.14628008,
       0.08872416, 0.08872416, 0.05381442])

```

The highest weight falls on all-defect $[0,0,0]$ (~24%) and the lowest on
all-contribute $[1,1,1]$ (~5%), as expected when $r < N$.

## Aspiration dynamics (ergodic chain)
```py
>>> tm_asp = ludics.generate_transition_matrix(
...     state_space=state_space,
...     fitness_function=ludics.fitness_functions.public_goods_game_fitness_function,
...     compute_transition_probability=ludics.compute_aspiration_transition_probability,
...     choice_intensity=1,
...     aspiration_vector=np.array([2,4,6]),
...     r=r,
...     alpha=alpha,
... )
>>> ludics.compute_steady_state(tm_asp)
array([0.12960668, 0.13042322, 0.12923832, 0.12984536, 0.12192501,
       0.12109432, 0.11965307, 0.11821403])

```

Here, players play in intermediate states where a one or two players contribute,
but not all. Players reach their baseline fitness in these states, and to begin
to contribute beyond this point would reduce their payoff.

## Summary

| Dynamic         | Chain type | Key quantity            |
| --------------- | ---------- | ----------------------- |
| Moran           | Absorbing  | Fixation probabilities  |
| Fermi imitation | Absorbing  | Fixation probabilities  |
| Introspection   | Ergodic    | Stationary distribution |
| Aspiration      | Ergodic    | Stationary distribution |