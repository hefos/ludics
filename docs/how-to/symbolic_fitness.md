# Use symbolic payoffs for a 2-player game

`general_four_state_fitness_function` returns fully symbolic payoffs for a
2-player, 2-action game. You can build and simplify the transition and
absorption matrices algebraically, then substitute specific payoffs at the end.

The four states and their symbolic labels are:

| State | Label |
|---|---|
| `[0, 0]` | $a$ |
| `[0, 1]` | $b$ |
| `[1, 0]` | $c$ |
| `[1, 1]` | $d$ |

Player $i$'s fitness in state $x$ is written $f_i(x)$.

## Evaluate the fitness function

```py
>>> import ludics.fitness_functions
>>> import numpy as np

>>> ludics.fitness_functions.general_four_state_fitness_function(
...     np.array([0, 1])
... )
array([f_1(b), f_2(b)], dtype=object)

>>> ludics.fitness_functions.general_four_state_fitness_function(
...     np.array([1, 0])
... )
array([f_1(c), f_2(c)], dtype=object)

```

## Build a symbolic transition matrix

```py
>>> import ludics
>>> import ludics.fitness_functions
>>> import numpy as np

>>> state_space = ludics.get_state_space(N=2, k=2)
>>> transition_matrix = ludics.generate_transition_matrix(
...     state_space=state_space,
...     fitness_function=ludics.fitness_functions.general_four_state_fitness_function,
...     compute_transition_probability=ludics.compute_moran_transition_probability,
...     selection_intensity=1,
... )

```

## Compute and simplify the symbolic absorption matrix

Use `calculate_absorption_matrix` for symbolic matrices, then `sympy.cancel`
to reduce the expressions:

```py
>>> import sympy as sym

>>> absorption = ludics.calculate_absorption_matrix(transition_matrix)
>>> sym.cancel(absorption)
Matrix([
[(1.0*f_1(b) + 1.0)/(1.0*f_1(b) + 1.0*f_2(b) + 2.0), (1.0*f_2(b) + 1.0)/(1.0*f_1(b) + 1.0*f_2(b) + 2.0)],
[(1.0*f_2(c) + 1.0)/(1.0*f_1(c) + 1.0*f_2(c) + 2.0), (1.0*f_1(c) + 1.0)/(1.0*f_1(c) + 1.0*f_2(c) + 2.0)]])

```

The columns correspond to fixation in `[0, 0]` and `[1, 1]` respectively. This
is the **standard Moran fixation probability formula** for a 2-player game.
From state $b = [0, 1]$, the fixation probability into all-zeros is:

$$\rho_{b \to [0,0]} = \frac{f_1(b) + 1}{f_1(b) + f_2(b) + 2}$$

The fixation probability is proportional to the fitness of the invading
strategy relative to the total fitness in the mixed state.

## Substitute specific payoffs

To evaluate for a particular game, substitute numerical values for the symbolic
fitness terms. For a Prisoner's Dilemma with $T=5$, $S=0$:

- State $b = [0, 1]$: player 0 defects, player 1 cooperates, so $f_1(b) = T = 5$, $f_2(b) = S = 0$
- State $c = [1, 0]$: player 0 cooperates, player 1 defects, so $f_1(c) = S = 0$, $f_2(c) = T = 5$

```py
>>> simplified = sym.cancel(absorption)
>>> subs = {
...     sym.Function("f_1")(sym.Symbol("b")): 5,
...     sym.Function("f_2")(sym.Symbol("b")): 0,
...     sym.Function("f_1")(sym.Symbol("c")): 0,
...     sym.Function("f_2")(sym.Symbol("c")): 5,
... }
>>> simplified.subs(subs)
Matrix([
[0.857142857142857, 0.142857142857143],
[0.857142857142857, 0.142857142857143]])

```

Both transient states fix on defection with probability ~86%, as expected when
defection is the dominant strategy.
