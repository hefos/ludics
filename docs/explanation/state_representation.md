# How states are represented

`ludics` represents the population as an **ordered vector of individual
strategies**, one entry per player:

$$\mathbf{a} = (a_1, a_2, \ldots, a_N)$$

where $a_i \in \{0, 1, \ldots, k-1\}$ is the strategy currently played by
player $i$, $N$ is the population size, and $k$ is the number of strategies.

For example, with $N = 3$ players and $k = 2$ strategies (0 = defect, 1 =
cooperate):

| State | Meaning |
|---|---|
| `[0, 0, 0]` | all three players defect |
| `[0, 0, 1]` | players 0 and 1 defect; player 2 cooperates |
| `[1, 1, 0]` | players 0 and 1 cooperate; player 2 defects |
| `[1, 1, 1]` | all three players cooperate |

The full state space contains $k^N$ states: for $N=3$, $k=2$ this is 8.

## Why individual-indexed states?

This representation tracks *who* is playing *what*, not just *how many* players
use each strategy. It is more general than a frequency-based representation: it
supports **heterogeneous players** (different contribution levels, different
aspiration levels) and **asymmetric payoff functions** where the identity of
the player matters, not just the aggregate counts.

The cost is a larger state space. For symmetric games where only the count of
each strategy matters, many states are payoff-equivalent: `[0, 1, 0]` and
`[0, 0, 1]` produce the same fitness values under a homogeneous PGG. `ludics`
does not collapse these equivalent states; it works with the full $k^N$ space.

## Absorbing states

In games with two strategies, the two fully-homogeneous states (all-0 and
all-1) are typically absorbing under extrinsic dynamics (Moran, Fermi), since
there is no fitness difference to drive a change. Their absorption
probabilities are the **fixation probabilities** central to evolutionary game
theory.
