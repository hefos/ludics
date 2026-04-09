# The public goods game

The public goods game is an $N$-player generalisation of the traditional
prisoner's dilemma. Like the prisoner's dilemma, each player may either
cooperate or defect, and the game follows the process:

1. Each player chooses whether to cooperate or defect
2. Players who cooperate contribute $\alpha$ to the group. Those who defect do
   not
3. The total contribution is multiplied by some value $r$
4. This multiplied amount is split between all players evenly, regardless of
   whether or not they contributed.

The payoff of each player is given by:

$$
f_i(a) = \frac{r\sum_{j=1}^N \alpha_j}{N} - \alpha_i
$$

Where $\alpha_i$ is the amount contributed by player $i$. It takes the value 0
if player $i$ defects, and the value of their contribution if they cooperate.
The contribution of different players may be a heterogeneous attribute of the
population.

See Hauert et al. (2002) in the [bibliography](../reference/bibliography.md).
