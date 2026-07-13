# Heterogeneous Intensities

When studying population dynamics, population dynamics consider parameters
$\epsilon$ and $\beta$, the _selection_ and _choice_ intensities respectively.
These control the rationality of the process: how much the payoff difference is
taken into account when selecting a player for consideration (selection
intensity) or when choosing whether or not to accept a strategy (choice
intensity). 

`ludics` models both heterogeneous selection and choice intensity. 

## Selection intensity

Selection intensity is used in fitness proportional selection to control how
much a player takes payoff value into account. The Moran process, for example, 
uses this value as follows:

$$
            \frac{1}{N} \cdot\frac{\sum_{i:a_i = b_{I(\textbf{a,b})}}{1 - \epsilon + \epsilon\pi_i(\mathbf{a})}}{\sum_{a_j}1 - \epsilon + \epsilon\pi_i(\mathbf{a})}
$$

If $\epsilon = 0$, then we have neutral drift, where the payoffs do not affect
the transition probabilities. If $\epsilon = 1$, fitness is equal to the payoff.
The value of $\epsilon$ must by chosen such that all players admit a strictly
positive fitness. 

`ludics` allows heterogeneous selection intensity. That is, a matrix $\epsilon$,
with entries $\epsilon_{ij}$. In this case, the transition probability in the
Moran process becomes:

$$
            \frac{1}{N} \cdot\frac{\sum_{i:a_i = b_{I(\textbf{a,b})}}{1 - \epsilon_{I(\textbf{a,b}), i} + \epsilon_{I(\textbf{a,b}), i}\pi_i(\mathbf{a})}}{\sum_{a_j}1 - \epsilon_{I(\textbf{a,b}), i} + \epsilon_{I(\textbf{a,b}), i}\pi_i(\mathbf{a})}
$$

A greater $\epsilon_{ij}$ indicates that player $i$ has a bias towards selecting
player $j$; that is, player $i$ sees an inflated value for player $j$'s fitness.
On the other hand, a lower $\epsilon_{ij}$ indicates that player $i$ is bias
against player $j$, seeing a decreased fitness for said player.

## Choice intensity

For population dynamcis which decides whether or not to update using the Fermi
imitation function $\frac{1}{1 + e^{\beta\Delta(\pi)}}$, choice intensity
$\beta$ controls the rationality of the decision. A higher $\beta$ leads to
players caring more about the payoff difference when making decision.
Conversely, a value of $\beta=0$ gives neutral drift.

A heterogeneous value of $\beta$ is implemented differently depending on the
process. For the three dynamics included in `ludics` which feature choice
intensity, it is implemented as follows:

- Fermi imitation dynamics: $\beta_{ij}$ is the rationality with which
player $i$ accepts the strategy of player $j$. As such, the choice intensity
matrix is of shape $(N,N)$.
- Introspection Dynamics: $\beta_{ik}$ is the rationality with which player $i$
  accepts strategy $k$, where $k$ is the index of the strategy according to the
  [representation of states](state_representation.md). The shape of the choice
  intensity matrix is $(N,K)$, where $K$ is the number of strategies.
- Aspiration Dynamics: $\beta_{ik}$ is the rationality with which player $i$
  changes strategy when playing strategy $k$. This is because aspiration
  dynamics does not consider which strategy to change to (traditionally it was
  only defined for 2 strategies). Thus a player considers the rationality by
  which they switch away from (or do not switch away from) their current
  strategy. The shape of the choice intensity matrix is $(N,K)$