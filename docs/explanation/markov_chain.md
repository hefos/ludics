# Modelling evolutionary games as Markov chains

We can model any evolutionary game as a [Markov
chain](https://en.wikipedia.org/wiki/Markov_chain) by taking our state space as
the set of possible populations. In doing so, we can define
a transition matrix for the system. For example, for a 2-player, 2-action game
we have:

$$
T =
\begin{pmatrix}
T_{11}, T_{12}, T_{13}, T_{14}\\
T_{21}, T_{22}, T_{23}, T_{24}\\
T_{31}, T_{32}, T_{33}, T_{34}\\
T_{41}, T_{42}, T_{43}, T_{44}
\end{pmatrix}
$$

where $T_{ab} = p(a \to b)$ is the probability of transitioning from state $a$ to state $b$.

## Types of Markov chain

### Absorbing Markov chains

The
first type of Markov chain is an _absorbing Markov chain_. This is a process where certain states can
only transition to themselves, and thus the Markov chain gets stuck in said
states. The transition matrix of such a process will have at least one row such
that

$$
T_{ij} = \begin{cases}
1 & \text{if i = j}\\
0 & \text{otherwise}
\end{cases}
$$

In such a Markov chain, we can calculate an _absorption matrix_ as follows:

1. Write the transition matrix in canonical form

This is the form

$$
  T =
  \left(\begin{array}{c|c}
    Q & R \\\hline
    0 & I
  \end{array}\right)
$$

Where $Q$ is the set of transitions from transitive (non-absorbing) states to
other transitive states, and $R$ is the set of transitions from transitive
states to absorbing states.

2. Compute the absorption matrix

$$
B = (I - Q)^{-1} R = \begin{pmatrix}
B_{11}& B_{12}\\
B_{21} & B_{22}
\end{pmatrix}
$$

where $B_{ab}$ is the probability of the Markov chain being absorbed into state
$b$ given that it started in state $a$.

This can be used to analyse which
strategies are favoured by the population. For example, in a Moran process,
absorbing states are those in which all players have the same action type, and
so the state with the highest probability of absorption would be that which has
all players playing the strategy which produces the highest fitness in the
model.

### Ergodic Markov chains

An ergodic Markov chain is one which has no absorbing states, and so you are
able to traverse from any state to any other state in a finite number of steps.
In this case, we consider state distributions in the form

$$ \pi^t = (\pi^t_1, \pi^t_2, \dots)$$

Where $\pi^t_i$ is the proportion of the time spent in a certain state at step
$T$. We have that $\sum_{i=1}^N \pi^t_i = 1$. Now, if we multiply $\pi^t$ by
the transition matrix $T$, we get the distribution at the next step:

$$ \pi^t T = \pi^{t+1}$$

Thus, by taking the left eigenvector of the transition matrix $T$, we obtain a
state distribution $\pi$ such that:

$$ \pi T = \pi$$

We call this the _steady state_ of the Markov chain, and it gives us the
distribution of the system amongst it's states when the system has stabilised.
A Markov chain will always reach it's unique steady state provided that it is
irreducible (all states can reach all other states in a finite number of steps)
and aperiodic (the states do not oscillate between certain subsets). Any
birth-death process fulfills these conditions, and thus reaches the steady
state in a finite number of steps.

### Homogeneous Birth-death chains

If we have a homogeneous population, states with $N_j$ players of each type are
indistinguishable. We can therefore collapse the state space into the set of
tuples:

$$ (N_1, N_2, N_3, \dots) $$

where two states are adjacent if they differ in exactly two positions,
increasing one by a single player and removing that player from the other.

In a two-action game with actions {$0,1$} we can represent this state space as the
numbers $0$ through $N$, where the state is represented by the number of players
using strategy $0$. In this case, we consider probabilities of "birth" and "death",
given by:

$$ \lambda_{i, i+1}, \lambda_{i,i-1} $$

The transition matrix of this process has the form:

$$ \begin{pmatrix}
1 & 0 & 0 & \dots & 0 & 0 & 0\\
\lambda_{1, 0} & 1 - \lambda_{1,0} - \lambda_{1,2} & \lambda_{1,2} & \dots & 0 & 0 & 0\\
0 & \lambda_{2,1} & 1 - \lambda_{2,1} - \lambda_{2,2} & \dots & 0 & 0 & 0\\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots\\
0 & 0 & 0 & \dots & \lambda_{N-1, N-2} & 1-\lambda_{N-1, N-2}-\lambda_{N-1, N}  & \lambda_{N-1, N}\\
0 & 0 & 0 & \dots & 0 & 0 & 1
\end{pmatrix}
$$

and the fixation probability of being absorbed into the state with $N$ players
using action $0$, after beginning with $i$ such players, is given by:

$$ \pi_i = \frac{1 + \sum_{j=1}^{i-1}\prod_{k=1}^j \frac{\lambda_{k,k-1}}{\lambda_{k,k+1}}}{1 + \sum_{j=1}^{i-1}\prod_{k=1}^{N-1} \frac{\lambda_{k,k-1}}{\lambda_{k,k+1}}} $$

For further reading, see chapters 6 and 7 of (Nowak, 2006).