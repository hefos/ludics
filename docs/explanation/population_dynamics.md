# Population dynamics

A population dynamic is the function by which players in a population update
their action types. The most common type of population dynamic is a
"birth-death process". This is a process in which one player will change their
action type at a time, traditionally representing a player "dying" and another
player "reproducing" - that is, creating a copy of themself to replace the
player who "died".

There are 5 population dynamics considered in `ludics`, all of which are
birth-death processes. They fall into two catagories:

1. Extrinsic Dynamics

   These are dynamics in which players decide on new strategies by copying
   other players who perform well in a given population. The purely extrinsic
   dynamics included in `ludics` are the Moran process (Moran, 1958) and Fermi
   imitation dynamics (Szabó and Tőke, 1998).

2. Intrinsic Dynamics

   These are dynamics in which players decide on new strategies entirely based
   on their own fitness. The purely intrinsic dynamics included in `ludics`
   are introspection dynamics (Couto, Giaimo and Hilbe, 2022) and aspiration
   dynamics (Du et al., 2014).

Introspective imitation dynamics (Foster, Knight and Krapohl, in preparation)
is also included in `ludics`, which has both intrinsic and extrinsic steps.

Full citations are in the [bibliography](../reference/bibliography.md).

## The Moran Process

This follows the algorithm:

1. A player $i$ is selected to reproduce with probability proportional to their
   fitness in the population.
2. A player $j$ is selected uniformly at random to be replaced.
3. Player $j$ adopts the strategy of player $i$.

The transition matrix for a Moran process is defined as follows:

$$
        T_\textbf{ab} =
        \begin{cases}
            \frac{1}{N} \cdot\frac{\sum_{a_i = b_{I(\textbf{a,b})}}{f_i(a)}}{\sum_{a_j}f_j(a)} & \text{if }\textbf{b} \in \text{Neb($\textbf{a}$)}\text{, differing at position }I(\textbf{a,b})\\
            0 & \text{if }\textbf{b} \notin \text{Neb($\textbf{a}$) and $\textbf{a}$}\neq \textbf{b}\\
            1 - \sum_{\textbf{c} \in S \setminus \text{\{\textbf{a}\}}}T_{ac} & \text{if }\textbf{a}=\textbf{b}
        \end{cases}
$$

Where we denote the fitness of player $i$ in state $a$ by $f_i(a)$, and the set
of states which differ from state $a$ in exactly one position by Neb($a$)

## Fermi imitation dynamics

This follows the algorithm:

1. A player $i$ is chosen at random to consider changing strategy, and
   another player $j$ is
   chosen to have their strategy considered
2. Player $i$ accepts player $j$'s strategy with a probability according
   to the Fermi logit function
   $\phi(\Delta(f)) = \frac{1}{1 + e^{\beta(\Delta(f))}}$
   where $\Delta(f) = f_i(a) - f_j(a)$ is the difference between player $i$'s
   fitness and player $j$'s fitness

Where $\beta$ is the _choice intensity_ of the system which defines how often a
player makes the "more rational" decision, with $\beta = 0$ resulting in
completely random updates.

The transition matrix of a process operating under to Fermi imitation dynamics is
defined as
follows:

$$
T_\textbf{ab}  =
        \begin{cases}
            \frac{1}{N(N-1)}\sum_{a_j=b_{I(\textbf{a,b})}}\phi(f_{\text{I(\textbf{a,b})}}(a)
            - f_{j}(\textbf{a})) & \text{if }\textbf{b} \in
            \text{Neb($\textbf{a}$)}\\
            0 & \text{if }\textbf{b} \notin \text{Neb($\textbf{a}$) and $\textbf{a}$}\neq \textbf{b}\\
            1 - \sum_{\textbf{c} \in S \setminus \text{\{\textbf{a}\}}}T_{ab} & \text{if }\textbf{a}=\textbf{b}
        \end{cases}
$$

## Aspiration dynamics

Aspiration dynamics is only defined for games with exactly two strategies, and follows the algorithm:

1. A player $i$ is picked to switch strategy at random
2. They change strategy with probability $\phi(f_i(a) - A_i)$, where $A_i$ is
   the _aspiration_ of player $i$, the fitness that they wish to obtain.

The transition matrix of a process operating under aspiration dynamics is defined as follows:

$$
T_{ab} = \begin{cases}
\frac{1}{N} \cdot \phi(f_{I(a,b)}(a) - A_{I(a,b)}) & \text{if $\textbf{b}$} \in
\text{Neb}(\textbf{a})\\
0 & \text{if }\textbf{b} \notin \text{Neb($\textbf{a}$) and $\textbf{a}$}\neq \textbf{b}\\
            1 - \sum_{\textbf{c} \in S \setminus \text{\{\textbf{a}\}}}T_{ab} & \text{if }\textbf{a}=\textbf{b}
\end{cases}
$$

## Introspection Dynamics

This follows the algorithm:

1. A player $i$ is picked at random to reconsider their strategy
2. A strategy $k$ is picked at random for them to consider
3. They accept the new strategy with a probability $\phi(\Delta(f))$, where
   $\Delta(f) = f_i(a) - f_i(b)$ is the difference between a player's current
   payoff and the possible payoff they could obtain by switching strategy.

The transition matrix of a process operating under introspection dynamics is
defined as follows:

$$
T_{\textbf{ab}} =
\begin{cases}
\dfrac{1}{N(m_j - 1)} \, \phi (f_i(a) - f_i(b))
    & \text{if } \textbf{b} \in \mathrm{Neb}(\textbf{a}) \text{ and } j = I(\textbf{a,b}),\\[1.2em]
0
    & \text{if } \textbf{b} \notin \mathrm{Neb}(\textbf{a}) \text{and $\textbf{a}$} \neq \textbf{b},\\[0.8em]
1 - \sum_{\textbf{c} \in S \setminus \text{\{\textbf{a}\}}}T_{ab} & \text{if }\textbf{a}=\textbf{b}
\end{cases}
$$

## Introspective imitation dynamics

This follows the algorithm:

1. A player $i$ is chosen at random to reconsider their strategy
2. A player $j$ is chosen proportional to their fitness in the population to
   have their strategy considered
3. Player $i$ accepts the strategy of player $j$ with a probability $\phi(\Delta(f))$, where
   $\Delta(f) = f_i(a) - f_i(b)$ is the difference between a player's current
   payoff and the possible payoff they could obtain by switching strategy.

The transition matrix of a process operating under introspective imitation
dynamics is defined as follows:

$$
T_{\textbf{ab}} =  \begin{cases}
    \frac{1}{N}\frac{\sum_{a_{j} = b_{I(\textbf{a}, \textbf{b})}}f_j(\textbf{a})}{\sum_{k}f_k(\textbf{a})}\phi(f_i(a) - f_i(b)) & \text{if $\textbf{b}$}\in \text{Neb($\textbf{a}$)}\\
    0 & \text{if $\textbf{b}$}\notin \text{Neb($\textbf{a}$) and $\textbf{a}$} \neq \textbf{b}\\
    1 - \sum_{\textbf{c} \in S \setminus \text{\{\textbf{a}\}}}T_{ab} & \text{if }\textbf{a}=\textbf{b}
    \end{cases}
$$

## Mutation

Mutation is a step which can be added to any birth-death process. It follows
the algorithm:

1. The population dynamic occurs as usual
2. Before the chosen player $i$ changes strategy, they may instead mutate to
   another action type $k$ with probability $\mu_{ik}$

We can apply mutation to functions by the following transformation:

$$
T^m_{ab} = T_{ab}(1 - \sum_{k \in K}\mu_{I(ab), k}) +  \frac{\mu_{I(a,b),
b_{I(a,b)}}}{N}
$$

where $K$ is the set of action types.
