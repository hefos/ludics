import numpy as np
import sympy as sym

def public_goods_game_fitness_function(
    state, r, alpha, **kwargs
):
    """Public goods fitness function. In this fitness function, player $i$'s
    fitness is defined as:

    $(\frac{r_i \alpha_i}{N} - \alpha_i$

    where $\alpha_i$ is their contribution and $r_i$ is their rate of return.
    We also allow a homogeneous value of $\alpha$ and $r$ where all players
    contribute the same or have the same rate of return. 

    To model a homogeneous public goods game, pass a float value of r and/or
    alpha. For example:

    r = 1.8
    alpha = 2
    state = np.array([1,1,0])
    ludics.fitness_functions.public_goods_game_fitness_function(r=r,
    alpha=alpha, state=state)

    To model a heterogeneous public goods game, pass a numpy.array value of r
    and/or alpha. For example:

    r= np.array([1.5, 2, 2.5])
    alpha = np.array([1,2,3])
    state = np.array([1,1,0])
    ludics.fitness_functions.public_goods_game_fitness_function(r=r,
    alpha=alpha, state=state)

    Parameters
    -----------

    state: numpy.array, the ordered set of actions each player takes; 1 for
    contributing, 0 for free-riding.

    r: float or numpy.array, the parameter which the public goods is multiplied 
    by for each player

    alpha: float or numpy.array, the value which each player contributes

    Returns
    -------

    numpy.array: an ordered vector of each player's fitness in a given state
    when playing a public goods game."""

    total_goods = sum(state * alpha)/ len(state)

    payoff_vector = (r *total_goods) - (state * alpha)

    return payoff_vector


def general_four_state_fitness_function(state, **kwargs):
    """
    Returns a general fitness function for each player in a general 2 player population,
    according to the rule
    $f_i(x)$ is the fitness of player i in state x, indexed from 1.

    In this case, the states correspond to:
    $a=(0,0)$, $b=(0,1)$, $c=(1,0)$, $d=(1,1)$
    This is the same state space as we have in the Population Dynamics section
    of main.tex.

    Parameters
    -----------
    state: numpy.array, the ordered set of actions each player takes

    Returns
    -------
    numpy.array: an ordered array of each player's fitness"""

    if (state == np.array([0, 0])).all():
        state_symbol = sym.Symbol("a")
    if (state == np.array([0, 1])).all():
        state_symbol = sym.Symbol("b")
    if (state == np.array([1, 0])).all():
        state_symbol = sym.Symbol("c")
    if (state == np.array([1, 1])).all():
        state_symbol = sym.Symbol("d")

    return np.array(
        [sym.Function(f"f_{i + 1}")(state_symbol) for i, j in enumerate(state)]
    )

def pairwise_interaction_fitness_function(state,a,b,c,d):
    """
    Returns the fitness of players in a symmetric pairwise interaction game. 
    The payoff matrix for the row player is given by:
    
    |   |  C  |  D  | 
    |---|-----|-----|
    | C | a_i | b_i |
    | D | c_i | d_i |

    And a player's fitness is given by the mean payoff they receive when
    interacting in a 2 player game with a random other member of the 
    population.

    Parameters:
    -----------
    state: numpy.array, the ordered set of actions each player takes. 1 is a
    cooperator, 0 is a defector,

    a,b,c,d: numpy.array, the entries to go into the payoff matrix. Entry i is the
    respective payoff for player i's payoff matrix

    Returns:
    ---------
    numpy.array: an ordered array of each player's fitness
    """
    N = len(state)
    number_of_cooperators = np.sum(state)
    cooperator_payoffs = np.array([((number_of_cooperators-1) * a[i] + (N - number_of_cooperators) * b[i])/(N-1) for i in range(N)])
    defector_payoffs = np.array([((number_of_cooperators) * c[i] + (N - number_of_cooperators-1) * d[i])/(N-1) for i in range(N)])
    return np.array([cooperator_payoffs[player] if action == 1 else defector_payoffs[player] for player, action in enumerate(state)])