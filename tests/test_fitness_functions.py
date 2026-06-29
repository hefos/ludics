import ludics
import sympy as sym
import numpy as np
import ludics.fitness_functions


def test_public_goods_game_fitness_function_for_homogeneous_contribution_and_return():
    """
    Tests that public_goods_game_fitness_function correctly
    handles a homogeneous contribution. Simultaneously shows that the function
    takes into account each player's choice of contribution. Also tests that a
    float value of r is successfully passed"""

    homogeneous_contributions = 2
    state = np.array([1, 0, 1])
    r = 1.8

    actual_return = ludics.fitness_functions.public_goods_game_fitness_function(
        state=state, alpha=homogeneous_contributions, r=r
    )

    expected_return = np.array([0.4, 2.4, 0.4])

    np.testing.assert_allclose(actual_return, expected_return)


def test_public_goods_game_fitness_function_for_heterogeneous_numeric_contribution():
    """
    Tests that public_goods_game_fitness_function returns the
    correct values for a purely numerical hetereogeneous contribution"""

    heterogeneous_contributions = np.array([1, 3, 4, 8])
    state = np.array([1, 1, 1, 1])
    r = 2.1

    actual_return = ludics.fitness_functions.public_goods_game_fitness_function(
        state=state,
        alpha=heterogeneous_contributions,
        r=r,
    )

    expected_return = np.array([7.4, 5.4, 4.4, 0.4])

    np.testing.assert_allclose(actual_return, expected_return)


def test_public_goods_game_fitness_function_for_heterogeneous_symbolic_contribution():
    """
    Tests that public_goods_game_fitness_function returns the
    correct values for a purely numerical hetereogeneous contribution"""

    a = sym.Symbol("a")
    b = sym.Symbol("b")

    heterogeneous_contributions = np.array([a, b])
    state = np.array([1, 1])
    r_vector = np.array([sym.Symbol(r"$r_1$"),sym.Symbol(r"$r_2$")])

    actual_return = ludics.fitness_functions.public_goods_game_fitness_function(
        state=state,
        alpha=heterogeneous_contributions,
        r=r_vector,
    )

    expected_return_p1 = r_vector[0] * (a/2 + b/2) - a
    expected_return_p2 = r_vector[1] * (a/2 + b/2) - b

    expected_return = np.array([expected_return_p1, expected_return_p2])

    np.testing.assert_array_equal(actual_return, expected_return)


def test_public_goods_game_fitness_function_for_heterogeneous_no_contribution():
    """
    Tests that public_goods_game_fitness_function returns the
    correct values when no players contribute"""

    heterogeneous_contributions = np.array([8, 41, 28, 19])
    state = np.array([0, 0, 0, 0])
    r = 3

    actual_return = ludics.fitness_functions.public_goods_game_fitness_function(
        state=state,
        alpha=heterogeneous_contributions,
        r=r,
    )

    expected_return = np.array([0, 0, 0, 0])

    np.testing.assert_allclose(actual_return, expected_return)


def test_public_goods_game_fitness_function_for_numeric_value():
    """
    Tests that public_goods_game_fitness_function returns the correct value for a
    purely numerical system."""

    alpha = 2
    state = np.array([1, 1, 0, 1, 0, 0])
    r = 2

    actual_return = ludics.fitness_functions.public_goods_game_fitness_function(state=state, alpha=alpha, r=r)

    expected_return = np.array([0, 0, 2, 0, 2, 2])

    np.testing.assert_allclose(actual_return, expected_return)


def test_public_goods_game_fitness_function_for_homogeneous_symbolic_value():
    """
    Tests that public_goods_game_fitness_function returns the correct value for
    purely symbolic values."""

    alpha = sym.Symbol("alpha")
    state = np.array([1, 1, 0])
    r = sym.Symbol("r")

    actual_return = ludics.fitness_functions.public_goods_game_fitness_function(state=state, alpha=alpha, r=r)

    contributor_payment = (2 * r * alpha / 3) - alpha
    defector_payment = 2 * r * alpha / 3

    expected_return = np.array(
        [contributor_payment, contributor_payment, defector_payment]
    )

    np.testing.assert_array_equal(actual_return, expected_return)


def test_public_goods_game_fitness_function_for_no_contribution():
    """
    Tests that public_goods_game_fitness_function returns the correct value when
    no players contribute."""

    alpha = 2
    state = np.array([0, 0, 0])
    r = 1.8

    actual_return = ludics.fitness_functions.public_goods_game_fitness_function(state=state, alpha=alpha, r=r)

    expected_return = np.array([0, 0, 0])

    np.testing.assert_array_equal(actual_return, expected_return)


def test_general_four_state_player_fitness_function_for_whole_state_space():
    """Tests that all players are assigned the correct values in all possible
    states of the four-state system where N=2, k=2"""
    four_state_space = ludics.get_state_space(N=2, k=2)

    expected_return_a = np.array(
        [
            sym.Function(f"f_{1}")(sym.Symbol("a")),
            sym.Function(f"f_{2}")(sym.Symbol("a")),
        ]
    )

    expected_return_b = np.array(
        [
            sym.Function(f"f_{1}")(sym.Symbol("b")),
            sym.Function(f"f_{2}")(sym.Symbol("b")),
        ]
    )

    expected_return_c = np.array(
        [
            sym.Function(f"f_{1}")(sym.Symbol("c")),
            sym.Function(f"f_{2}")(sym.Symbol("c")),
        ]
    )

    expected_return_d = np.array(
        [
            sym.Function(f"f_{1}")(sym.Symbol("d")),
            sym.Function(f"f_{2}")(sym.Symbol("d")),
        ]
    )

    expected_returns = [
        expected_return_a,
        expected_return_b,
        expected_return_c,
        expected_return_d,
    ]

    for i in range(4):
        np.testing.assert_array_equal(
            expected_returns[i],
            ludics.fitness_functions.general_four_state_fitness_function(four_state_space[i]),
        )

def test_public_goods_game_fitness_function_for_homogeneous_contribution():
    """
    Tests that public_goods_game_fitness_function correctly
    handles a homogeneous contribution. Simultaneously shows that the function
    takes into account each player's action type"""

    homogeneous_contributions = np.array([2, 2, 2])
    state = np.array([1, 0, 1])
    r = np.array([1.8, 1.8, 1.8])

    actual_return = ludics.fitness_functions.public_goods_game_fitness_function(
        state=state, alpha=homogeneous_contributions, r=r
    )

    expected_return = np.array([0.4, 2.4, 0.4])

    np.testing.assert_allclose(actual_return, expected_return)

def test_public_goods_game_fitness_function_for_heterogeneous_contribution_and_homogeneous_r():
    """
    Tests that public_goods_game_fitness_function correctly
    handles a heterogeneous contribution with a homogeneous r."""

    heterogeneous_contributions = np.array([1, 3, 4, 8])
    state = np.array([1, 1, 1, 1])
    r = np.array([2.1,2.1,2.1,2.1])

    actual_return = ludics.fitness_functions.public_goods_game_fitness_function(
        state=state,
        alpha=heterogeneous_contributions,
        r=r,
    )

    expected_return = np.array([7.4, 5.4, 4.4, 0.4])

    np.testing.assert_allclose(actual_return, expected_return)

def test_public_goods_game_fitness_function_for_homoogeneous_contribution_and_heterogeneous_r():
    """
    Tests that public_goods_game_fitness_function correctly
    handles a homogeneous contribution with a heterogeneous r."""

    heterogeneous_contributions = np.array([2,2,2])
    state = np.array([1, 1, 1])
    r = np.array([1,1.5,2])

    actual_return = ludics.fitness_functions.public_goods_game_fitness_function(
        state=state,
        alpha=heterogeneous_contributions,
        r=r,
    )

    expected_return = np.array([0,1,2])

    np.testing.assert_allclose(actual_return, expected_return)

def test_pairwise_interaction_fitness_function_for_symmetric_prisoners_dilemma():
    """
    Tests that pairwise_interaction_fitness_function returns the correct
    value for a symmetric, standard prisoner's dilemma"""

    state = np.array([0,1,1,0,1])
    a = np.array([3 for _ in state])
    b = np.array([0 for _ in state])
    c = np.array([5 for _ in state])
    d = np.array([1 for _ in state])
    
    actual_value = ludics.fitness_functions.pairwise_interaction_fitness_function(state=state,a=a,b=b,c=c,d=d)
    expected_value = np.array([4, 6/4, 6/4, 4, 6/4])

    np.testing.assert_allclose(actual_value, expected_value)

def test_pairwise_interaction_fitness_function_for_asymmetric_game():
    """
    Tests that pairwise_interaction_fitness_function returns the correct value for
    an asymmetric game"""

    state = np.array([1,0,1,0])
    a = np.array([1,2,3,4])
    b = np.array([2,4,6,8])
    c = np.array([5,4,3,2])
    d = np.array([2,2,2,1])

    actual_value = ludics.fitness_functions.pairwise_interaction_fitness_function(state=state,a=a,b=b,c=c,d=d)
    
    expected_value = np.array([5/3, 10/3, 5, 5/3])

    np.testing.assert_allclose(actual_value, expected_value)
    