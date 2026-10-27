import numpy as np
import scipy.sparse
import sympy as sym

import ludics
import ludics.numerics


def test_exponentiate_for_a_float():
    """
    Tests that exponentiate returns the exponential of a float, and that the

    value is numeric rather than symbolic.
    """

    value = ludics.exponentiate(1.5)

    assert np.isclose(value, np.exp(1.5))
    assert isinstance(value, (float, np.floating))


def test_exponentiate_for_an_integer():
    """
    Tests that exponentiate returns the exponential of an integer, which is

    the case $e^0 = 1$ used whenever the selection intensity is zero.
    """

    assert np.isclose(ludics.exponentiate(0), 1.0)
    assert np.isclose(ludics.exponentiate(2), np.exp(2))


def test_exponentiate_for_a_negative_float():
    """
    Tests that exponentiate handles a negative exponent, which arises whenever

    a payoff difference runs against the transition being considered.
    """

    assert np.isclose(ludics.exponentiate(-3.25), np.exp(-3.25))


def test_exponentiate_for_an_array():
    """
    Tests that exponentiate maps over an array of floats elementwise and

    returns an array of floats, since the fitness of a population is a vector.
    """

    exponent = np.array([-1.0, 0.0, 1.0, 2.5])
    value = ludics.exponentiate(exponent)

    assert isinstance(value, np.ndarray)
    assert value.dtype == np.float64
    np.testing.assert_allclose(value, np.exp(exponent))


def test_exponentiate_for_a_symbol():
    """
    Tests that exponentiate returns a symbolic expression for a sympy symbol,

    which is what the closed form derivations of the transition probabilities

    rely upon.
    """

    beta = sym.Symbol("beta")
    value = ludics.exponentiate(beta)

    assert value == sym.E**beta
    assert value.free_symbols == {beta}


def test_exponentiate_for_a_symbolic_expression():
    """
    Tests that exponentiate carries an expression rather than only a bare

    symbol, since the exponent is usually a product of an intensity and a

    payoff difference.
    """

    beta = sym.Symbol("beta")
    delta = sym.Symbol("delta")
    value = ludics.exponentiate(beta * delta)

    assert value == sym.E ** (beta * delta)
    assert value.free_symbols == {beta, delta}


def test_exponentiate_for_an_array_of_symbols():
    """
    Tests that exponentiate maps over an array which holds symbols, which is

    the symbolic counterpart of a fitness vector.
    """

    first, second = sym.Symbol("a"), sym.Symbol("b")
    value = ludics.exponentiate(np.array([first, second], dtype=object))

    assert isinstance(value, np.ndarray)
    assert value.dtype == object
    assert list(value) == [sym.E**first, sym.E**second]


def test_exponentiate_agrees_with_sympy_on_floats():
    """
    Tests that the numeric branch agrees with the symbolic one it replaces.

    The maps previously raised sym.E to a float in every case, so the two must

    give the same number to within floating point error.
    """

    for exponent in (-5.0, -0.5, 0.0, 0.5, 5.0):
        assert np.isclose(
            float(ludics.exponentiate(exponent)), float(sym.E**exponent)
        )


def test_exponentiate_is_used_by_the_fitness_map_and_imitation_function():
    """
    Tests that the exponential fitness map and the Fermi imitation function

    both return numeric values for numeric input, and symbolic values for

    symbolic input.
    """

    numeric_fitness = ludics.exponential_fitness_map(
        fitness=np.array([1.0, 2.0]), selection_intensity=0.5
    )
    np.testing.assert_allclose(numeric_fitness, np.exp(np.array([0.5, 1.0])))
    assert numeric_fitness.dtype == np.float64

    assert np.isclose(
        ludics.fermi_imitation_function(delta=0.0, choice_intensity=1.0), 0.5
    )

    beta = sym.Symbol("beta")
    assert ludics.fermi_imitation_function(
        delta=1.0, choice_intensity=beta
    ).free_symbols == {beta}


def build_cycle_transition_matrix(number_of_states, forward_probability):
    """Returns a random walk on a cycle, whose stationary distribution is
    uniform and which is sparse enough to take the sparse path."""

    transition_matrix = np.zeros((number_of_states, number_of_states))
    for state in range(number_of_states):
        transition_matrix[state, (state + 1) % number_of_states] = forward_probability
        transition_matrix[state, (state - 1) % number_of_states] = (
            1 - forward_probability
        )

    return transition_matrix


reducible_transition_matrix = np.array(
    [
        [1.00, 0.00, 0.00, 0.00],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.00, 0.00, 0.00, 1.00],
    ]
)


def test_solve_stationary_distribution_for_a_dense_matrix():
    """
    Tests that solve_stationary_distribution returns the stationary

    distribution of a small chain, which is dense enough to be solved as one.
    """

    transition_matrix = np.array(
        [
            [0.3, 0.3, 0.3, 0.1],
            [0.0, 0.3, 0.2, 0.5],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.0, 0.0, 0.9],
        ]
    )

    np.testing.assert_allclose(
        ludics.solve_stationary_distribution(transition_matrix),
        np.array([0.11585366, 0.07317073, 0.16463415, 0.64634146]),
    )


def test_solve_stationary_distribution_for_a_sparse_matrix():
    """
    Tests that solve_stationary_distribution takes the sparse path for a chain

    with few transitions out of each state, and returns the same answer whether

    the matrix is given densely or as a scipy sparse array.

    The walk is periodic, so iterating would never settle, which the direct
    
    solve does not mind.
    """

    transition_matrix = build_cycle_transition_matrix(
        number_of_states=10, forward_probability=0.7
    )
    uniform = np.full(10, 0.1)

    np.testing.assert_allclose(
        ludics.solve_stationary_distribution(transition_matrix), uniform
    )
    np.testing.assert_allclose(
        ludics.solve_stationary_distribution(
            scipy.sparse.csr_array(transition_matrix)
        ),
        uniform,
    )


def test_solve_stationary_distribution_returns_none_when_not_unique():
    """
    Tests that solve_stationary_distribution returns None for a chain with two

    absorbing states, which has a family of stationary distributions rather

    than one. The dense and the sparse paths fail differently, the first by

    raising and the second by returning values which are not finite, so both

    are checked.
    """

    assert (
        ludics.solve_stationary_distribution(reducible_transition_matrix) is None
    )
    assert (
        ludics.solve_stationary_distribution(
            scipy.sparse.csr_array(reducible_transition_matrix)
        )
        is None
    )


def test_compute_steady_state_falls_back_to_iterating():
    """
    Tests that compute_steady_state iterates from the uniform distribution when

    the chain has no single stationary distribution and no initial distribution

    was given. Starting uniformly splits evenly between the two absorbing
    
    states.
    """

    np.testing.assert_allclose(
        ludics.compute_steady_state(reducible_transition_matrix),
        np.array([0.5, 0.0, 0.0, 0.5]),
        atol=10**-5,
    )


def test_solve_stationary_distribution_checks_the_answer(monkeypatch):
    """
    Tests that solve_stationary_distribution checks what the solver returns

    rather than trusting it, by giving it a solver which returns a vector that

    is not a stationary distribution. A singular system has no way to signal

    the failure other than the answer it gives back, so returning a wrong

    distribution quietly would be worse than returning None.
    """

    transition_matrix = np.array([[0.4, 0.6], [0.4, 0.6]])

    for bad_answer in (
        np.array([np.inf, -np.inf]),
        np.array([-0.5, 1.5]),
        np.array([0.4, 0.4]),
        np.array([0.9, 0.1]),
    ):
        monkeypatch.setattr(
            ludics.numerics.scipy.linalg,
            "solve",
            lambda *_, answer=bad_answer, **__: answer,
        )
        assert ludics.solve_stationary_distribution(transition_matrix) is None
