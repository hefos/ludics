import numpy as np
import sympy as sym

import ludics


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
