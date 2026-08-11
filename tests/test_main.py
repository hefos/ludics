import ludics
import numpy as np
import sympy as sym
import pytest

def test_compute_moran_transition_probability_for_trivial_fitness_function():
    """
    Tests whether the compute_moran_transition_probability

    works properly for a standard fitness function. Given two states

    (source and target, both numpy.arrays) and a trivial

    fitness function (returning 1 for all entries within the state),

    test that compute_moran_transition_probability returns the

    correct value. Here we see (0,1,0) -> (1,1,0) with a correct

    value of 1/9, and then we see a transition with Hamming distance

    2, correct value 0, and then a transition with Hamming distance

    0, correct value None."""

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    source = np.array((0, 1, 0))
    target = np.array((1, 1, 0))
    selection_intensity = np.full(shape=(3,3), fill_value=0.5)
    fitness_map = ludics.linear_fitness_map
    assert (
        ludics.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=trivial_fitness_function,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map
        )
        == 1 / 9
    )
    source = np.array((0, 1, 0))
    target = np.array((1, 1, 1))
    assert (
        ludics.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=trivial_fitness_function,
            selection_intensity=0.5,
            fitness_map=fitness_map
        )
        == 0
    )
    source = np.array((0, 0, 0))
    target = np.array((0, 0, 0))
    assert (
        ludics.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=trivial_fitness_function,
            selection_intensity=0.5,
            fitness_map=fitness_map
        )
        is None
    )


def test_compute_moran_transition_probability_for_specific_fitness_function():
    """
    Tests to see that the compute_moran_transition_probability

    function works correctly when the fitness function takes into account

    all entries within the state. Given two states (source and target, both numpy.arrays)

    and a specific fitness function (which returns the number of entries

    in the state sharing a type with a given entry (including itself)),

    test that compute_moran_transition_probability returns the

    correct value. Here we see (0,1,0) -> (1,1,0) with a correct

    value of 1/12, and then we see a transition with Hamming distance

    2, correct value 0, and then a transition with Hamming distance

    0, correct value None.

    An example for the fitness function can be seen as in the state

    f((0,0,1)) = (2, 2, 1)"""

    def fitness_function(state):
        return np.array([np.count_nonzero(state == _) for _ in state])

    source = np.array((0, 1, 0))
    target = np.array((1, 1, 0))
    selection_intensity=np.full(shape=(3,3), fill_value=0.5)
    fitness_map = ludics.linear_fitness_map
    assert ludics.compute_moran_transition_probability(
        source=source,
        target=target,
        fitness_function=fitness_function,
        selection_intensity=selection_intensity,
        fitness_map=fitness_map
    ) == 1/12
    source = np.array((0, 1, 1))
    target = np.array((0, 0, 0))
    assert (
        ludics.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=fitness_function,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map
        )
        == 0
    )
    source = np.array((1, 1, 0))
    target = np.array((1, 1, 0))
    assert (
        ludics.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=fitness_function,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map
        )
        is None
    )


def test_compute_moran_transition_probability_for_ordered_fitness_function():
    """
    Tests to see that the compute_moran_transition_probability

    function works correctly when the fitness function takes into account

    the position of entries within the state, both in relation to an entry and

    the position of an entry itself. Given two states (source and target, both numpy.arrays)

    and a specific fitness function (which for a given entry in position i

    (indexed from 0) will return the number of prior (self-included) entries

    with the same value as the entry + (i % 2)), tests that

    compute_moran_transition_probability returns the correct value. Here we see (0,1,0) -> (1,1,0)

    with an expected value of 2/15, and then we see a transition with Hamming

    distance 2, correct value 0, and then a transition with Hamming distance

    0, correct value None.

    An example for the fitness function can be seen as in the state

    f((0,0,1)) = (1, 3, 1)"""

    def ordered_fitness_function(state):
        fitness = np.array([0 for _ in state])
        zero_encountered = 0
        one_encountered = 0
        for position, value in enumerate(state):
            if value == 0:
                zero_encountered += 1
                fitness[position] = zero_encountered + (position % 2)
            else:
                one_encountered += 1
                fitness[position] = one_encountered + (position % 2)
        return fitness

    source = np.array((0, 1, 0))
    target = np.array((1, 1, 0))
    selection_intensity=np.full(shape=(3,3), fill_value=1)
    fitness_map = ludics.linear_fitness_map
    assert ludics.compute_moran_transition_probability(
        source=source,
        target=target,
        fitness_function=ordered_fitness_function,
        selection_intensity=selection_intensity,
        fitness_map=fitness_map
    ) == 2 / 15
    source = np.array((0, 1, 1))
    target = np.array((0, 0, 0))
    assert (
        ludics.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=ordered_fitness_function,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map
        )
        == 0
    )
    source = np.array((1, 1, 0))
    target = np.array((1, 1, 0))
    assert (
        ludics.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=ordered_fitness_function,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map
        )
        is None
    )


def test_compute_moran_transition_probability_for_symbolic_fitness_function():
    """
    Tests for whether compute_transition_prbability returns the correct

    value for a fitness function which works symbolically.

    Given two states (source and target, both numpy.arrays) and a

    symbolic fitness function (i.e, replacing 1 with x and 0 with y, via

    sympy), tests that compute_moran_transition_probability returns the correct

    value. tests (0,1,0) -> (1,1,0), with correct value

    x / ((3 * x) + (6 * y)), then transitions with Hamming distances

    2 and 0, with correct values 0 and None respectively."""

    def symbolic_fitness_function(state):
        return np.array(
            [
                sym.Symbol("x") if individual == 1 else sym.Symbol("y")
                for individual in state
            ]
        )

    source = np.array((0, 1, 0))
    target = np.array((1, 1, 0))
    x = sym.Symbol("x")
    y = sym.Symbol("y")
    epsilon = sym.Symbol("\epsilon")
    selection_intensity=np.full(shape=(3,3), fill_value=epsilon)
    fitness_map = ludics.linear_fitness_map
    assert sym.simplify(ludics.compute_moran_transition_probability(
        source=source,
        target=target,
        fitness_function=symbolic_fitness_function,
        selection_intensity=selection_intensity,
        fitness_map=fitness_map
    )) == sym.simplify((1 - epsilon + epsilon * x) / ((3 * (1 - epsilon + epsilon * x)) + (6 * (1 - epsilon + epsilon * y))))
    source = np.array((0, 1, 1))
    target = np.array((0, 0, 0))
    assert (
        ludics.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=symbolic_fitness_function,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map
        )
        == 0
    )
    source = np.array((1, 1, 0))
    target = np.array((1, 1, 0))
    assert (
        ludics.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=symbolic_fitness_function,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map
        )
        is None
    )
    source = np.array((0, 1))
    target = np.array((0, 0))
    selection_intensity=np.full(shape=(2,2), fill_value=epsilon)
    assert ludics.compute_moran_transition_probability(
        source=source,
        target=target,
        fitness_function=symbolic_fitness_function,
        selection_intensity=selection_intensity,
        fitness_map=fitness_map
    ) == (1 - epsilon + epsilon * y) / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y))

    source = np.array((0, 1))
    target1 = np.array((0, 0))
    target2 = np.array((1, 1))
    assert 1 - ludics.compute_moran_transition_probability(
        source=source,
        target=target1,
        fitness_function=symbolic_fitness_function,
        selection_intensity=selection_intensity,
        fitness_map=fitness_map
    ) - ludics.compute_moran_transition_probability(
        source=source,
        target=target2,
        fitness_function=symbolic_fitness_function,
        selection_intensity=selection_intensity,
        fitness_map=fitness_map
    ) == (
        1
        - ((1 - epsilon + epsilon * y) / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y)))
        - (1 - epsilon + epsilon * x) / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y))
    )


def test_compute_moran_transition_probability_for_kwargs_fitness_function():
    """
    tests the compute_moran_transition_probability function for

    a fitness function which takes kwargs
    """

    def kwargs_fitness_function(state, c, r):
        return np.array([c if individual == 1 else r for individual in state])

    source = np.array((0, 1, 0))
    target = np.array((1, 1, 0))
    c = 2
    r = 3
    selection_intensity=np.full(shape=(3,3), fill_value=0.1)
    fitness_map = ludics.linear_fitness_map
    expected_transition_probability = 0.1047619048

    np.testing.assert_almost_equal(
        ludics.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=kwargs_fitness_function,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map,
            c=c,
            r=r,
        ), expected_transition_probability
    )


def test_generate_state_space_for_N_eq_3_and_k_eq_2():
    """
    Given a value of $N$: the number of individuals and a value of $k$: the
    number of types generate $S = [1, ..., k] ^ N$.

    This tests this for N = 3, k = 2.
    """
    k = 2
    N = 3
    expected_state_space = np.array(
        [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (0, 0, 0),
            (1, 1, 1),
        ]
    )
    obtained_state_space = ludics.get_state_space(N=N, k=k)
    np.testing.assert_array_equal(
        sorted(tuple(x) for x in obtained_state_space),
        sorted(tuple(x) for x in expected_state_space),
    )


def test_generate_state_space_for_N_eq_3_and_k_eq_1():
    """
    Given a value of $N$: the number of individuals and a value of $k$: the
    number of types generate $S = [1, ..., k] ^ N$.

    This tests this for N = 3, k = 1.
    """
    k = 1
    N = 3
    expected_state_space = [
        (0, 0, 0),
    ]
    obtained_state_space = ludics.get_state_space(N=N, k=k)
    np.testing.assert_allclose(
        sorted(expected_state_space), sorted(obtained_state_space)
    )


def test_generate_state_space_for_N_eq_1_and_k_eq_3():
    """
    Given a value of $N$: the number of individuals and a value of $k$: the
    number of types generate $S = [1, ..., k] ^ N$.

    This tests this for N = 1, k = 3.
    """
    k = 3
    N = 1
    expected_state_space = [
        (0,),
        (1,),
        (2,),
    ]
    obtained_state_space = ludics.get_state_space(N=N, k=k)
    np.testing.assert_allclose(
        sorted(expected_state_space), sorted(obtained_state_space)
    )


def test_generate_transition_matrix_for_trivial_fitness_function():
    """
    Tests whether generate_transition_matrix returns the correct matrix

    for a trivial fitness function an a state space N = 3, K = 2.
    """

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])
    selection_intensity=np.full(shape=(3,3), fill_value=0.1)
    fitness_map = ludics.linear_fitness_map
    state_space = np.array(
        [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (0, 0, 0),
            (1, 1, 1),
        ]
    )
    expected_transition_matrix = np.array(
        [
            [0.55555556, 0, 0, 0.11111111, 0.11111111, 0, 0.22222222, 0],
            [0, 0.55555556, 0, 0.11111111, 0, 0.11111111, 0.22222222, 0],
            [0, 0, 0.55555556, 0, 0.11111111, 0.11111111, 0.22222222, 0],
            [0.11111111, 0.11111111, 0, 0.55555556, 0, 0, 0, 0.22222222],
            [0.11111111, 0, 0.11111111, 0, 0.55555556, 0, 0, 0.22222222],
            [0, 0.11111111, 0.11111111, 0, 0, 0.55555556, 0, 0.22222222],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    np.testing.assert_array_almost_equal(
        ludics.generate_transition_matrix(
            state_space=state_space,
            fitness_function=trivial_fitness_function,
            compute_transition_probability=ludics.compute_moran_transition_probability,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map
        ),
        expected_transition_matrix,
    )


def test_generate_transition_matrix_for_ordered_fitness_function():
    """
    Tests whether generate_transition_matrix returns the correct matrix

    for a fitness function based on order (see test_compute_transition_matrix_for_ordered_fitness_function

    for a description of the fitness function) and a state space N = 3, K = 2.
    """

    def ordered_fitness_function(state):
        fitness = np.array([0 for _ in state])
        zero_encountered = 0
        one_encountered = 0
        for position, value in enumerate(state):
            if value == 0:
                zero_encountered += 1
                fitness[position] = zero_encountered + (position % 2)
            else:
                one_encountered += 1
                fitness[position] = one_encountered + (position % 2)
        return fitness
    fitness_map = ludics.linear_fitness_map
    state_space = np.array(
        [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (0, 0, 0),
            (1, 1, 1),
        ]
    )

    selection_intensity=np.full(shape=(3,3), fill_value=0.5)

    expected_transition_matrix = np.array(
        [
            [7/12, 0, 0, 1/12, 1/12, 0, 3/12, 0],
            [0, 13/24, 0, 1/8, 0, 1/8, 5/24, 0],
            [0, 0, 7/12, 0, 1/12, 1/12, 3/12, 0],
            [1/12, 1/12, 0, 7/12, 0, 0, 0, 3/12],
            [1/8, 0, 1/8, 0, 13/24, 0, 0, 5/24],
            [0, 1/12, 1/12, 0, 0, 7/12, 0, 3/12],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    np.testing.assert_allclose(
        ludics.generate_transition_matrix(
            state_space=state_space,
            fitness_function=ordered_fitness_function,
            compute_transition_probability=ludics.compute_moran_transition_probability,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map
        ),
        expected_transition_matrix,
    )


def test_generate_transition_matrix_for_different_state_space():
    """
    Tests whether generate_transition_matrix returns the correct matrix

    for a state space N = 2, K = 3.
    """

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    state_space = np.array(
        [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2)]
    )
    selection_intensity=np.full(shape=(2,2), fill_value=0.5)
    fitness_map = ludics.linear_fitness_map
    expected_transition_matrix = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.25, 0.5, 0, 0.25, 0, 0, 0, 0, 0],
            [0.25, 0, 0.5, 0.25, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0.25, 0, 0, 0, 0.5, 0, 0, 0, 0.25],
            [0.25, 0, 0, 0, 0, 0.5, 0, 0, 0.25],
            [0, 0, 0, 0.25, 0, 0, 0.5, 0, 0.25],
            [0, 0, 0, 0.25, 0, 0, 0, 0.5, 0.25],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    np.testing.assert_allclose(
        ludics.generate_transition_matrix(
            state_space=state_space,
            fitness_function=trivial_fitness_function,
            compute_transition_probability=ludics.compute_moran_transition_probability,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map
        ),
        expected_transition_matrix,
    )


def test_generate_transition_matrix_for_symbolic_fitness_function():
    """
    Tests whether generate_transition_matrix returns the correct matrix

    for a symbolic fitness function function based on (see test_compute_transition_matrix_for_symbolic_fitness_function

    for a description of the fitness function) and a smaller state space N = 2, K = 2.
    """

    def symbolic_fitness_function(state):
        return np.array(
            [
                sym.Symbol("x") if individual == 1 else sym.Symbol("y")
                for individual in state
            ]
        )

    state_space = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
    fitness_map = ludics.linear_fitness_map
    x = sym.Symbol("x")
    y = sym.Symbol("y")
    epsilon = sym.Symbol("\epsilon")
    selection_intensity=np.full(shape=(2,2), fill_value=epsilon)
    expected_transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [
                (1 - epsilon + epsilon * y) / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y)),
                (
                    1
                    - (
                        (1 - epsilon + epsilon * y)
                        / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y))
                    )
                    - (1 - epsilon + epsilon * x)
                    / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y))
                ),
                0,
                (1 - epsilon + epsilon * x) / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y)),
            ],
            [
                (1 - epsilon + epsilon * y) / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y)),
                0,
                (
                    1
                    - (
                        (1 - epsilon + epsilon * y)
                        / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y))
                    )
                    - (1 - epsilon + epsilon * x)
                    / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y))
                ),
                (1 - epsilon + epsilon * x) / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y)),
            ],
            [0, 0, 0, 1],
        ]
    )
    np.testing.assert_array_almost_equal(
        ludics.generate_transition_matrix(
            state_space=state_space,
            fitness_function=symbolic_fitness_function,
            compute_transition_probability=ludics.compute_moran_transition_probability,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map
        ),
        expected_transition_matrix,
    )


def test_generate_transition_matrix_with_individual_to_action_mutation_probability_moran():
    """
    Tests that the generate_transition_matrix function works properly for the
    case where we have a non-zero mutation vector in the Moran process"""

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    state_space = ludics.get_state_space(N=2, k=2)

    individual_to_action_mutation_probability = np.array([[0.2, 0.15], [0.1, 0.05]])
    fitness_map=ludics.linear_fitness_map
    selection_intensity=np.full(shape=(2,2), fill_value=0)
    actual_transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=trivial_fitness_function,
        compute_transition_probability=ludics.compute_moran_transition_probability,
        selection_intensity=selection_intensity,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
        fitness_map=fitness_map
    )

    expected_transition_matrix = np.array(
        [
            [0.9, 0.025, 0.075, 0],
            [0.2625, 0.5, 0, 0.2375],
            [0.2625, 0, 0.5, 0.2375],
            [0, 0.1, 0.05, 0.85],
        ]
    )

    np.testing.assert_array_almost_equal(
        actual_transition_matrix, expected_transition_matrix
    )


def test_generate_transition_matrix_with_individual_to_action_mutation_probability_fermi():
    """
    Tests that the generate_transition_matrix function works properly for the
    case where we have a non-zero mutation vector in Fermi imitation dynamics"""

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    state_space = ludics.get_state_space(N=2, k=2)

    individual_to_action_mutation_probability = np.array([[0.01, 0.15], [0.05, 0.2]])

    choice_intensity=np.full(shape=(2,2), fill_value=1)

    actual_transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=trivial_fitness_function,
        compute_transition_probability=ludics.compute_fermi_transition_probability,
        choice_intensity=choice_intensity,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
    )

    expected_transition_matrix = np.array(
        [
            [0.825, 0.1, 0.075, 0],
            [0.2125, 1 - 0.2125 - 0.285, 0, 0.285],
            [0.215, 0, 1 - 0.215 - 0.2875, 0.2875],
            [0, 0.005, 0.025, 0.97],
        ]
    )

    np.testing.assert_array_almost_equal(
        actual_transition_matrix, expected_transition_matrix
    )


def test_generate_transition_matrix_with_individual_to_action_mutation_probability_imispection():
    """
    Tests that the generate_transition_matrix function works properly for the
    case where we have a non-zero mutation vector in introspective imitation dynamics"""

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    state_space = ludics.get_state_space(N=2, k=2)

    individual_to_action_mutation_probability = np.array([[0.01, 0.1], [0.15, 0.2]])
    fitness_map = ludics.linear_fitness_map
    choice_intensity = np.full(shape=(2,2), fill_value=1)
    selection_intensity = np.full(shape=(2,2), fill_value=0)

    actual_transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=trivial_fitness_function,
        compute_transition_probability=ludics.compute_introspective_imitation_transition_probability,
        choice_intensity=choice_intensity,
        selection_intensity=selection_intensity,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
        fitness_map=fitness_map
    )

    expected_transition_matrix = np.array(
        [
            [0.85, 0.1, 0.05, 0],
            [0.15625, 1 - 0.15625 - 0.16125, 0, 0.16125],
            [0.11625, 0, 1 - 0.11625 - 0.18125, 0.18125],
            [0, 0.005, 0.075, 1 - 0.005 - 0.075],
        ]
    )

    np.testing.assert_array_almost_equal(
        actual_transition_matrix, expected_transition_matrix
    )


def test_generate_transition_matrix_with_individual_to_action_mutation_probability_introspection():
    """
    Tests that the generate_transition_matrix function works properly for the
    case where we have a non-zero mutation vector in introspection dynamics"""

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    state_space = ludics.get_state_space(N=2, k=2)

    individual_to_action_mutation_probability = np.array([[0.1, 0.2], [0.3, 0.4]])

    choice_intensity = np.full(shape=(2,2), fill_value=1)

    actual_transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=trivial_fitness_function,
        compute_transition_probability=ludics.compute_introspection_transition_probability,
        choice_intensity=choice_intensity,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
        number_of_strategies=2,
    )

    expected_transition_matrix = np.array(
        [
            [1 - 0.275 - 0.275, 0.275, 0.275, 0],
            [0.225, 1 - 0.225 - 0.275, 0, 0.275],
            [0.225, 0, 1 - 0.225 - 0.275, 0.275],
            [0, 0.225, 0.225, 1 - 0.225 - 0.225],
        ]
    )

    np.testing.assert_array_almost_equal(
        actual_transition_matrix, expected_transition_matrix
    )


def test_generate_transition_matrix_for_symbolic_fitness_function_with_mutation():
    """
    Tests whether generate_transition_matrix returns the correct matrix

    for a symbolic fitness function function and symbolic mutation probabilities.
    """

    def symbolic_fitness_function(state, **kwargs):
        return np.array(
            [
                sym.Symbol("x") if individual == 1 else sym.Symbol("y")
                for individual in state
            ]
        )

    state_space = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
    fitness_map=ludics.linear_fitness_map
    x = sym.Symbol("x")
    y = sym.Symbol("y")
    epsilon = sym.Symbol("\epsilon")
    selection_intensity=np.full(shape=(2,2), fill_value=epsilon)
    mu_11 = sym.Symbol("\mu_{11}")
    mu_12 = sym.Symbol("\mu_{12}")
    mu_21 = sym.Symbol("\mu_{21}")
    mu_22 = sym.Symbol("\mu_{22}")
    individual_to_action_mutation_probability = np.array(
        [[mu_11, mu_12], [mu_21, mu_22]]
    )

    mu_sum_p1 = mu_11 + mu_12
    mu_sum_p2 = mu_21 + mu_22
    actual_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=symbolic_fitness_function,
        compute_transition_probability=ludics.compute_moran_transition_probability,
        selection_intensity=selection_intensity,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
        fitness_map=fitness_map
    )

    expected_transition_matrix = np.array(
        [
            [1 - mu_22 / 2 - mu_12 / 2, mu_22 / 2, mu_12 / 2, 0],
            [
                ((1 - epsilon + epsilon * y) / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y)))
                * (1 - mu_sum_p2)
                + mu_21 / 2,
                (
                    1
                    - (
                        (
                            (1 - epsilon + epsilon * y)
                            / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y))
                        )
                        * (1 - mu_sum_p2)
                        + mu_21 / 2
                    )
                    - (
                        (
                            (1 - epsilon + epsilon * x)
                            / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y))
                        )
                        * (1 - mu_sum_p1)
                        + mu_12 / 2
                    )
                ),
                0,
                ((1 - epsilon + epsilon * x) / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y)))
                * (1 - mu_sum_p1)
                + mu_12 / 2,
            ],
            [
                ((1 - epsilon + epsilon * y) / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y)))
                * (1 - mu_sum_p1)
                + mu_11 / 2,
                0,
                (
                    1
                    - (
                        (
                            (1 - epsilon + epsilon * y)
                            / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y))
                        )
                        * (1 - mu_sum_p1)
                        + mu_11 / 2
                    )
                    - (
                        (
                            (1 - epsilon + epsilon * x)
                            / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y))
                        )
                        * (1 - mu_sum_p2)
                        + mu_22 / 2
                    )
                ),
                ((1 - epsilon + epsilon * x) / (2 * (1 - epsilon + epsilon * x) + 2 * (1 - epsilon + epsilon * y)))
                * (1 - mu_sum_p2)
                + mu_22 / 2,
            ],
            [0, mu_11 / 2, mu_21 / 2, 1 - mu_11 / 2 - mu_21 / 2],
        ],
        dtype=object,
    )

    np.testing.assert_array_equal(
        sym.simplify(expected_transition_matrix - actual_matrix),
        sym.zeros(actual_matrix.shape[0], actual_matrix.shape[1]),
    )


def test_generate_transition_matrix_for_kwargs_fitness_function():
    """
    tests the generate_transition_matrix function for

    a fitness function which takes kwargs
    """

    def kwargs_fitness_function(state, c, r):
        return np.array([c if individual == 1 else r for individual in state])

    state_space = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    c = 1
    r = 4
    expected_transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [7/22, 1/2, 0, 2/11],
            [7/22, 0, 1/2, 2/11],
            [0, 0, 0, 1],
        ]
    )
    fitness_map=ludics.linear_fitness_map
    selection_intensity=np.full(shape=(2,2), fill_value=0.25)
    np.testing.assert_array_almost_equal(
        expected_transition_matrix,
        ludics.generate_transition_matrix(
            state_space=state_space,
            fitness_function=kwargs_fitness_function,
            compute_transition_probability=ludics.compute_moran_transition_probability,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map,
            c=c,
            r=r,
        ),
    )


def test_get_absorbing_state_index_for_N_eq_2_k_eq_4():
    """
    Tests that get_absorbing_state_index correctly identifies

    the absorbing states in a standard state space"""

    state_space = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 0],
            [1, 1],
            [1, 2],
            [1, 3],
            [2, 0],
            [2, 1],
            [2, 2],
            [2, 3],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
        ]
    )

    expected_absorbing_states = np.array([0, 5, 10, 15])

    np.testing.assert_array_equal(
        expected_absorbing_states,
        ludics.get_absorbing_state_index(state_space=state_space),
    )


def test_get_absorbing_state_index_for_no_absorbing_states():
    """
    Tests that get_absorbing_state_index correctly identifies

    that there are no absorbing states in a given state

    space"""

    non_absorbing_state_space = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 0],
            [1, 2],
            [1, 3],
            [2, 0],
            [2, 1],
            [2, 3],
            [3, 0],
            [3, 1],
            [3, 2],
        ]
    )

    expected_absorbing_states = None

    assert expected_absorbing_states == ludics.get_absorbing_state_index(
        state_space=non_absorbing_state_space
    )


def test_get_absorbing_state_index_for_symbolic_state_space():
    """Tests the get_absorbing_state_index function for
    a symbolic state space."""

    A = sym.Symbol("A")
    B = sym.Symbol("B")

    symbolic_state_space = np.array(
        [
            [A, B],
            [A, A],
            [B, B],
            [B, A],
        ]
    )

    expected_absorbing_states = np.array([1, 2])
    np.testing.assert_array_equal(
        expected_absorbing_states,
        ludics.get_absorbing_state_index(state_space=symbolic_state_space),
    )


def test_get_absorbing_states_for_standard_state_space():
    """Tests the get_absorbing_states function

    for a standard state space"""

    state_space = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 0],
            [1, 1],
            [1, 2],
            [1, 3],
            [2, 0],
            [2, 1],
            [2, 2],
            [2, 3],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
        ]
    )

    expected_absorbing_states = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
        ]
    )

    np.testing.assert_array_equal(
        expected_absorbing_states,
        ludics.get_absorbing_states(state_space=state_space),
    )


def test_get_absorbing_states_for_no_absorbing_states():
    """
    Tests that get_absorbing_states correctly identifies

    that there are no absorbing states in a given state

    space"""

    non_absorbing_state_space = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 0],
            [1, 2],
            [1, 3],
            [2, 0],
            [2, 1],
            [2, 3],
            [3, 0],
            [3, 1],
            [3, 2],
        ]
    )

    assert (
        ludics.get_absorbing_states(state_space=non_absorbing_state_space) is None
    )


def test_get_absorbing_states_for_symbolic_state_space():
    """Tests the get_absorbing_states function for
    a symbolic state space."""

    A = sym.Symbol("A")
    B = sym.Symbol("B")

    symbolic_state_space = np.array(
        [
            [A, B],
            [A, A],
            [B, B],
            [B, A],
        ]
    )
    expected_absorbing_states = np.array(
        [
            [A, A],
            [B, B],
        ]
    )

    np.testing.assert_array_equal(
        expected_absorbing_states,
        ludics.get_absorbing_states(state_space=symbolic_state_space),
    )

def test_extract_Q_for_numeric_transition_matrix():
    """
    Tests the extract_Q function for a transition matrix with numeric values

    and no symbolic values. We take N=2 and K=2"""

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 0.25, 0.3, 0.45],
            [0, 0, 1, 0],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )

    expected_Q = np.array(
        [
            [0.25, 0.45],
            [0.25, 0.25],
        ]
    )

    np.testing.assert_array_equal(
        expected_Q, ludics.extract_Q(transition_matrix=transition_matrix)
    )


def test_extract_Q_for_symbolic_transition_matrix():
    """
    Tests the extract_Q function for a transition matrix with just symbolic values. We take N=2 and K=2
    """

    A = sym.Symbol("A")
    B = sym.Symbol("B")
    C = sym.Symbol("C")

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, A, B, B],
            [0, 0, 1, 0],
            [C + A, C, B, C + A],
        ]
    )

    expected_Q = np.array(
        [
            [A, B],
            [C, C + A],
        ]
    )

    np.testing.assert_array_equal(
        expected_Q, ludics.extract_Q(transition_matrix=transition_matrix)
    )


def test_extract_Q_for_mixed_transition_matrix():
    """
    Tests the extract_Q function for a transition matrix with symbolic values

    and numeric values. We take N=2 and K=2"""

    A = sym.Symbol("A")
    B = sym.Symbol("B")
    C = sym.Symbol("C")

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, A, B, B / 3],
            [0, 0, 1, 0],
            [C + A, 0.5, B, C + 0.2],
        ]
    )

    expected_Q = np.array(
        [
            [A, B / 3],
            [0.5, C + 0.2],
        ]
    )

    np.testing.assert_array_equal(
        expected_Q, ludics.extract_Q(transition_matrix=transition_matrix)
    )


def test_extract_R_numerical_for_numeric_transition_matrix():
    """
    Tests the extract_R_numerical function for a transition matrix with numeric

    values and no symbolic values. We take N=2 and K=2"""

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 0.25, 0.3, 0.45],
            [0, 0, 1, 0],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )

    expected_R = np.array(
        [
            [0, 0.3],
            [0.25, 0.25],
        ]
    )

    np.testing.assert_array_equal(
        expected_R, ludics.extract_R_numerical(transition_matrix=transition_matrix)
    )


def test_extract_R_symbolic_for_mixed_transition_matrix():
    """
    Tests the extract_R_symbolic function for a transition matrix with symbolic values

    and numeric values. We take N=2 and K=2"""

    A = sym.Symbol("A")
    B = sym.Symbol("B")
    C = sym.Symbol("C")

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0.5, A, B, B / 3],
            [0, 0, 1, 0],
            [C + A, 0.2, 0.3, C],
        ]
    )

    expected_R = np.array(
        [
            [0.5, B],
            [C + A, 0.3],
        ]
    )

    np.testing.assert_array_equal(
        expected_R, ludics.extract_R_symbolic(transition_matrix=transition_matrix)
    )


def test_extract_R_symbolic_for_purely_symbolic_transition_matrix():
    """
    Tests the extract_Q function for a transition matrix with symbolic values

    and no numeric values. We take N=2 and K=2"""

    A = sym.Symbol("A")
    B = sym.Symbol("B")
    C = sym.Symbol("C")

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, A, B, B],
            [0, 0, 1, 0],
            [C + A, C, B, C + A],
        ]
    )

    expected_R = np.array(
        [
            [0, B],
            [C + A, B],
        ]
    )

    np.testing.assert_array_equal(
        expected_R, ludics.extract_R_symbolic(transition_matrix=transition_matrix)
    )


def test_compute_absorption_matrix_for_numeric_transition_matrix():
    """
    Tests the compute_absorption_matrix function for an entirely

    numeric transition matrix"""

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0.25, 0.75, 0],
            [0.3, 0, 0, 0.7],
        ]
    )

    expected_absorption_matrix = np.array([[0, 1], [1, 0]])

    np.testing.assert_array_almost_equal(
        expected_absorption_matrix,
        ludics.compute_absorption_matrix(transition_matrix=transition_matrix),
    )


def test_calculate_absorption_matrix_for_symbolic_transition_matrix():
    """
    Tests the calculate_absorption_matrix function for an symbolic

    transition matrix"""

    A = sym.Symbol("A")
    B = sym.Symbol("B")
    C = sym.Symbol("C")

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, A, B, 0],
            [C, C, 0, 0],
        ]
    )

    expected_absorption_matrix = np.array([[0, A / (1 - B)], [C, C]])

    np.testing.assert_array_almost_equal(
        expected_absorption_matrix,
        ludics.calculate_absorption_matrix(transition_matrix=transition_matrix),
    )


def test_compute_absorption_matrix_for_standard_transition_matrix():
    """
    Tests the compute_absorption_matrix function for an entirely

    numeric transition matrix"""

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0.25, 0.75, 0],
            [0.3, 0, 0, 0.7],
        ]
    )

    expected_absorption_matrix = np.array([[0, 1], [1, 0]])

    np.testing.assert_array_almost_equal(
        expected_absorption_matrix,
        ludics.compute_absorption_matrix(transition_matrix=transition_matrix),
    )


def test_calculate_absorption_matrix_for_standard_transition_matrix():
    """
    Tests the calculate_absorption_matrix function for a standard
    symbolic transition matrix"""

    A = sym.Symbol("A")
    B = sym.Symbol("B")
    C = sym.Symbol("C")

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, A, B, 0],
            [C, C, 0, 0],
        ]
    )

    expected_absorption_matrix = np.array([[0, A / (1 - B)], [C, C]])

    np.testing.assert_array_almost_equal(
        expected_absorption_matrix,
        ludics.calculate_absorption_matrix(transition_matrix=transition_matrix),
    )


def test_generate_absorption_matrix_functions_accuracy_for_r_values():
    """Tests that the equations generated by the symbolic

    generate_absorption_matrix function will give the correct value for various

    r values"""

    def public_goods_fitness_function(state, alpha, r):
        number_of_contributors = state.sum()
        public_good = r * alpha * (number_of_contributors) / (len(state))
        payoff = np.array([public_good - alpha * x for x in state])
        return payoff

    r = sym.Symbol("r")
    alpha = sym.Symbol(r"$\alpha$")
    fitness_map=ludics.linear_fitness_map
    r_test_values = np.array([1.5, 1.75, 2, 2.25, 2.5])

    expected_results = [
        1/6,
        3/14,
        1/4,
        5/18,
        3/10
    ]

    state_space = ludics.get_state_space(N=2, k=2)
    selection_intensity=np.full(shape=(2,2), fill_value=0.5)
    transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=public_goods_fitness_function,
        compute_transition_probability=ludics.compute_moran_transition_probability,
        selection_intensity=selection_intensity,
        fitness_map=fitness_map,
        r=r,
        alpha=alpha,
    )

    absorption_matrix = ludics.calculate_absorption_matrix(transition_matrix)

    symbolic_expression = sym.lambdify(
        (r, alpha), sym.Matrix(absorption_matrix)[0, 1], "numpy"
    )

    obtained_results = symbolic_expression(r_test_values, 2)

    np.testing.assert_array_almost_equal(expected_results, obtained_results)


def test_calculate_absorption_matrix_for_5_by_5_symbolic_transition_matrix():
    """
    Tests the calculate_absorption_matrix function for a 5x5 symbolic

    transition matrix"""

    A = sym.Symbol("A")
    B = sym.Symbol("B")
    C = sym.Symbol("C")
    D = sym.Symbol("D")

    transition_matrix = np.array(
        [
            [1, 0, 0, 0, 0],
            [A, 1 / 3, B, 0, 0],
            [0, A, 0, C, 0],
            [0, 0, C, D, 1 / 3],
            [0, 0, 0, 0, 1],
        ]
    )

    Q = sym.Matrix(np.array([[1 / 3, B, 0], [A, 0, C], [0, C, D]]))

    identity = sym.Matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    R = sym.Matrix(np.array([[A, 0], [0, 0], [0, 1 / 3]]))

    expected_absorption_matrix = ((identity - Q) ** -1) * R

    obtained_absorption_matrix = ludics.calculate_absorption_matrix(
        transition_matrix=transition_matrix
    )

    zero_matrix = sym.Matrix(np.zeros((3, 2)))

    np.testing.assert_array_almost_equal(
        expected_absorption_matrix - obtained_absorption_matrix, zero_matrix
    )


def test_compute_steady_state_for_trivial_transition_matrix():
    """
    Tests compute_steady_state for a trivial transition matrix
    """

    numeric_matrix = np.array([[0.4, 0.6], [0.4, 0.6]])

    expected_numeric_output = np.array([0.4, 0.6])

    np.testing.assert_allclose(
        expected_numeric_output, ludics.compute_steady_state(numeric_matrix)
    )


def test_compute_steady_state_for_absorbing_transition_matrix():
    """
    Tests compute_steady_state for an absorbing transition matrix
    """

    numeric_matrix = np.array(
        [[1, 0, 0, 0], [0.3, 0.6, 0, 0.1], [0, 0.3, 0.4, 0.3], [0.2, 0.1, 0.1, 0.6]]
    )

    expected_numeric_output = np.array([1, 0, 0, 0])

    np.testing.assert_allclose(
        expected_numeric_output,
        ludics.compute_steady_state(numeric_matrix),
        rtol=1**-5,
    )


def test_calculate_steady_state_for_trivial_transition_matrix():
    """
    Tests whether the calculate_steady_state function returns the correct matrix for
    a 2x2 transition matrix with the simple form [[p, 1-p], [p,1-p]]"""

    p = sym.Symbol("p")
    q = sym.Symbol("q")

    symbolic_matrix = sym.Matrix(
        [[0.5 + p + q, 0.5 - p - q], [0.5 + p + q, 0.5 - p - q]]
    )

    expected_symbolic_output = sym.Matrix([[0.5 + p + q, 0.5 - p - q]])

    assert expected_symbolic_output - ludics.calculate_steady_state(
        symbolic_matrix
    ) == sym.zeros(rows=1, cols=2)


def test_calculate_steady_state_for_absorbing_symbolic_transition_matrix():
    """
    Tests whether the calculate_steady_state function still returns the
    correct value if the matrix passed to it is absorbing and symbolic. It
    should return a steady state corresponding to just the absorbing state of
    the transition matrix"""

    p = sym.Symbol("p")

    transition_matrix = np.array([[p, 1 - p - 0.1, 0.1], [0, 1, 0], [0.6, 0.2, 0.2]])

    expected_output = sym.Matrix([[0, 1, 0]])

    assert expected_output - ludics.calculate_steady_state(
        transition_matrix
    ) == sym.zeros(rows=1, cols=3)


def test_fermi_imitation_function_for_numeric_value():
    """
    Tests whether the fermi_imitation_function returns the desired value for
    numeric values of delta and selection_intesntiy"""

    delta = 3
    choice_intensity = 0.5

    expected_fermi_value = 0.1824255238

    actual_fermi_value = ludics.fermi_imitation_function(
        delta=delta, choice_intensity=choice_intensity
    )

    np.testing.assert_almost_equal(expected_fermi_value, actual_fermi_value)


def test_fermi_imitation_function_for_symbolic_value():
    """
    Tests whether the fermi_imitation_function returns the desired expression
    for symbolic values of delta and selection_intensity"""

    delta = sym.Symbol("Delta")
    choice_intensity = sym.Symbol("beta")

    expected_fermi_value = 1 / (1 + sym.E ** (delta * choice_intensity))

    actual_fermi_value = ludics.fermi_imitation_function(
        delta=delta, choice_intensity=choice_intensity
    )

    assert expected_fermi_value == actual_fermi_value


def test_compute_fermi_transition_probability_for_trivial_fitness_function():
    """
    Tests whether the compute_fermi_transition_probability function returns the
    desired value for a trivial fitness function"""

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    source = np.array([0, 1])
    target = np.array([1, 1])
    choice_intensity = np.full(shape=(2,2), fill_value=1)

    actual_probability = ludics.compute_fermi_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
    )

    expected_probability = 0.25

    assert expected_probability == actual_probability


def test_compute_fermi_transition_probability_for_symbolic_fitness_function():
    """
    Tests whether the compute_fermi_transition_probability function returns the
    correct expression for a symbolic fitness function"""

    def symbolic_fitness_function(state, **kwargs):
        return np.array([sym.Symbol("x") if i == 0 else sym.Symbol("y") for i in state])

    source = np.array([0, 1, 1])
    target = np.array([1, 1, 1])
    beta = sym.Symbol("beta")
    choice_intensity = np.full(shape=(3,3), fill_value=beta)
    actual_probability = ludics.compute_fermi_transition_probability(
        source=source,
        target=target,
        fitness_function=symbolic_fitness_function,
        choice_intensity=choice_intensity,
    )

    x = sym.Symbol("x")
    y = sym.Symbol("y")

    expected_probability = (1 / 6) * (
        1 / (1 + sym.E ** ((x - y) * beta)) + 1 / (1 + sym.E ** ((x - y) * beta))
    )

    assert actual_probability == expected_probability


def test_compute_fermi_transition_probability_for_infeasible_states_and_no_change():
    """
    Tests whether compute_fermi_transition_probability returns the correct
    values when the state transition is not of hamming distance 1"""

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    source1 = np.array([0, 1])
    target1 = np.array([1, 0])
    choice_intensity = 0.5

    actual_probability1 = ludics.compute_fermi_transition_probability(
        source=source1,
        target=target1,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
    )

    expected_probability1 = 0

    assert expected_probability1 == actual_probability1

    source2 = np.array([0, 1])
    target2 = np.array([0, 1])

    actual_probability2 = ludics.compute_fermi_transition_probability(
        source=source2,
        target=target2,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
    )

    assert actual_probability2 is None

    _ = trivial_fitness_function(source1)  # prevents unused function warning


def test_compute_fermi_transition_probability_for_impossible_transition():
    """Tests compute_fermi_transition_probability for a
    transition which introduces a new strategy to the population"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i + 1 for i in state])

    source = np.array([1, 1, 0, 0])
    target = np.array([1, 1, 2, 0])

    choice_intensity = np.full(shape=(4,3), fill_value=0.5)

    actual_probability = ludics.compute_fermi_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
    )

    expected_probability = 0.0

    np.testing.assert_almost_equal(actual_probability, expected_probability)


def test_compute_introspective_imitation_transition_probability_for_trivial_fitenss_function():
    """
    Tests that the compute_introspective_imitation_transition_probability
    function returns the correct value for a trivial fitness function."""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i + 1 for i in state])

    source = np.array([1, 1, 0, 0])
    target = np.array([1, 1, 1, 0])

    selection_intensity = np.full(shape=(4,4), fill_value=0.1)
    choice_intensity = np.full(shape=(4,2), fill_value=0.8)
    fitness_map=ludics.linear_fitness_map
    actual_probability = (
        ludics.compute_introspective_imitation_transition_probability(
            source=source,
            target=target,
            fitness_function=trivial_fitness_function,
            choice_intensity=choice_intensity,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map
        )
    )

    expected_probability = 0.0903538011

    np.testing.assert_almost_equal(
        actual_probability, expected_probability, err_msg=actual_probability
    )


def test_compute_introspective_imitation_transition_probability_for_symbolic_fitness_function():
    """
    Tests whether the compute_introspective_imitation_transition_probability
    function returns the correct expression for a symbolic fitness function"""

    def symbolic_fitness_function(state, **kwargs):
        return np.array([sym.Symbol("x") if i == 0 else sym.Symbol("y") for i in state])

    source = np.array([0, 1, 1, 0, 0])
    target = np.array([1, 1, 1, 0, 0])
    beta = sym.Symbol("\beta")
    epsilon = sym.Symbol("\epsilon")
    selection_intensity = np.full(shape=(5,5), fill_value=epsilon)
    choice_intensity = np.full(shape=(5,2), fill_value=beta)
    fitness_map=ludics.linear_fitness_map

    actual_probability = (
        ludics.compute_introspective_imitation_transition_probability(
            source=source,
            target=target,
            fitness_function=symbolic_fitness_function,
            choice_intensity=choice_intensity,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map
        )
    )

    x = sym.Symbol("x")
    y = sym.Symbol("y")
    fy = 1 - epsilon + epsilon * y
    fx = 1 - epsilon + epsilon * x

    expected_probability = (
        (1 / 5)
        * (2 * fy)
        * (1 / ((2 * fy) + (3 * fx)))
        * (1 / (1 + sym.E ** ((x - y) * beta)))
    )

    assert sym.simplify(actual_probability - expected_probability) == 0


def test_compute_introspective_imitation_transition_probability_for_infeasible_states_and_no_change():
    """
    Tests whether compute_introspective_imitation_transition_probability returns the correct
    values when the state transition is not of hamming distance 1"""

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    source1 = np.array([0, 1])
    target1 = np.array([1, 0])
    choice_intensity = 0.5
    selection_intensity = np.full(shape=(2,2), fill_value=0.8)
    fitness_map=ludics.linear_fitness_map

    actual_probability1 = (
        ludics.compute_introspective_imitation_transition_probability(
            source=source1,
            target=target1,
            fitness_function=trivial_fitness_function,
            choice_intensity=choice_intensity,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map
        )
    )

    expected_probability1 = 0

    assert expected_probability1 == actual_probability1

    source2 = np.array([0, 1])
    target2 = np.array([0, 1])

    actual_probability2 = (
        ludics.compute_introspective_imitation_transition_probability(
            source=source2,
            target=target2,
            fitness_function=trivial_fitness_function,
            choice_intensity=choice_intensity,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map
        )
    )

    assert actual_probability2 is None

    _ = trivial_fitness_function(source1)  # prevents unused function warning


def test_compute_introspective_imitation_for_impossible_transition():
    """Tests compute_introspective_imitation_transition_probability for a
    transition which introduces a new strategy to the population"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i + 1 for i in state])

    source = np.array([1, 1, 0, 0])
    target = np.array([1, 1, 2, 0])
    fitness_map=ludics.linear_fitness_map

    choice_intensity = np.full(shape=(4,3), fill_value=0.5)
    selection_intensity = np.full(shape=(4,4), fill_value=0.8)

    actual_probability = (
        ludics.compute_introspective_imitation_transition_probability(
            source=source,
            target=target,
            fitness_function=trivial_fitness_function,
            choice_intensity=choice_intensity,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map
        )
    )

    expected_probability = 0.0

    np.testing.assert_almost_equal(actual_probability, expected_probability)


def test_compute_introspective_imitation_for_global_transition():
    """Tests compute_introspective_imitation_transition_probability for a
    transition which gives a different fitness to the changing player in the
    new state."""

    def heterogeneous_fitness_function(state, **kwargs):
        return np.array([i + np.sum(state) for i in state])

    source = np.array([1, 1, 0, 0])
    target = np.array([1, 1, 1, 0])

    choice_intensity = np.full(shape=(4,2), fill_value=0.5)
    selection_intensity = np.full(shape=(4,4), fill_value=0.5)
    fitness_map=ludics.linear_fitness_map

    actual_probability = (
        ludics.compute_introspective_imitation_transition_probability(
            source=source,
            target=target,
            fitness_function=heterogeneous_fitness_function,
            choice_intensity=choice_intensity,
            selection_intensity=selection_intensity,
            fitness_map=fitness_map
        )
    )

    expected_probability = 0.1044369398
    np.testing.assert_almost_equal(
        actual_probability, expected_probability, err_msg=actual_probability
    )


def test_compute_introspection_transition_probability_for_trivial_fitness_function():
    """
    Tests that the compute_introspection_transition_probability
    function returns the correct value for a trivial fitness function."""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i + 1 for i in state])

    source = np.array([1, 1, 0])
    target = np.array([1, 1, 2])

    choice_intensity = np.full(shape=(3,3), fill_value=0.5)
    number_of_strategies = 3
    fitness_map=ludics.linear_fitness_map

    actual_probability = ludics.compute_introspection_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
        number_of_strategies=number_of_strategies,
        fitness_map=fitness_map
    )

    expected_probability = 0.1218430964

    np.testing.assert_almost_equal(actual_probability, expected_probability)


def test_compute_introspection_transition_probability_for_symbolic_fitness_function():
    """
    Tests that the compute_introspective_imitation_transition_probability
    function returns the correct value for a trivial fitness function."""

    def symbolic_fitness_function(state, **kwargs):
        return np.array([sym.Symbol(f"x_{i}") for i in state])

    source = np.array([1, 1, 0])
    target = np.array([1, 1, 2])
    fitness_map=ludics.linear_fitness_map
    choice_intensity = np.full(shape=(3,3), fill_value=sym.Symbol("Beta"))
    number_of_strategies = sym.Symbol("k")
    x_0 = sym.Symbol("x_0")
    x_2 = sym.Symbol("x_2")

    actual_probability = ludics.compute_introspection_transition_probability(
        source=source,
        target=target,
        fitness_function=symbolic_fitness_function,
        choice_intensity=choice_intensity,
        number_of_strategies=number_of_strategies,
        fitness_map=fitness_map
    )

    expected_probability = (
        (1 / (3 * (number_of_strategies - 1)))
        * 1
        / (1 + sym.E ** ((x_0 - x_2) * choice_intensity))
    )

    assert sym.simplify(actual_probability == expected_probability)


def test_compute_introspection_transition_probability_for_infeasible_states_and_no_change():
    """
    Tests whether compute_introspection_transition_probability returns the correct
    values when the state transition is not of hamming distance 1"""

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    source1 = np.array([0, 1])
    target1 = np.array([1, 0])
    choice_intensity = 0.5
    number_of_strategies = 2

    actual_probability1 = ludics.compute_introspection_transition_probability(
        source=source1,
        target=target1,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
        number_of_strategies=number_of_strategies,
    )

    expected_probability1 = 0

    assert expected_probability1 == actual_probability1

    source2 = np.array([0, 1])
    target2 = np.array([0, 1])

    actual_probability2 = ludics.compute_introspection_transition_probability(
        source=source2,
        target=target2,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
        number_of_strategies=number_of_strategies,
    )

    assert actual_probability2 is None

    _ = trivial_fitness_function(source1)  # prevents unused function warning


def test_compute_steady_state_for_different_initial_dist():
    """tests that the compute_steady_state function correctly
    approximates a system's steady state for a different initial distribution"""

    initial_dist_1 = np.array([1, 0, 0, 0])
    initial_dist_2 = np.array([0, 0, 0, 1])

    transition_matrix = np.array(
        [[1, 0, 0, 0], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0, 0, 0, 1]]
    )

    steady_state_1 = np.array([1, 0, 0, 0])
    steady_state_2 = np.array([0, 0, 0, 1])

    np.testing.assert_array_equal(
        ludics.compute_steady_state(
            transition_matrix=transition_matrix, initial_dist=initial_dist_1
        ),
        steady_state_1,
    )

    np.testing.assert_array_equal(
        ludics.compute_steady_state(
            transition_matrix=transition_matrix, initial_dist=initial_dist_2
        ),
        steady_state_2,
    )


def test_compute_aspiration_transition_probability_for_trivial_fitness_function():
    """
    Tests the compute_aspiration_transition_probability function for a trivial
    fitness function"""

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    source = np.array([1, 1, 0])
    target = np.array([1, 1, 1])
    aspiration_vector = np.array([2, 2, 2])
    choice_intensity = np.full(shape=(3,2), fill_value=0.5)

    expected_transition_probability = 0.2074864437
    actual_transition_probability = (
        ludics.compute_aspiration_transition_probability(
            source=source,
            target=target,
            fitness_function=trivial_fitness_function,
            choice_intensity=choice_intensity,
            aspiration_vector=aspiration_vector,
        )
    )

    np.testing.assert_almost_equal(
        actual_transition_probability, expected_transition_probability
    )


def test_compute_aspiration_transition_probability_for_heterogeneous_aspiration_vector():
    """
    Tests the compute_aspiration_transition_probability function for a
    heterogeneous aspiration vector"""

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    source = np.array([1, 1, 1])
    target = np.array([1, 0, 1])
    aspiration_vector = np.array([2, 3, 4])
    choice_intensity = np.full(shape=(3,2), fill_value=0.5)

    expected_transition_probability = 0.2436861929
    actual_transition_probability = (
        ludics.compute_aspiration_transition_probability(
            source=source,
            target=target,
            fitness_function=trivial_fitness_function,
            choice_intensity=choice_intensity,
            aspiration_vector=aspiration_vector,
        )
    )

    np.testing.assert_almost_equal(
        actual_transition_probability, expected_transition_probability
    )


def test_compute_aspiration_transition_probability_for_non_trivial_fitness_function():
    """
    Tests the compute_aspiration_transition_probability function for a
    non-trivial fitness function"""

    def trivial_fitness_function(state):
        return np.array([i + 3 for i in state])

    source = np.array([0, 1, 1])
    target = np.array([1, 1, 1])
    aspiration_vector = np.array([2, 3, 4])
    choice_intensity = np.full(shape=(3,2), fill_value=0.2)

    expected_transition_probability = 0.1500553342
    actual_transition_probability = (
        ludics.compute_aspiration_transition_probability(
            source=source,
            target=target,
            fitness_function=trivial_fitness_function,
            choice_intensity=choice_intensity,
            aspiration_vector=aspiration_vector,
        )
    )

    np.testing.assert_almost_equal(
        actual_transition_probability, expected_transition_probability
    )


def test_compute_aspiration_transition_probability_for_infeasible_transition():
    """
    Tests the compute_aspiration_transition_probability function returns 0 for
    the case where source and target are a distance >=2 away from each other"""

    def trivial_fitness_function(state):
        return np.array([i + 3 for i in state])

    aspiration_vector = np.array([2, 3, 4])
    choice_intensity = 0.5

    source = np.array([0, 1, 1])
    target = np.array([1, 0, 1])

    assert (
        ludics.compute_aspiration_transition_probability(
            source=source,
            target=target,
            fitness_function=trivial_fitness_function,
            choice_intensity=choice_intensity,
            aspiration_vector=aspiration_vector,
        )
        == 0
    )


def test_compute_aspiration_transition_probability_fails_for_too_many_types():
    """
    Tests the compute_aspiration_transition_probability function fails for the
    case where vectors contain 3 different types"""

    def trivial_fitness_function(state):
        return np.array([i + 3 for i in state])

    aspiration_vector = np.array([2, 3, 4])
    choice_intensity = 0.5

    source = np.array([0, 1, 2])
    target = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        ludics.compute_aspiration_transition_probability(
            source=source,
            target=target,
            fitness_function=trivial_fitness_function,
            choice_intensity=choice_intensity,
            aspiration_vector=aspiration_vector,
        )


def test_compute_aspiration_transition_probability_for_self_transition():
    """
    Tests the compute_aspiration_transition_probability function returns None
    when a state transitions to itself"""

    def trivial_fitness_function(state):
        return np.array([i + 3 for i in state])

    aspiration_vector = np.array([2, 3, 4])
    choice_intensity = 0.5

    source = np.array([0, 1, 1])
    target = np.array([0, 1, 1])

    assert (
        ludics.compute_aspiration_transition_probability(
            source=source,
            target=target,
            fitness_function=trivial_fitness_function,
            choice_intensity=choice_intensity,
            aspiration_vector=aspiration_vector,
        )
        is None
    )


def test_get_neighbourhood_states_for_standard_state():
    """
    Tests that get_neighbourhood_states returns the correct array for a
    standard state."""

    state = np.array([1, 0, 1])
    number_of_strategies = 2

    expected_neighbourhood = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 0]])

    actual_neighbourhood = ludics.get_neighbourhood_states(
        state=state, number_of_strategies=number_of_strategies
    )

    np.testing.assert_array_equal(actual_neighbourhood, expected_neighbourhood)


def test_get_neighbourhood_states_for_lots_of_strategies():
    """
    Tests that get_neighbourhood_states returns the correct array for a state
    with many strategies."""

    state = np.array([1, 0])
    number_of_strategies = 6

    expected_neighbourhood = np.array(
        [
            [0, 0],
            [2, 0],
            [3, 0],
            [4, 0],
            [5, 0],
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
        ]
    )

    actual_neighbourhood = ludics.get_neighbourhood_states(
        state=state, number_of_strategies=number_of_strategies
    )

    np.testing.assert_array_equal(actual_neighbourhood, expected_neighbourhood)


def test_apply_mutation_probability_for_standard_mutation_vector():
    """
    Tests that apply_mutation_probability correctly applies under standard
    circumstances - a transition probability and a correctly formatted mutation
    vector"""

    source = np.array([1, 1, 0])
    target = np.array([1, 2, 0])
    transition_probability = 0.8
    individual_to_action_mutation_probability = np.array(
        [[0, 0.1, 0.1], [0.09, 0, 0.06], [0, 0, 0.1]]
    )

    expected_mutation_transition_probability = 0.7

    actual_mutation_transition_probability = ludics.apply_mutation_probability(
        source=source,
        target=target,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
        transition_probability=transition_probability,
    )

    np.testing.assert_almost_equal(
        expected_mutation_transition_probability, actual_mutation_transition_probability
    )


def test_apply_mutation_probability_for_no_mutation_vector():
    """
    Tests that apply_mutation_probability correctly applies when the mutation
    vector is the zero vector"""

    source = np.array([1, 1, 0])
    target = np.array([1, 2, 0])
    transition_probability = 0.8
    individual_to_action_mutation_probability = np.array(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    )

    expected_mutation_transition_probability = 0.8

    actual_mutation_transition_probability = ludics.apply_mutation_probability(
        source=source,
        target=target,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
        transition_probability=transition_probability,
    )

    assert (
        expected_mutation_transition_probability
        == actual_mutation_transition_probability
    )


def test_apply_mutation_probability_for_infeasible_transition():
    """
    Tests that apply_mutation_probability correctly applies when the mutation
    vector is the zero vector"""

    source = np.array([1, 1, 0])
    target = np.array([1, 2, 2])
    transition_probability = 0
    individual_to_action_mutation_probability = np.array(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    )

    expected_mutation_transition_probability = 0

    actual_mutation_transition_probability = ludics.apply_mutation_probability(
        source=source,
        target=target,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
        transition_probability=transition_probability,
    )

    assert (
        expected_mutation_transition_probability
        == actual_mutation_transition_probability
    )


def test_simulate_markov_chain_for_trivial_fitness_function():
    """
    tests that simulate_markov_chain returns the proper values for a trivial
    fitness function and a small number of time steps"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([1 for _ in state])

    initial_state = np.array([1, 1, 1])
    number_of_strategies = 2
    seed = 2
    iterations = 5
    fitness_function = trivial_fitness_function
    choice_intensity = np.full(shape=(3,2), fill_value=0.3)

    expected_states_over_time = [
        tuple(np.array([1, 1, 1])),
        tuple(np.array([1, 1, 0])),
        tuple(np.array([0, 1, 0])),
        tuple(np.array([0, 1, 0])),
        tuple(np.array([0, 1, 1])),
    ]

    actual_states_over_time, _ = ludics.simulate_markov_chain(
        initial_state=initial_state,
        number_of_strategies=number_of_strategies,
        fitness_function=fitness_function,
        compute_transition_probability=ludics.compute_introspection_transition_probability,
        seed=seed,
        iterations=iterations,
        choice_intensity=choice_intensity,
    )
    assert actual_states_over_time == expected_states_over_time


def test_simulate_markov_chain_for_warmup():
    """
    tests that simulate_markov_chain returns the proper values for a trivial
    fitness function and a small number of time steps"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([1 for _ in state])

    initial_state = np.array([1, 1, 1])
    number_of_strategies = 2
    seed = 2
    iterations = 5
    warmup = 1
    fitness_function = trivial_fitness_function
    choice_intensity = np.full(shape=(3,2), fill_value=0.3)

    expected_states_over_time = [
        tuple(np.array([1, 1, 0])),
        tuple(np.array([0, 1, 0])),
        tuple(np.array([0, 1, 0])),
        tuple(np.array([0, 1, 1])),
    ]

    actual_states_over_time, _ = ludics.simulate_markov_chain(
        initial_state=initial_state,
        number_of_strategies=number_of_strategies,
        fitness_function=fitness_function,
        compute_transition_probability=ludics.compute_introspection_transition_probability,
        seed=seed,
        iterations=iterations,
        warmup=warmup,
        choice_intensity=choice_intensity,
    )
    assert actual_states_over_time == expected_states_over_time


def test_simulate_markov_chain_for_moran_process():
    """
    Tests that simulate_markov_chain returns the correct values when using the
    moran process"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([1 for _ in state])

    initial_state = np.array([1, 1, 0])
    number_of_strategies = 2
    seed = 2
    iterations = 5
    fitness_function = trivial_fitness_function
    selection_intensity=np.full(shape=(3,3), fill_value=0.3)

    expected_states_over_time = [
        tuple(np.array([1, 1, 0])),
        tuple(np.array([1, 1, 1])),
        tuple(np.array([1, 1, 1])),
        tuple(np.array([1, 1, 1])),
        tuple(np.array([1, 1, 1])),
    ]
    fitness_map=ludics.linear_fitness_map

    actual_states_over_time, _ = ludics.simulate_markov_chain(
        initial_state=initial_state,
        number_of_strategies=number_of_strategies,
        fitness_function=fitness_function,
        compute_transition_probability=ludics.compute_moran_transition_probability,
        seed=seed,
        iterations=iterations,
        selection_intensity=selection_intensity,
        fitness_map=fitness_map
    )

    assert actual_states_over_time == expected_states_over_time


def test_simulate_markov_chain_for_moran_process_counter():
    """
    Tests that simulate_markov_chain returns the correct values when using the
    moran process"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([1 for _ in state])

    initial_state = np.array([1, 1, 0])
    number_of_strategies = 2
    seed = 2
    iterations = 5
    fitness_function = trivial_fitness_function
    selection_intensity=np.full(shape=(3,3), fill_value=0.3)
    fitness_map=ludics.linear_fitness_map
    expected_state_distribution = {
        tuple(np.array([1, 1, 0])): 1,
        tuple(np.array([1, 1, 1])): 4,
    }

    _, actual_state_distribution = ludics.simulate_markov_chain(
        initial_state=initial_state,
        number_of_strategies=number_of_strategies,
        fitness_function=fitness_function,
        compute_transition_probability=ludics.compute_moran_transition_probability,
        seed=seed,
        iterations=iterations,
        selection_intensity=selection_intensity,
        fitness_map=fitness_map
    )

    assert actual_state_distribution == expected_state_distribution


def test_simulate_markov_chain_gives_correct_numeric_results_introspection():
    """
    Tests that the results we see from simulate_markov_chain give us the
    correct approximate values that we see from our numeric function
    compute_steady_state when using introspection dynamics"""

    def fitness_function(state, **kwargs):
        return np.array([i + j for i, j in enumerate(state)])

    initial_state = np.array([0, 1, 0])
    choice_intensity = np.full(shape=(3,2), fill_value=1)
    number_of_strategies = 2
    seed = 1
    iterations = 10000
    state_space = ludics.get_state_space(N=3, k=number_of_strategies)
    individual_to_action_mutation_probability = np.full((3, number_of_strategies), 0.2)
    transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=fitness_function,
        compute_transition_probability=ludics.compute_introspection_transition_probability,
        number_of_strategies=number_of_strategies,
        choice_intensity=choice_intensity,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
    )

    expected_state_distribution = ludics.compute_steady_state(
        transition_matrix=transition_matrix
    )

    _, states_and_counts = ludics.simulate_markov_chain(
        initial_state=initial_state,
        number_of_strategies=number_of_strategies,
        fitness_function=fitness_function,
        seed=seed,
        iterations=iterations,
        compute_transition_probability=ludics.compute_introspection_transition_probability,
        choice_intensity=choice_intensity,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
    )

    actual_state_distribution = np.array(
        [states_and_counts[tuple(state.tolist())] / iterations for state in state_space]
    )

    np.testing.assert_array_almost_equal(
        actual_state_distribution, expected_state_distribution, decimal=2
    )


def test_simulate_markov_chain_gives_correct_numeric_results_moran():
    """
    Tests that the results we see from simulate_markov_chain give us the
    correct approximate values that we should see from our numeric function
    compute_steady_state using the moran process"""

    def fitness_function(state, **kwargs):
        return np.array([i + j for i, j in enumerate(state)])

    initial_state = np.array([0, 1, 0])
    selection_intensity=np.full(shape=(3,3), fill_value=0.5)
    number_of_strategies = 2
    seed = 1
    iterations = 10000
    state_space = ludics.get_state_space(N=3, k=number_of_strategies)
    fitness_map=ludics.linear_fitness_map

    individual_to_action_mutation_probability = np.full((3, number_of_strategies), 0.2)
    transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=fitness_function,
        compute_transition_probability=ludics.compute_moran_transition_probability,
        selection_intensity=selection_intensity,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
        fitness_map=fitness_map
    )

    expected_state_distribution = ludics.compute_steady_state(
        transition_matrix=transition_matrix
    )

    _, states_and_counts = ludics.simulate_markov_chain(
        initial_state=initial_state,
        number_of_strategies=number_of_strategies,
        fitness_function=fitness_function,
        seed=seed,
        iterations=iterations,
        compute_transition_probability=ludics.compute_moran_transition_probability,
        selection_intensity=selection_intensity,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
        fitness_map=fitness_map
    )

    actual_state_distribution = np.array(
        [states_and_counts[tuple(state.tolist())] / iterations for state in state_space]
    )

    np.testing.assert_array_almost_equal(
        actual_state_distribution, expected_state_distribution, decimal=2
    )


def test_simulate_markov_chain_gives_correct_numeric_results_fermi():
    """
    Tests that the results we see from simulate_markov_chain give us the
    correct approximate values that we should see from our numeric function
    compute_steady_state using fermi imitation dynamics"""

    def fitness_function(state, **kwargs):
        return np.array([i + j for i, j in enumerate(state)])

    initial_state = np.array([0, 1, 0])
    choice_intensity = np.full(shape=(3,3), fill_value=0.12)
    number_of_strategies = 2
    seed = 1
    iterations = 10000
    state_space = ludics.get_state_space(N=3, k=number_of_strategies)

    individual_to_action_mutation_probability = np.full((3, number_of_strategies), 0.2)
    transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=fitness_function,
        compute_transition_probability=ludics.compute_fermi_transition_probability,
        choice_intensity=choice_intensity,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
    )

    expected_state_distribution = ludics.compute_steady_state(
        transition_matrix=transition_matrix
    )

    _, states_and_counts = ludics.simulate_markov_chain(
        initial_state=initial_state,
        number_of_strategies=number_of_strategies,
        fitness_function=fitness_function,
        seed=seed,
        iterations=iterations,
        compute_transition_probability=ludics.compute_fermi_transition_probability,
        choice_intensity=choice_intensity,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
    )

    actual_state_distribution = np.array(
        [states_and_counts[tuple(state.tolist())] / iterations for state in state_space]
    )

    np.testing.assert_array_almost_equal(
        actual_state_distribution, expected_state_distribution, decimal=2
    )


def test_simulate_markov_chain_gives_correct_numeric_results_imispection():
    """
    Tests that the results we see from simulate_markov_chain give us the
    correct approximate values that we should see from our numeric function
    compute_steady_state using introspective imitation dynamics"""

    def fitness_function(state, **kwargs):
        return np.array([i + j for i, j in enumerate(state)])

    initial_state = np.array([0, 1, 0])
    choice_intensity = np.full(shape=(3,2), fill_value=0.3)
    selection_intensity = np.full(shape=(3,3), fill_value=0.8)
    number_of_strategies = 2
    seed = 2
    iterations = 100000
    state_space = ludics.get_state_space(N=3, k=number_of_strategies)
    fitness_map=ludics.linear_fitness_map

    individual_to_action_mutation_probability = np.full((3, number_of_strategies), 0.2)
    transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=fitness_function,
        compute_transition_probability=ludics.compute_introspective_imitation_transition_probability,
        choice_intensity=choice_intensity,
        selection_intensity=selection_intensity,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
        fitness_map=fitness_map
    )

    expected_state_distribution = ludics.compute_steady_state(
        transition_matrix=transition_matrix
    )

    _, states_and_counts = ludics.simulate_markov_chain(
        initial_state=initial_state,
        number_of_strategies=number_of_strategies,
        fitness_function=fitness_function,
        seed=seed,
        iterations=iterations,
        compute_transition_probability=ludics.compute_introspective_imitation_transition_probability,
        choice_intensity=choice_intensity,
        selection_intensity=selection_intensity,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
        fitness_map=fitness_map
    )

    actual_state_distribution = np.array(
        [states_and_counts[tuple(state.tolist())] / iterations for state in state_space]
    )

    np.testing.assert_array_almost_equal(
        actual_state_distribution, expected_state_distribution, decimal=2
    )


def test_simulate_markov_chain_gives_correct_numeric_results_aspiration():
    """
    Tests that the results we see from simulate_markov_chain give us the
    correct approximate values that we should see from our numeric function
    compute_steady_state using aspiration dynamics"""

    def fitness_function(state, **kwargs):
        return np.array([i + j for i, j in enumerate(state)])

    initial_state = np.array([0, 1, 0])
    choice_intensity = np.full(shape=(3,2), fill_value=0.12)
    number_of_strategies = 2
    seed = 1
    iterations = 10000
    state_space = ludics.get_state_space(N=3, k=number_of_strategies)
    aspiration_vector = np.array([2, 2, 2])

    individual_to_action_mutation_probability = np.full((3, number_of_strategies), 0.2)

    transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=fitness_function,
        compute_transition_probability=ludics.compute_aspiration_transition_probability,
        choice_intensity=choice_intensity,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
        aspiration_vector=aspiration_vector,
    )

    expected_state_distribution = ludics.compute_steady_state(
        transition_matrix=transition_matrix
    )

    _, states_and_counts = ludics.simulate_markov_chain(
        initial_state=initial_state,
        number_of_strategies=number_of_strategies,
        fitness_function=fitness_function,
        seed=seed,
        iterations=iterations,
        compute_transition_probability=ludics.compute_aspiration_transition_probability,
        choice_intensity=choice_intensity,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
        aspiration_vector=aspiration_vector,
    )

    actual_state_distribution = np.array(
        [states_and_counts[tuple(state.tolist())] / iterations for state in state_space]
    )

    np.testing.assert_array_almost_equal(
        actual_state_distribution, expected_state_distribution, decimal=2
    )


def test_generate_transition_matrix_for_hybrid_population_dynamic():
    """Tests that generate_transition_matrix returns the correct values when
    using different population dynamics for each player
    
    Note that in this test, we use a homogeneous choice intensity across
    Fermi and introspection dynamics. Thus, the slightly different way
    that choice intensity works (player to player in Fermi and player to
    strategy in introspection) has no impact here. The larger of the 
    two arrays (shape = (3,3) rather than (3,2)) works for both 
    dynamics"""

    population_dynamic_array = np.array(
        [
            ludics.compute_moran_transition_probability,
            ludics.compute_fermi_transition_probability,
            ludics.compute_introspection_transition_probability,
        ]
    )

    N = 3
    number_of_strategies = 2
    state_space = ludics.get_state_space(N=N, k=number_of_strategies)
    r = 2
    contribution_vector = np.array([1, 2, 3])
    choice_intensity = np.full(shape=(3,3), fill_value=1)
    selection_intensity=np.full(shape=(3,3), fill_value=0.1)
    hybrid_population_dynamic = ludics.build_hybrid_population_dynamic(
        population_dynamic_array
    )
    fitness_map=ludics.linear_fitness_map

    actual_transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=ludics.fitness_functions.public_goods_game_fitness_function,
        compute_transition_probability=hybrid_population_dynamic,
        r=r,
        alpha=contribution_vector,
        choice_intensity=choice_intensity,
        selection_intensity=selection_intensity,
        number_of_strategies=number_of_strategies,
        fitness_map=fitness_map
    )

    expected_transition_matrix = np.array(
        [
            [1 - 0.08964714046, 0.08964714046, 0, 0, 0, 0, 0, 0],
            [0.2436861929, 1 - 0.2436861929 - 0.007904312196 - 0.08888888889, 0, 0.007904312196, 0, 0.08888888889, 0, 0],
            [
                0.293599026,
                0,
                1 - 0.08964714046 - 0.293599026 - 0.09578544061,
                0.08964714046,
                0,
                0,
                0.09578544061,
                0,
            ],
            [0, 0.146799513, 0.2436861929, 1 - 0.146799513 - 0.2436861929 - 0.2048611111, 0, 0, 0, 0.2048611111],
            [0.2301587302, 0, 0, 0, 1 - 0.2301587302 - 0.08964714046 - 0.04482357023, 0.08964714046, 0.04482357023, 0],
            [
                0,
                0.1254480287,
                0,
                0,
                0.2436861929,
                1 - 0.1254480287 - 0.2436861929 - 0.0527278824245936,
                0,
                0.0527278824245936,
            ],
            [0, 0, 0.1222222222, 0, 0.146799513, 0, 1 - 0.08964714046 - 0.146799513 - 0.1222222222, 0.08964714046],
            [0, 0, 0, 0, 0, 0, 0.2436861929, 1 - 0.2436861929],
        ]
    )

    np.testing.assert_array_almost_equal(
        expected_transition_matrix, actual_transition_matrix
    )


def test_build_hybrid_dynamic_calls_correct_functions():
    """
    Tests that the function built by build_hybrid_dynamics calls the correct
    function at any given time"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i for i in state])

    population_dynamic_array = np.array(
        [
            ludics.compute_fermi_transition_probability,
            ludics.compute_introspection_transition_probability,
        ]
    )
    choice_intensity = np.full(shape=(2,2), fill_value=0.5)

    state_1 = np.array([0, 0])
    state_2 = np.array([1, 0])
    state_3 = np.array([0, 1])

    hybrid_dynamic = ludics.build_hybrid_population_dynamic(
        population_dynamic_array
    )

    expected_results = np.array([0, 1 / (2 * (1 + np.exp(-0.5)))])
    actual_results = np.array(
        [
            hybrid_dynamic(
                source=state_1,
                target=state_2,
                fitness_function=trivial_fitness_function,
                choice_intensity=choice_intensity,
                number_of_strategies=2,
            ),
            hybrid_dynamic(
                source=state_1,
                target=state_3,
                fitness_function=trivial_fitness_function,
                choice_intensity=choice_intensity,
                number_of_strategies=2,
            ),
        ]
    )
    np.testing.assert_array_almost_equal(actual_results, expected_results)

def test_aspiration_fails_for_gt_2_action_types():
    """
    Tests that compute_aspiration_transition_probability fails for too many
    action types"""
    def trivial_fitness_function(state, **kwargs):
        return np.array([i for i in state])

    source = np.array([0,1,1])
    target = np.array([0,1,2])
    choice_intensity = np.full(shape=(3,3), fill_value=0.5)
    aspiration_vector = np.array([1,2,3])

    with pytest.raises(ValueError):
        ludics.compute_aspiration_transition_probability(
            source=source,
            target=target,
            fitness_function=trivial_fitness_function,
            choice_intensity=choice_intensity,
            aspiration_vector=aspiration_vector,
        )

def test_build_hybrid_population_dynamics_returns_none():
    """
    Tests that build_hybrid_population_dynamics returns None if 
    states are equal"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i for i in state])

    population_dynamic_array = np.array(
        [
            ludics.compute_fermi_transition_probability,
            ludics.compute_introspection_transition_probability,
        ]
    )
    choice_intensity = np.full(shape=(2,2), fill_value=0.5)

    state_1 = np.array([0, 0])
    state_2 = np.array([0, 0])

    hybrid_dynamic = ludics.build_hybrid_population_dynamic(
        population_dynamic_array
    )

    assert hybrid_dynamic(source=state_1, target=state_2, fitness_function=trivial_fitness_function, choice_intensity=choice_intensity) is None

def test_compute_moran_transition_probability_for_heterogeneous_intensity():
    """
    Tests that compute_moran_transition_probability computes the correct value
    for a heterogeneous selection intensity"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([2 for i in state])
    fitness_map=ludics.linear_fitness_map
    
    source = np.array([1,0,1,0])
    target_1 = np.array([0,0,1,0])
    target_2 = np.array([1,0,1,1])

    selection_intensity = np.array([
        [0.1,0.2,0.3,0.4],
        [0.05, 0.01, 0.02, 0.03],
        [0.06, 0.07, 0.08, 0.09],
        [0.5,0.6,0.7,0.8]
    ])

    actual_transition_probability_1 = ludics.compute_moran_transition_probability(
        source=source,
        target=target_1,
        fitness_function=trivial_fitness_function,
        selection_intensity=selection_intensity,
        fitness_map=fitness_map
    )

    expected_transition_probability_1 = 0.13

    assert actual_transition_probability_1 == expected_transition_probability_1

    actual_transition_probability_2 = ludics.compute_moran_transition_probability(
        source=source,
        target=target_2,
        fitness_function=trivial_fitness_function,
        selection_intensity=selection_intensity,
        fitness_map=fitness_map
    )

    expected_transition_probability_2 = 4/33

    np.testing.assert_almost_equal(actual_transition_probability_2, expected_transition_probability_2)

def test_compute_fermi_transition_probability_for_heterogeneous_intensity():
    """
    Tests that compute_fermi_transition_probability computes the correct value
    for a heterogeneous choice intensity"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i+1 for i,_ in enumerate(state)])
    
    source = np.array([1,0,1,0])
    target_1 = np.array([1,0,0,0])
    target_2 = np.array([1,1,1,0])

    choice_intensity = np.array([
        [0.1,0.2,0.3,0.4],
        [0.05, 0.01, 0.02, 0.03],
        [0.06, 0.07, 0.08, 0.09],
        [0.5,0.6,0.7,0.8]
    ])

    actual_transition_probability_1 = ludics.compute_fermi_transition_probability(
        source=source,
        target=target_1,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity
    )

    expected_transition_probability_1 = 0.08374933059

    np.testing.assert_almost_equal(actual_transition_probability_1,expected_transition_probability_1)

    actual_transition_probability_2 = ludics.compute_fermi_transition_probability(
        source=source,
        target=target_2,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity
    )

    expected_transition_probability_2 = 0.0827085364

    np.testing.assert_almost_equal(actual_transition_probability_2, expected_transition_probability_2)

def test_compute_introspection_transition_probability_for_heterogeneous_intensity():
    """
    Tests that compute_introspection_transition_probability computes the correct value
    for a heterogeneous choice intensity"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i for _,i in enumerate(state)])
    
    source = np.array([1,0,0])
    target_1 = np.array([0,0,0])
    target_2 = np.array([1,1,0])

    choice_intensity = np.array([
        [0.1,0.2],
        [0.05, 0.12],
        [0.06, 0.07],
    ])

    actual_transition_probability_1 = ludics.compute_introspection_transition_probability(
        source=source,
        target=target_1,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
        number_of_strategies=2
    )

    expected_transition_probability_1 = 0.1583402708

    np.testing.assert_almost_equal(actual_transition_probability_1,expected_transition_probability_1)

    actual_transition_probability_2 = ludics.compute_introspection_transition_probability(
        source=source,
        target=target_2,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
        number_of_strategies=2
    )

    expected_transition_probability_2 = 0.1766546839

    np.testing.assert_almost_equal(actual_transition_probability_2, expected_transition_probability_2)

def test_compute_aspiration_transition_probability_for_heterogeneous_intensity():
    """
    Tests that compute_aspiration_transition_probability computes the correct value
    for a heterogeneous choice intensity"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i for _,i in enumerate(state)])
    
    source = np.array([0,1,1,0])
    target_1 = np.array([0,1,0,0])
    target_2 = np.array([0,1,1,1])

    choice_intensity = np.array([
        [0.1,0.2],
        [0.05, 0.12],
        [0.06, 0.07],
        [0.8, 0.2],
    ])

    aspiration_vector = np.array([1,3,4,2])

    actual_transition_probability_1 = ludics.compute_aspiration_transition_probability(
        source=source,
        target=target_1,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
        aspiration_vector=aspiration_vector
    )

    expected_transition_probability_1 = 0.1380769774

    np.testing.assert_almost_equal(actual_transition_probability_1,expected_transition_probability_1)

    actual_transition_probability_2 = ludics.compute_aspiration_transition_probability(
        source=source,
        target=target_2,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
        aspiration_vector=aspiration_vector
    )

    expected_transition_probability_2 = 0.2080045963

    np.testing.assert_almost_equal(actual_transition_probability_2, expected_transition_probability_2)

def test_compute_introspective_imitation_transition_probability_for_heterogeneous_intensity():
    """
    Tests that compute_introspective_imitation_transition_probability computes
    the correct value for a heterogeneous selection and choice intensities"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i + 2 for i in state])
    
    source = np.array([1,0,1,0])
    target_1 = np.array([0,0,1,0])
    target_2 = np.array([1,0,1,1])
    fitness_map=ludics.linear_fitness_map
    selection_intensity = np.array([
        [0.1,0.2,0.3,0.4],
        [0.05, 0.01, 0.02, 0.03],
        [0.06, 0.07, 0.08, 0.09],
        [0.5,0.6,0.7,0.8]
    ])

    choice_intensity = np.array([
        [0.1, 0.2],
        [0.9,0.8],
        [0.05, 0.01],
        [0.2, 0.17]
    ])

    actual_transition_probability_1 = ludics.compute_introspective_imitation_transition_probability(
        source=source,
        target=target_1,
        fitness_function=trivial_fitness_function,
        selection_intensity=selection_intensity,
        choice_intensity=choice_intensity,
        fitness_map=fitness_map
    )

    expected_transition_probability_1 = 0.05717843114

    np.testing.assert_almost_equal(actual_transition_probability_1,expected_transition_probability_1)

    actual_transition_probability_2 = ludics.compute_introspective_imitation_transition_probability(
        source=source,
        target=target_2,
        fitness_function=trivial_fitness_function,
        selection_intensity=selection_intensity,
        choice_intensity=choice_intensity,
        fitness_map=fitness_map
    )
    expected_transition_probability_2 = 0.07649201729
    
    np.testing.assert_almost_equal(actual_transition_probability_2, expected_transition_probability_2)

def test_linear_fitness_map():
    """
    Tests that the linear fitness map returns the correct value for a
    heterogeneous selection intensity"""

    selection_intensity = np.array([
        [0.1, 0.4, 0.8, 0.2],
        [0.1, 0.1, 0.1, 0.1],
        [0.4, 0.2, 0.8, 0.3],
        [0.1, 0.3, 0.7, 0.6],
    ])

    fitness = np.array([1,2,3,4])

    actual_mapped_fitness = ludics.linear_fitness_map(fitness=fitness, selection_intensity=selection_intensity[0])
    expected_mapped_fitness = np.array([1, 1.4, 2.6, 1.6])

    np.testing.assert_array_almost_equal(actual_mapped_fitness, expected_mapped_fitness)

def test_exponential_fitness_map():
    """
    Tests that the exponential fitness map returns the correct value for a
    heterogeneous selection intensity"""

    selection_intensity = np.array([
        [0.1, 0.4, 0.8, 0.2],
        [0.1, 0.1, 0.1, 0.1],
        [0.4, 0.2, 0.8, 0.3],
        [0.1, 0.3, 0.7, 0.6],
    ])

    fitness = np.array([0,1,2,3])

    actual_mapped_fitness = ludics.exponential_fitness_map(fitness=fitness, selection_intensity=selection_intensity[2])
    expected_mapped_fitness = np.array([1, 1.2214027582, 4.9530324244, 2.4596031112])

    np.testing.assert_array_almost_equal(actual_mapped_fitness, expected_mapped_fitness)

def test_compute_moran_transition_probability_for_homogeneous_intensity():
    """
    Tests that passing a float, np.float(32), and np.float(64) returns
    the same value as passing a numpy.array in the Moran process"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i for _,i in enumerate(state)])

    float_selection_intensity = 0.5
    npfloat32_selection_intensity = np.float32(0.5)
    npfloat64_selection_intensity = np.float64(0.5)
    array_selection_intensity = np.full(shape=(3,3), fill_value=0.5)

    source = np.array([1,1,0])
    target = np.array([0,1,0])

    float_transition_probability = ludics.compute_moran_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        selection_intensity=float_selection_intensity,
        fitness_map=ludics.linear_fitness_map
    )

    np32_transition_probability = ludics.compute_moran_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        selection_intensity=npfloat32_selection_intensity,
        fitness_map=ludics.linear_fitness_map
    )

    np64_transition_probability = ludics.compute_moran_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        selection_intensity=npfloat64_selection_intensity,
        fitness_map=ludics.linear_fitness_map
    )
    
    array_transition_probability = ludics.compute_moran_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        selection_intensity=array_selection_intensity,
        fitness_map=ludics.linear_fitness_map
    )

    assert float_transition_probability == np32_transition_probability == np64_transition_probability == array_transition_probability


def test_compute_fermi_transition_probability_for_homogeneous_intensity():
    """
    Tests that passing a float, np.float(32), and np.float(64) returns
    the same value as passing a numpy.array in Fermi imitation dynamics"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i for _,i in enumerate(state)])

    float_choice_intensity = 0.5
    npfloat32_choice_intensity = np.float32(0.5)
    npfloat64_choice_intensity = np.float64(0.5)
    array_choice_intensity = np.full(shape=(3,3), fill_value=0.5)

    source = np.array([1,1,0])
    target = np.array([0,1,0])

    float_transition_probability = ludics.compute_fermi_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=float_choice_intensity,
    )

    np32_transition_probability = ludics.compute_fermi_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=npfloat32_choice_intensity,
    )

    np64_transition_probability = ludics.compute_fermi_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=npfloat64_choice_intensity,
    )
    
    array_transition_probability = ludics.compute_fermi_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=array_choice_intensity,
    )

    assert float_transition_probability == np32_transition_probability == np64_transition_probability == array_transition_probability

def test_compute_introspection_transition_probability_for_homogeneous_intensity():
    """
    Tests that passing a float, np.float(32), and np.float(64) returns
    the same value as passing a numpy.array in introspection dynamics"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i for _,i in enumerate(state)])

    float_choice_intensity = 0.5
    npfloat32_choice_intensity = np.float32(0.5)
    npfloat64_choice_intensity = np.float64(0.5)
    array_choice_intensity = np.full(shape=(3,13), fill_value=0.5)
    number_of_strategies = 13

    source = np.array([1,4,9])
    target = np.array([12,4,9])

    float_transition_probability = ludics.compute_introspection_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=float_choice_intensity,
        number_of_strategies=number_of_strategies
    )

    np32_transition_probability = ludics.compute_introspection_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=npfloat32_choice_intensity,
        number_of_strategies=number_of_strategies
    )

    np64_transition_probability = ludics.compute_introspection_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=npfloat64_choice_intensity,
        number_of_strategies=number_of_strategies
    )
    
    array_transition_probability = ludics.compute_introspection_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=array_choice_intensity,
        number_of_strategies=number_of_strategies
    )

    assert float_transition_probability == np32_transition_probability == np64_transition_probability == array_transition_probability

def test_compute_aspiration_transition_probability_for_homogeneous_intensity():
    """
    Tests that passing a float, np.float(32), and np.float(64) returns
    the same value as passing a numpy.array in aspiration dynamics"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i for _,i in enumerate(state)])

    float_choice_intensity = 0.5
    npfloat32_choice_intensity = np.float32(0.5)
    npfloat64_choice_intensity = np.float64(0.5)
    array_choice_intensity = np.full(shape=(3,2), fill_value=0.5)
    aspiration_vector = np.array([1,2,3])

    source = np.array([1,0,1])
    target = np.array([1,0,0])

    float_transition_probability = ludics.compute_aspiration_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=float_choice_intensity,
        aspiration_vector=aspiration_vector
    )

    np32_transition_probability = ludics.compute_aspiration_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=npfloat32_choice_intensity,
        aspiration_vector=aspiration_vector
    )

    np64_transition_probability = ludics.compute_aspiration_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=npfloat64_choice_intensity,
        aspiration_vector=aspiration_vector
    )
    
    array_transition_probability = ludics.compute_aspiration_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=array_choice_intensity,
        aspiration_vector=aspiration_vector
    )

    assert float_transition_probability == np32_transition_probability == np64_transition_probability == array_transition_probability


def test_compute_introspective_imitation_transition_probability_for_homogeneous_intensity():
    """
    Tests that passing a float, np.float(32), and np.float(64) returns
    the same value as passing a numpy.array in introspective imitation dynamics"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i for _,i in enumerate(state)])

    float_selection_intensity = 0.5
    npfloat32_selection_intensity = np.float32(0.5)
    npfloat64_selection_intensity = np.float64(0.5)
    array_selection_intensity = np.full(shape=(3,3), fill_value=0.5)

    float_choice_intensity = 0.5
    npfloat32_choice_intensity = np.float32(0.5)
    npfloat64_choice_intensity = np.float64(0.5)
    array_choice_intensity = np.full(shape=(3,2), fill_value=0.5)

    source = np.array([0,1,0])
    target = np.array([1,1,0])

    float_transition_probability = ludics.compute_introspective_imitation_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        selection_intensity=float_selection_intensity,
        choice_intensity=float_choice_intensity,
        fitness_map=ludics.linear_fitness_map
    )

    np32_transition_probability = ludics.compute_introspective_imitation_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        selection_intensity=npfloat32_selection_intensity,
        choice_intensity=npfloat32_choice_intensity,
        fitness_map=ludics.linear_fitness_map
    )

    np64_transition_probability = ludics.compute_introspective_imitation_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        selection_intensity=npfloat64_selection_intensity,
        choice_intensity=npfloat64_choice_intensity,
        fitness_map=ludics.linear_fitness_map
    )
    
    array_transition_probability = ludics.compute_introspective_imitation_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        selection_intensity=array_selection_intensity,
        choice_intensity=array_choice_intensity,
        fitness_map=ludics.linear_fitness_map
    )

    assert float_transition_probability == np32_transition_probability == np64_transition_probability == array_transition_probability

def test_compute_moran_transition_probability_default_fitness_map():
    """
    Tests that the default map of the Moran process works as expected"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i for _,i in enumerate(state)])

    selection_intensity = 0.5

    source = np.array([0,1])
    target = np.array([0,0])

    transition_probability = ludics.compute_moran_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        selection_intensity=selection_intensity,
    )

    expected_transition_probability = 1/6

    assert transition_probability == expected_transition_probability

def test_compute_introspective_imitation_transition_probability_default_fitness_map():
    """
    Tests that the default map of introspective imitation dynamics works as
    expected"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i for _,i in enumerate(state)])

    selection_intensity = 0.5
    choice_intensity = 0.2

    source = np.array([0,1])
    target = np.array([0,0])

    transition_probability = ludics.compute_introspective_imitation_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        selection_intensity=selection_intensity,
        choice_intensity=choice_intensity
    )

    expected_transition_probability = 0.07502766711

    np.testing.assert_almost_equal(transition_probability,expected_transition_probability)

def test_compute_moran_transition_probability_returns_zero_for_infeasible_transition():
    """Tests that when transitioning between two states which have distance 1 but have an
    infeasible transition (here we use (0,1,1) -> (0,1,2)) the Moran process returns 0"""

    def trivial_fitness_function(source):
        return np.array([1 for _ in source])

    selection_intensity=0.1
    source = np.array([0,1,1])
    target = np.array([0,1,2])

    actual_transition_probability = ludics.compute_moran_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        selection_intensity=selection_intensity
    )

    expected_transition_probability = 0

    assert actual_transition_probability == expected_transition_probability

def test_compute_fermi_transition_probability_returns_zero_for_infeasible_transition():
    """Tests that when transitioning between two states which have distance 1 but have an
    infeasible transition (here we use (0,1,1) -> (0,1,2)) Fermi imitation dynamics returns 0"""

    def trivial_fitness_function(source):
        return np.array([400 for _ in source])

    choice_intensity=0.1
    source = np.array([0,1,1])
    target = np.array([0,1,2])

    actual_transition_probability = ludics.compute_fermi_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity
    )

    expected_transition_probability = 0

    assert actual_transition_probability == expected_transition_probability

def test_compute_introspective_imitation_transition_probability_returns_zero_for_infeasible_transition():
    """Tests that when transitioning between two states which have distance 1 but have an
    infeasible transition (here we use (0,1,1) -> (0,1,2)) introspective imitation dynamics returns 0"""

    def trivial_fitness_function(source):
        return np.array([7 for _ in source])

    choice_intensity=0.1
    selection_intensity=0.1
    source = np.array([0,1,1])
    target = np.array([0,1,2])

    actual_transition_probability = ludics.compute_introspective_imitation_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
        selection_intensity=selection_intensity
    )

    expected_transition_probability = 0

    assert actual_transition_probability == expected_transition_probability

def test_get_different_indices_for_feasible_transition():
    """
    Tests that get_different_indices correctly returns an array with the right
    index of difference for a feasible transition"""

    source = np.array([0,1,1])
    target = np.array([0,1,2])

    actual_different_indices = ludics.get_different_indices(source=source, target=target)
    expected_different_indices = np.array([2])

    assert actual_different_indices == expected_different_indices

def test_get_different_indices_for_infeasible_transition():
    """
    Tests that get_different_indices correctly returns an array with the right
    index of difference for an infeasible transition"""

    source = np.array([0,1,1])
    target = np.array([0,2,2])

    actual_different_indices = ludics.get_different_indices(source=source, target=target)
    expected_different_indices = np.array([1,2])

    np.testing.assert_array_equal(actual_different_indices,expected_different_indices)

def test_get_different_indices_for_self_transition():
    """
    Tests that get_different_indices correctly returns an array with the right
    index of difference for an infeasible transition"""

    source = np.array([0,1,1])
    target = np.array([0,1,1])

    actual_different_indices = ludics.get_different_indices(source=source, target=target)
    expected_different_indices = np.array([])

    np.testing.assert_array_equal(actual_different_indices,expected_different_indices)

def test_check_valid_extrinsic_transition_for_valid_extrinsic_transition():
    """
    Checks that check_valid_extrinsic_transition correctly returns True for a 
    valid extrinsic transition"""

    source = np.array([12,1])
    target = np.array([12,12])
    assert ludics.check_valid_extrinsic_transition(source=source, target=target) is True

def test_check_valid_extrinsic_transition_for_invalid_extrinsic_transition():
    """
    Checks that check_valid_extrinsic_transition correctly returns False for an
    invalid extrinsic transition"""

    source = np.array([12,1])
    target = np.array([12,15])
    assert ludics.check_valid_extrinsic_transition(source=source, target=target) is False

    source = np.array([12,1])
    target = np.array([12,1])
    assert ludics.check_valid_extrinsic_transition(source=source, target=target) is False

    source = np.array([12,1])
    target = np.array([15,12])
    assert ludics.check_valid_extrinsic_transition(source=source, target=target) is False