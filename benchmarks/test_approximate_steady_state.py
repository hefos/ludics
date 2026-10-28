import numpy as np
import pytest

import ludics


def generate_doubly_stochastic_cycle_matrix(n, prob_prior):
    """Returns a random walk on a cycle. Each column sums to one as well as
    each row, so the uniform distribution is already stationary."""

    identity = np.eye(n)
    return prob_prior * (np.roll(identity, -1, axis=1)) + (1 - prob_prior) * (
        np.roll(identity, 1, axis=1)
    )


def generate_biased_cycle_matrix(n):
    """Returns a lazy random walk on a cycle whose forward probability varies
    with the state. The columns do not sum to one, so the stationary
    distribution is not uniform, and the self loops make it aperiodic. This is
    the sparse chain the solver is actually asked about."""

    transition_matrix = np.zeros((n, n))
    for state in range(n):
        forward_probability = 0.2 + 0.5 * (state / n)
        transition_matrix[state, state] = 0.2
        transition_matrix[state, (state + 1) % n] = 0.8 * forward_probability
        transition_matrix[state, (state - 1) % n] = 0.8 * (1 - forward_probability)

    return transition_matrix


def generate_dense_stochastic_matrix(n, seed=0):
    """Returns a dense ergodic chain whose stationary distribution is not
    uniform, which takes the dense branch of the solver."""

    generator = np.random.default_rng(seed)
    transition_matrix = generator.random((n, n)) + 0.01

    return transition_matrix / transition_matrix.sum(axis=1, keepdims=True)


def generate_dense_slowly_mixing_matrix(
    n, small_fraction=0.1, coupling_out=0.01, coupling_back=0.001, seed=0
):
    """Returns a dense ergodic chain made of two blocks of very different size
    and stickiness, weakly coupled and coupled unequally, so its stationary
    distribution sits far from uniform. Every entry is non zero, so it takes
    the dense branch of the solver, and an iteration started from uniform has
    to carry mass between the blocks at the rate they are coupled. A dense
    chain which mixes quickly is the best case an iteration can have, so this
    is the other end of that range."""

    generator = np.random.default_rng(seed)
    small = max(2, int(n * small_fraction))

    transition_matrix = np.zeros((n, n))
    transition_matrix[:small, :small] = (
        generator.random((small, small)) + 0.5
    ) * (1 - coupling_out)
    transition_matrix[small:, small:] = (
        generator.random((n - small, n - small)) + 0.5
    ) * (1 - coupling_back)
    transition_matrix[:small, small:] = coupling_out / (n - small)
    transition_matrix[small:, :small] = coupling_back / small

    return transition_matrix / transition_matrix.sum(axis=1, keepdims=True)


def build_population_transition_matrix(n, k, mutation_probability):
    """Returns the transition matrix of a population under imispection
    dynamics, which is the kind of chain compute_steady_state is asked for in
    practice: large, sparse, and slow to mix when the mutation is small."""

    def fitness_function(state, **kwargs):
        return np.array([i + j for i, j in enumerate(state)])

    state_space = ludics.get_state_space(N=n, k=k)

    return ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=fitness_function,
        compute_transition_probability=(
            ludics.compute_introspective_imitation_transition_probability
        ),
        individual_to_action_mutation_probability=np.full(
            shape=(n, k), fill_value=mutation_probability
        ),
        selection_intensity=np.full(shape=(n, n), fill_value=0.5),
        choice_intensity=np.full(shape=(n, k), fill_value=0.5),
        number_of_strategies=k,
        fitness_map=ludics.exponential_fitness_map,
    )


def test_generate_doubly_stochastic_cycle_matrix():
    """Tests that generate_doubly_stochastic_cycle_matrix is correct for both
    the deterministic cycle and the stochastic valency"""

    deterministic_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    np.testing.assert_array_equal(
        deterministic_matrix,
        generate_doubly_stochastic_cycle_matrix(n=3, prob_prior=0),
    )

    stochastic_matrix = np.array(
        [[0, 0.3, 0, 0.7], [0.7, 0, 0.3, 0], [0, 0.7, 0, 0.3], [0.3, 0, 0.7, 0]]
    )
    np.testing.assert_array_almost_equal(
        stochastic_matrix,
        generate_doubly_stochastic_cycle_matrix(n=4, prob_prior=0.7),
    )


def test_generate_biased_cycle_matrix():
    """Tests that generate_biased_cycle_matrix gives a chain whose rows sum to
    one but whose columns do not, which is what keeps its stationary
    distribution away from uniform"""

    transition_matrix = generate_biased_cycle_matrix(n=8)

    np.testing.assert_allclose(transition_matrix.sum(axis=1), np.ones(8))
    assert not np.allclose(transition_matrix.sum(axis=0), np.ones(8))


def test_generate_dense_stochastic_matrix():
    """Tests that generate_dense_stochastic_matrix gives a dense chain whose
    rows sum to one"""

    transition_matrix = generate_dense_stochastic_matrix(n=6)

    np.testing.assert_allclose(transition_matrix.sum(axis=1), np.ones(6))
    assert np.count_nonzero(transition_matrix) == 36


@pytest.mark.parametrize("n", [20, 100, 400])
def test_compute_steady_state_for_a_sparse_chain(n, benchmark):
    """Benchmarks compute_steady_state on a sparse ergodic chain, which takes
    the sparse branch of the solver"""

    transition_matrix = generate_biased_cycle_matrix(n)
    benchmark(ludics.compute_steady_state, transition_matrix)


def test_generate_dense_slowly_mixing_matrix():
    """Tests that generate_dense_slowly_mixing_matrix gives a dense chain whose
    rows sum to one, which crosses between its blocks rarely, and whose
    stationary distribution is therefore far from uniform"""

    transition_matrix = generate_dense_slowly_mixing_matrix(n=20)

    np.testing.assert_allclose(transition_matrix.sum(axis=1), np.ones(20))
    assert np.count_nonzero(transition_matrix) == 400
    assert transition_matrix[:2, 2:].sum() < 0.05
    assert transition_matrix[2:, :2].sum() < 0.05


@pytest.mark.parametrize("n", [20, 100, 200])
def test_compute_steady_state_for_a_dense_chain(n, benchmark):
    """Benchmarks compute_steady_state on a dense ergodic chain, which takes
    the dense branch of the solver"""

    transition_matrix = generate_dense_stochastic_matrix(n)
    benchmark(ludics.compute_steady_state, transition_matrix)


@pytest.mark.parametrize("n", [20, 100, 200])
def test_compute_steady_state_for_a_dense_slowly_mixing_chain(n, benchmark):
    """Benchmarks compute_steady_state on a dense chain which mixes slowly,
    where the number of steps an iteration needs is large even though every
    entry is non zero"""

    transition_matrix = generate_dense_slowly_mixing_matrix(n)
    benchmark(ludics.compute_steady_state, transition_matrix)


@pytest.mark.parametrize("n", [200])
def test_compute_steady_state_for_a_doubly_stochastic_chain(n, benchmark):
    """Benchmarks compute_steady_state on a doubly stochastic chain. The
    uniform distribution is already stationary here, so an iteration started
    from it stops at once and nothing can beat it. We keep one such case so
    that the floor is visible rather than hidden in a grid of them."""

    transition_matrix = generate_doubly_stochastic_cycle_matrix(n, 0.5)
    benchmark(ludics.compute_steady_state, transition_matrix)


def test_compute_steady_state_for_specific_four_by_four(benchmark):
    """A benchmark with a specific 4 by 4 matrix"""

    transition_matrix = np.array(
        [
            [0.5, 0, 0.2, 0.3],
            [0.1, 0.7, 0.2, 0],
            [0.3, 0.3, 0.3, 0.1],
            [0.3, 0.1, 0.1, 0.5],
        ]
    )

    benchmark(ludics.compute_steady_state, transition_matrix)


@pytest.mark.parametrize("n, k", [(4, 3), (6, 2), (6, 3), (7, 3)])
def test_compute_steady_state_for_a_population(n, k, benchmark):
    """Benchmarks compute_steady_state for the transition matrix of a
    population with mutation, which is ergodic and so has a single stationary
    distribution to solve for"""

    transition_matrix = build_population_transition_matrix(
        n=n, k=k, mutation_probability=0.001
    )
    benchmark(ludics.compute_steady_state, transition_matrix)


@pytest.mark.parametrize("mutation_probability", [0.01, 0.001, 0.0001])
def test_compute_steady_state_for_a_weakly_mixing_population(
    mutation_probability, benchmark
):
    """Benchmarks compute_steady_state as the mutation falls and the chain
    takes longer to mix, which is what decides how many steps an iteration
    needs"""

    transition_matrix = build_population_transition_matrix(
        n=6, k=3, mutation_probability=mutation_probability
    )
    benchmark(ludics.compute_steady_state, transition_matrix)


@pytest.mark.parametrize("n, k", [(4, 3), (6, 2), (6, 3), (7, 3)])
def test_compute_steady_state_for_an_absorbing_population(n, k, benchmark):
    """Benchmarks compute_steady_state for a population without mutation,
    whose monomorphic states are absorbing. The chain then has a family of
    stationary distributions rather than one, so the answer comes from
    iterating either way: this measures what looking for a direct solve costs
    when there is none to find."""

    transition_matrix = build_population_transition_matrix(
        n=n, k=k, mutation_probability=0
    )
    benchmark(ludics.compute_steady_state, transition_matrix)
