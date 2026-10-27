import numpy as np
import pytest

import ludics

n_range = 200
step_size_n = 10
prob_range = 20
step_size_prob = 2


def generate_stochastic_2_valency_cycle_matrix(n, prob_prior):
    identity = np.eye(n)
    return prob_prior * (np.roll(identity, -1, axis=1)) + (1 - prob_prior) * (
        np.roll(identity, 1, axis=1)
    )


def generate_full_uniform_matrix(n):
    return np.full((n, n), 1 / n)


def test_generate_stochastic_2_valency_cycle_matrix():
    """Tests that generate_stochastic_2_valency_cycle_matrix is correct for the
    case of both deterministic cycle and stochastic valency"""

    deterministic_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    np.testing.assert_array_equal(
        deterministic_matrix,
        generate_stochastic_2_valency_cycle_matrix(n=3, prob_prior=0),
    )

    stochastic_matrix = np.array(
        [[0, 0.3, 0, 0.7], [0.7, 0, 0.3, 0], [0, 0.7, 0, 0.3], [0.3, 0, 0.7, 0]]
    )
    np.testing.assert_array_almost_equal(
        stochastic_matrix,
        generate_stochastic_2_valency_cycle_matrix(n=4, prob_prior=0.7),
    )


def test_generate_full_uniform_matrix():
    """Tests that generate_full_uniform_matrix is correct"""

    uniform_matrix = np.array(
        [
            [1 / 4, 1 / 4, 1 / 4, 1 / 4],
            [1 / 4, 1 / 4, 1 / 4, 1 / 4],
            [1 / 4, 1 / 4, 1 / 4, 1 / 4],
            [1 / 4, 1 / 4, 1 / 4, 1 / 4],
        ]
    )
    np.testing.assert_array_equal(uniform_matrix, generate_full_uniform_matrix(4))


@pytest.mark.parametrize("n", range(2, n_range, step_size_n))
def test_compute_steady_state_for_deterministic_cycle_matrix(n, benchmark):
    """Benchmarks compute_steady_state for the deterministic cycle
    matrix"""

    transition_matrix = generate_stochastic_2_valency_cycle_matrix(n, 0)
    benchmark(ludics.compute_steady_state, transition_matrix)


@pytest.mark.parametrize(
    "n, prob_prior",
    [
        (n, p)
        for n in range(2, n_range, step_size_n)
        for p in [1 / (i) for i in range(1, prob_range + 1, step_size_prob)]
    ],
)
def test_compute_steady_state_for_stochastic_2_valency_cycle_matrix(
    n, prob_prior, benchmark
):
    """Benchmarks compute_steady_state for the stochastic 2-valency cycle
    matrix"""

    transition_matrix = generate_stochastic_2_valency_cycle_matrix(n, prob_prior)
    benchmark(ludics.compute_steady_state, transition_matrix)


@pytest.mark.parametrize("n", range(2, n_range, step_size_n))
def test_compute_steady_state_for_full_uniform_matrix(n, benchmark):
    """Benchmarks compute_steady_state for the full uniform
    matrix"""

    transition_matrix = generate_full_uniform_matrix(n)
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


@pytest.mark.parametrize(
    "n, k",
    [(n_val, k_val) for n_val in range(2, 9) for k_val in range(2, 4)],
)
def test_compute_steady_state_for_a_population(n, k, benchmark):
    """Benchmarks compute_steady_state for the transition matrix of a
    population"""

    transition_matrix = build_population_transition_matrix(
        n=n, k=k, mutation_probability=0.001
    )
    benchmark(ludics.compute_steady_state, transition_matrix)


@pytest.mark.parametrize("mutation_probability", [0.1, 0.01, 0.001, 0.0001])
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
