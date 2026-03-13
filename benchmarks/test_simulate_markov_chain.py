import pytest
import numpy as np
import ludics.main

max_n = 102
max_iterations = 5000
n_step_size = 10
iteration_step_size = 1000


@pytest.mark.parametrize(
    "n, iterations",
    [
        (n_val, iterations_val)
        for n_val in range(2, max_n, n_step_size)
        for iterations_val in range(1000, max_iterations, iteration_step_size)
    ],
)
def test_simulate_markov_chain_for_moran_process(
    n, iterations, benchmark
):
    """
    benchmarks simulate_markov_chain when using the moran process"""

    def fitness_function(state, **kwargs):
        return np.array([i + j for i, j in enumerate(state)])

    selection_intensity = 0.5
    initial_state = np.zeros(n, dtype=int)
    initial_state[0] = 1
    number_of_strategies = 2

    benchmark(
        ludics.main.simulate_markov_chain,
        initial_state=initial_state,
        number_of_strategies=number_of_strategies,
        fitness_function=fitness_function,
        compute_transition_probability=ludics.main.compute_moran_transition_probability,
        seed=2,
        iterations=iterations,
        selection_intensity=selection_intensity,
    )


@pytest.mark.parametrize(
    "n, iterations",
    [
        (n_val, iterations_val)
        for n_val in range(2, max_n, n_step_size)
        for iterations_val in range(500, max_iterations, iteration_step_size)
    ],
)
def test_simulate_markov_chain_for_introspection(
    n, iterations, benchmark
):
    """
    benchmarks simulate_markov_chain when using introspection dynamics"""

    def fitness_function(state, **kwargs):
        return np.array([i + j for i, j in enumerate(state)])

    choice_intensity = 0.5
    initial_state = np.zeros(n, dtype=int)
    initial_state[0] = 1
    number_of_strategies = 2

    benchmark(
        ludics.main.simulate_markov_chain,
        initial_state=initial_state,
        number_of_strategies=number_of_strategies,
        fitness_function=fitness_function,
        compute_transition_probability=ludics.main.compute_introspection_transition_probability,
        seed=2,
        iterations=iterations,
        choice_intensity=choice_intensity,
    )


@pytest.mark.parametrize(
    "n, iterations",
    [
        (n_val, iterations_val)
        for n_val in range(2, max_n, n_step_size)
        for iterations_val in range(500, max_iterations, iteration_step_size)
    ],
)
def test_simulate_markov_chain_for_fermi(
    n, iterations, benchmark
):
    """
    benchmarks simulate_markov_chain when using fermi imitation dynamics"""

    def fitness_function(state, **kwargs):
        return np.array([i + j for i, j in enumerate(state)])

    choice_intensity = 0.5
    initial_state = np.zeros(n, dtype=int)
    initial_state[0] = 1
    number_of_strategies = 2

    benchmark(
        ludics.main.simulate_markov_chain,
        initial_state=initial_state,
        number_of_strategies=number_of_strategies,
        fitness_function=fitness_function,
        compute_transition_probability=ludics.main.compute_fermi_transition_probability,
        seed=2,
        iterations=iterations,
        choice_intensity=choice_intensity,
    )


@pytest.mark.parametrize(
    "n, iterations",
    [
        (n_val, iterations_val)
        for n_val in range(2, max_n, n_step_size)
        for iterations_val in range(500, max_iterations, iteration_step_size)
    ],
)
def test_simulate_markov_chain_for_imispection(
    n, iterations, benchmark
):
    """
    benchmarks simulate_markov_chain when using imispection dynamics"""

    def fitness_function(state, **kwargs):
        return np.array([i + j for i, j in enumerate(state)])

    choice_intensity = 0.5
    selection_intensity = 0.5
    initial_state = np.zeros(n, dtype=int)
    initial_state[0] = 1
    number_of_strategies = 2

    benchmark(
        ludics.main.simulate_markov_chain,
        initial_state=initial_state,
        number_of_strategies=number_of_strategies,
        fitness_function=fitness_function,
        compute_transition_probability=ludics.main.compute_fermi_transition_probability,
        seed=2,
        iterations=iterations,
        choice_intensity=choice_intensity,
        selection_intensity=selection_intensity,
    )
