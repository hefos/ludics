import pytest
import numpy as np
import ludics.main
import ludics.fitness_functions

n_range = 200
time_steps_range = 10000
n_step_size = 10
time_step_step_size = 500
number_of_strategies_range = 10
number_of_strategies_step_size = 1


@pytest.mark.parametrize(
    "n, time_steps, number_of_strategies",
    [
        (n_val, time_steps_val, number_of_strategies_val)
        for n_val in range(2, n_range, n_step_size)
        for time_steps_val in range(500, time_steps_range, time_step_step_size)
        for number_of_strategies_val in range(
            2, number_of_strategies_range, number_of_strategies_step_size
        )
    ],
)
def test_simulate_markov_chain_for_moran_process(
    n, time_steps, number_of_strategies, benchmark
):
    """
    benchmarks simulate_markov_chain when using the moran process"""

    def fitness_function(state):
        return (i + j for i, j in enumerate(state))

    selection_intensity = 0.5
    initial_state = np.zeros(n)
    initial_state[0] = 1

    benchmark(
        ludics.main.simulate_markov_chain,
        initial_state=initial_state,
        number_of_strategies=number_of_strategies,
        fitness_function=fitness_function,
        compute_transition_probability=ludics.main.compute_moran_transition_probability,
        seed=2,
        time_steps=time_steps,
        selection_intensity=selection_intensity,
    )


@pytest.mark.parametrize(
    "n, time_steps, number_of_strategies",
    [
        (n_val, time_steps_val, number_of_strategies_val)
        for n_val in range(2, n_range, n_step_size)
        for time_steps_val in range(500, time_steps_range, time_step_step_size)
        for number_of_strategies_val in range(
            2, number_of_strategies_range, number_of_strategies_step_size
        )
    ],
)
def test_simulate_markov_chain_for_introspection(
    n, time_steps, number_of_strategies, benchmark
):
    """
    benchmarks simulate_markov_chain when using introspection dynamics"""

    def fitness_function(state):
        return (i + j for i, j in enumerate(state))

    choice_intensity = 0.5
    initial_state = np.zeros(n)
    initial_state[0] = 1

    benchmark(
        ludics.main.simulate_markov_chain,
        initial_state=initial_state,
        number_of_strategies=number_of_strategies,
        fitness_function=fitness_function,
        compute_transition_probability=ludics.main.compute_introspection_transition_probability,
        seed=2,
        time_steps=time_steps,
        choice_intensity=choice_intensity,
    )


@pytest.mark.parametrize(
    "n, time_steps, number_of_strategies",
    [
        (n_val, time_steps_val, number_of_strategies_val)
        for n_val in range(2, n_range, n_step_size)
        for time_steps_val in range(500, time_steps_range, time_step_step_size)
        for number_of_strategies_val in range(
            2, number_of_strategies_range, number_of_strategies_step_size
        )
    ],
)
def test_simulate_markov_chain_for_fermi(
    n, time_steps, number_of_strategies, benchmark
):
    """
    benchmarks simulate_markov_chain when using fermi imitation dynamics"""

    def fitness_function(state):
        return (i + j for i, j in enumerate(state))

    choice_intensity = 0.5
    initial_state = np.zeros(n)
    initial_state[0] = 1

    benchmark(
        ludics.main.simulate_markov_chain,
        initial_state=initial_state,
        number_of_strategies=number_of_strategies,
        fitness_function=fitness_function,
        compute_transition_probability=ludics.main.compute_fermi_transition_probability,
        seed=2,
        time_steps=time_steps,
        choice_intensity=choice_intensity,
    )


@pytest.mark.parametrize(
    "n, time_steps, number_of_strategies",
    [
        (n_val, time_steps_val, number_of_strategies_val)
        for n_val in range(2, n_range, n_step_size)
        for time_steps_val in range(500, time_steps_range, time_step_step_size)
        for number_of_strategies_val in range(
            2, number_of_strategies_range, number_of_strategies_step_size
        )
    ],
)
def test_simulate_markov_chain_for_imispection(
    n, time_steps, number_of_strategies, benchmark
):
    """
    benchmarks simulate_markov_chain when using imispection dynamics"""

    def fitness_function(state):
        return (i + j for i, j in enumerate(state))

    choice_intensity = 0.5
    selection_intensity = 0.5
    initial_state = np.zeros(n)
    initial_state[0] = 1

    benchmark(
        ludics.main.simulate_markov_chain,
        initial_state=initial_state,
        number_of_strategies=number_of_strategies,
        fitness_function=fitness_function,
        compute_transition_probability=ludics.main.compute_fermi_transition_probability,
        seed=2,
        time_steps=time_steps,
        choice_intensity=choice_intensity,
        selection_intensity=selection_intensity,
    )
