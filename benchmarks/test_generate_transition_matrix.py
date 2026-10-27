import numpy as np
import pytest

import ludics

max_n = 8
compute_transition_probability_functions = [ludics.compute_fermi_transition_probability, ludics.compute_moran_transition_probability, ludics.compute_introspection_transition_probability, ludics.compute_introspective_imitation_transition_probability]
n_step_size = 1
max_actions = 4
action_step_size = 1
mutation_values = [0, 0.05]

def fitness_function(state, **kwargs):
    return np.array([i + 2 for i in state])

def test_fitness_function_returns_expected_value_and_takes_kwargs():
    """
    Tests that the fitness function returns the correct value and accepts
    kwargs"""
    state = np.array([1,0,2,3])
    arbitrary_kwarg = 0.1
    expected_fitness = np.array([3,2,4,5])
    actual_fitness = fitness_function(state=state, arbitrary_kwarg=arbitrary_kwarg)
    np.testing.assert_array_equal(actual_fitness, expected_fitness)

@pytest.mark.parametrize(
        "n, actions, transition_probability, mu",
        [(n,action_types,compute_transition_probability, mu)
         for n in range(2, max_n, n_step_size)
         for action_types in range(2, max_actions, action_step_size)
         for compute_transition_probability in compute_transition_probability_functions
         for mu in mutation_values]
)
def test_generate_transition_matrix_no_mutation(n, actions, transition_probability, mu, benchmark):
    """
    Benchmarks generate_transition_matrix when there is no mutation for
    different population sizes, numbers of actions, and probability functions"""
    state_space = ludics.get_state_space(N=n, k=actions)
    selection_intensity = 2
    choice_intensity = 2
    benchmark(ludics.generate_transition_matrix, state_space=state_space,
            compute_transition_probability=transition_probability,
            individual_to_action_mutation_probability=np.full(shape=(n,actions), fill_value=mu),
            fitness_function=fitness_function,
            selection_intensity=selection_intensity,
            choice_intensity=choice_intensity,
            number_of_strategies=actions)