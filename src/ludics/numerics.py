"""
Numeric helpers shared across the library.

Several of the maps in `ludics` are used both with floats, when a chain is
solved or simulated, and with sympy symbols, when a transition probability is
derived in closed form. The helpers here carry that dual use, so that the maps
themselves do not each have to.
"""

import warnings

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sympy as sym


def exponentiate(exponent):
    """
    Returns $e$ raised to an exponent, symbolically only when it has to be.

    Raising `sym.E` to a float builds a symbolic object. That is correct, but
    it costs far more than the arithmetic it stands in for, and solving a chain
    exactly evaluates these maps once for every neighbour of every state. We
    therefore use numpy whenever the exponent is numeric, and fall back to
    sympy for symbols and for arrays which hold them.

    Parameters
    -----------
    exponent: numpy.array, float or sym.Symbol, the exponent

    Returns
    --------
    numpy.array, numpy.float64 or sym.Expr: $e$ raised to the exponent"""

    if isinstance(exponent, np.ndarray):
        return sym.E**exponent if exponent.dtype == object else np.exp(exponent)
    if isinstance(exponent, (int, float, np.integer, np.floating)):
        return np.exp(exponent)

    return sym.E**exponent


def solve_stationary_distribution(
    transition_matrix, residual_tolerance=10**-8, sparse_density_threshold=0.25
):
    """
    Returns the unique stationary distribution of a transition matrix, or None.

    Writing $\\pi P = \\pi$ as $(P^T - I)\\pi^T = 0$ and replacing the last row
    with the condition that the entries sum to one gives a linear system whose
    solution is the stationary distribution. We hand that system to scipy,
    which factorises it sparsely when the matrix is sparse and densely when it
    is not, rather than iterating towards the answer.

    A chain with more than one stationary distribution, such as one with two
    absorbing states, makes the system singular. There is then no single answer
    to return, so we return None and leave the caller to iterate instead. We
    check the solution rather than trusting it, since a singular system gives
    scipy no way to signal the failure other than the answer it returns. A
    singular system is an expected outcome here rather than an error, so we
    silence the warning the sparse factorisation raises for it, which would
    otherwise point a caller at these internals for a case we handle.

    Parameters
    -----------
    transition_matrix: numpy.array or scipy.sparse array, a transition matrix

    residual_tolerance: float, how far $\\pi P$ may sit from $\\pi$ for the
    solution to be accepted

    sparse_density_threshold: float, the proportion of entries which must be
    non zero before the system is factorised densely rather than sparsely

    Returns
    --------
    numpy.array or None: the stationary distribution, or None when the chain
    does not have exactly one"""

    number_of_states = transition_matrix.shape[0]
    normalisation = np.zeros(number_of_states)
    normalisation[-1] = 1

    if scipy.sparse.issparse(transition_matrix):
        density = transition_matrix.nnz / number_of_states**2
    else:
        transition_matrix = np.asarray(transition_matrix, dtype=float)
        density = np.count_nonzero(transition_matrix) / number_of_states**2

    try:
        if density < sparse_density_threshold:
            system = (
                scipy.sparse.csc_array(transition_matrix).T
                - scipy.sparse.eye_array(number_of_states, format="csc")
            ).tolil()
            system[-1, :] = 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", scipy.sparse.linalg.MatrixRankWarning)
                solution = scipy.sparse.linalg.spsolve(system.tocsc(), normalisation)
        else:
            system = np.asarray(transition_matrix, dtype=float).T - np.eye(
                number_of_states
            )
            system[-1, :] = 1
            solution = scipy.linalg.solve(system, normalisation)
    except (ValueError, RuntimeError, scipy.linalg.LinAlgError):
        return None

    if (
        not np.all(np.isfinite(solution))
        or solution.min() < -residual_tolerance
        or abs(solution.sum() - 1) > residual_tolerance
        or np.max(np.abs(solution @ transition_matrix - solution))
        > residual_tolerance
    ):
        return None

    return solution
