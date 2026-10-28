"""
Numeric helpers shared across the library.

Several of the maps in `ludics` are used both with floats, when a chain is
solved or simulated, and with sympy symbols, when a transition probability is
derived in closed form. The helpers here carry that dual use, so that the maps
themselves do not each have to.
"""

import numpy as np
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
