# exponentiate

```
ludics.exponentiate(exponent)
```

Raises $e$ to an exponent, symbolically only when it has to be

Raising `sympy.E` to a float builds a symbolic object. That is correct, but it
costs far more than the arithmetic it stands in for, and solving a chain
exactly evaluates these maps once for every neighbour of every state. We
therefore use `numpy` whenever the exponent is numeric, and fall back to
`sympy` for symbols and for arrays which hold them.

### Parameters:

- `exponent`: _numpy.array_, _float_ or _sympy.Symbol_ - the exponent

### Returns:

- _numpy.array_, _numpy.float64_ or _sympy.Expr_ - $e$ raised to the exponent

### Example

```py
>>> import ludics
>>> import numpy as np
>>> import sympy as sym

>>> ludics.exponentiate(0)
np.float64(1.0)

>>> beta = sym.Symbol("beta")
>>> ludics.exponentiate(beta)
exp(beta)

```
