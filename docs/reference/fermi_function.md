# fermi_imitation_function

```
ludics.fermi_imitation_function(delta, choice_intensity=0.5, **kwargs)
```

returns the value of the Fermi logit function $\frac{1}{1 +
e^{\beta\Delta(f)}}$

### Parameters:

- `delta`: _float_ - the difference between a player's current fitness and the
  fitness they consider
- `choice_intensity`: _float_ - the choice intensity of the equation

### Returns:

- _float_ - the value of the Fermi logit function
