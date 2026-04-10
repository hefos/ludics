# Welcome to `ludics`!

`ludics` is a Python library for modelling and analysing evolutionary games as
Markov chains. It supports exact symbolic computation via
[SymPy](https://www.sympy.org/) as well as numerical approximation, and handles
both absorbing and ergodic chains.

## Implemented evolutionary dynamics

| Dynamic         | Type                       | Chain structure |
| --------------- | -------------------------- | --------------- |
| Moran process   | Extrinsic (imitation)      | Absorbing       |
| Fermi imitation | Extrinsic (imitation)      | Absorbing       |
| Introspection   | Intrinsic                  | Ergodic         |
| Aspiration      | Intrinsic (binary actions) | Ergodic         |

## Built-in fitness functions

- **Homogeneous Public Goods Game**: all players contribute equally
- **Heterogeneous Public Goods Game**: player-specific contribution amounts
- **General symbolic 4-state**: fully symbolic payoffs for 2-player, 2-action games

## Key features

- Build transition matrices for any combination of dynamic and fitness function
- Compute **fixation probabilities** (absorbing chains) or **stationary
  distributions** (ergodic chains), exactly or numerically; works on any
  Markov chain
- Add **mutation** between strategies via a per-player mutation matrix
- **Simulate** trajectories forward in time when the state space is too large
  for exact computation
- Full **symbolic support**: work with SymPy expressions and simplify results
  algebraically

## Get started

- Get a handle on the basics with our [tutorial](tutorial/basics.md)
- Learn how to use `ludics` with our [how-to guides](how-to/how-to.md)
- Explore the functionality with our API [reference](reference/functionality.md) section
- Dive deeper into evolutionary games with our [explanation](explanation/explanation_reference.md) section
- View the source code on [github](https://github.com/hefos/ludics)
