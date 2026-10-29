"""Common plant interface.

A plant is a continuous-time model x_dot = f(x, u, p). The same function is used
for simulation (NumPy) and for optimisation (CasADi) through the ``lib`` argument,
so the true plant, the prior model and the oracle model share one code path.
"""
from dataclasses import dataclass, field

import numpy as np


def stack(lib, items):
    """Stack scalar expressions into a column for NumPy or CasADi."""
    if lib is np:
        return np.array(items, dtype=float)
    return lib.vertcat(*items)


@dataclass
class Plant:
    name: str
    nx: int
    nu: int
    dt: float
    state_names: list
    input_names: list
    true_params: dict
    prior_params: dict
    u_min: np.ndarray
    u_max: np.ndarray
    # Position-like and velocity-like state indices, paired element by element:
    # d/dt x[pos_idx[i]] = x[vel_idx[i]] in every plant of this benchmark.
    pos_idx: list = field(default_factory=list)
    vel_idx: list = field(default_factory=list)
    # States (besides the inputs) that the learned models use as features.
    # Translation-invariant positions are left out.
    feature_idx: list = field(default_factory=list)

    def f(self, x, u, p, lib=np):
        raise NotImplementedError

    def equilibrium_input(self, x_ref, p):
        """Input that holds the state at x_ref under parameters p."""
        raise NotImplementedError


def rk4_step(plant, x, u, p, dt, substeps=1, lib=np):
    """Explicit RK4 with a zero-order-hold input over one sampling period."""
    h = dt / substeps
    for _ in range(substeps):
        k1 = plant.f(x, u, p, lib)
        k2 = plant.f(x + 0.5 * h * k1, u, p, lib)
        k3 = plant.f(x + 0.5 * h * k2, u, p, lib)
        k4 = plant.f(x + h * k3, u, p, lib)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
