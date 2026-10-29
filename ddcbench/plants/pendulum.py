"""Inverted pendulum, angle measured from the upright position.

x = [theta, omega], u = torque. The true plant has Coulomb-like friction
(smoothed with tanh) that the prior model does not know, plus a wrong mass and
no viscous friction in the prior.
"""
import numpy as np

from .base import Plant, stack


class Pendulum(Plant):
    def __init__(self):
        super().__init__(
            name="pendulum",
            nx=2,
            nu=1,
            dt=0.05,
            state_names=["theta", "omega"],
            input_names=["tau"],
            true_params=dict(m=1.0, l=0.5, b=0.1, bc=0.1, g=9.81),
            prior_params=dict(m=0.6, l=0.5, b=0.0, bc=0.0, g=9.81),
            u_min=np.array([-3.0]),
            u_max=np.array([3.0]),
            pos_idx=[0],
            vel_idx=[1],
            feature_idx=[0, 1],
        )

    def f(self, x, u, p, lib=np):
        th, om = x[0], x[1]
        ml2 = p["m"] * p["l"] ** 2
        acc = (p["g"] / p["l"]) * lib.sin(th) + (
            u[0] - p["b"] * om - p["bc"] * lib.tanh(om / 0.05)
        ) / ml2
        return stack(lib, [om, acc])

    def equilibrium_input(self, x_ref, p):
        return np.array([-p["m"] * p["g"] * p["l"] * np.sin(x_ref[0])])
