"""Mass-spring-damper: the linear sanity plant.

x = [p, v], u = force. The plant is linear and time invariant, so the
fundamental lemma behind DeePC holds exactly for noise-free data.
"""
import numpy as np

from .base import Plant, stack


class MassSpringDamper(Plant):
    def __init__(self):
        super().__init__(
            name="msd",
            nx=2,
            nu=1,
            dt=0.05,
            state_names=["p", "v"],
            input_names=["F"],
            true_params=dict(m=1.0, k=2.0, c=0.4),
            prior_params=dict(m=0.7, k=0.8, c=0.1),
            u_min=np.array([-4.0]),
            u_max=np.array([4.0]),
            pos_idx=[0],
            vel_idx=[1],
            feature_idx=[0, 1],
        )

    def f(self, x, u, p, lib=np):
        pos, vel = x[0], x[1]
        acc = (u[0] - p["k"] * pos - p["c"] * vel) / p["m"]
        return stack(lib, [vel, acc])

    def equilibrium_input(self, x_ref, p):
        return np.array([p["k"] * x_ref[0]])
