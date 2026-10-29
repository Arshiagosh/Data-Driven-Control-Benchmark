"""Planar quadrotor.

x = [px, pz, phi, vx, vz, omega], u = [T1, T2] (rotor thrusts).
The true plant is heavier than the prior model and has linear drag and a
constant wind force in x, none of which the prior model knows.
"""
import numpy as np

from .base import Plant, stack


class PlanarQuadrotor(Plant):
    def __init__(self):
        super().__init__(
            name="quadrotor",
            nx=6,
            nu=2,
            dt=0.05,
            state_names=["px", "pz", "phi", "vx", "vz", "omega"],
            input_names=["T1", "T2"],
            true_params=dict(m=1.0, J=0.015, r=0.15, cd=0.4, wind=0.6, g=9.81),
            prior_params=dict(m=0.8, J=0.015, r=0.15, cd=0.0, wind=0.0, g=9.81),
            u_min=np.array([0.0, 0.0]),
            u_max=np.array([8.0, 8.0]),
            pos_idx=[0, 1, 2],
            vel_idx=[3, 4, 5],
            feature_idx=[2, 3, 4, 5],
        )

    def f(self, x, u, p, lib=np):
        phi, vx, vz, om = x[2], x[3], x[4], x[5]
        thrust = u[0] + u[1]
        ax = (-thrust * lib.sin(phi) - p["cd"] * vx + p["wind"]) / p["m"]
        az = (thrust * lib.cos(phi) - p["cd"] * vz) / p["m"] - p["g"]
        al = (u[1] - u[0]) * p["r"] / p["J"]
        return stack(lib, [vx, vz, om, ax, az, al])

    def equilibrium_input(self, x_ref, p):
        # Hover at phi = 0. With wind there is no exact hover at phi = 0; the
        # learned and nominal controllers only use this as a cost reference.
        h = p["m"] * p["g"] / 2.0
        return np.array([h, h])
