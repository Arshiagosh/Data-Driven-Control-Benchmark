"""Data generation.

The pendulum and the quadrotor are open-loop unstable, so data are collected in
closed loop: an LQR designed on the *prior* model tracks random set-points while
a Gaussian dither excites the plant. The mass-spring-damper uses the same
scheme for uniformity. Every learned model in the benchmark is fitted to the
same dataset, with noisy state measurements.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_discrete_are

from .plants import rk4_step
from .simulate import true_step


def linearise(plant, x, u, p, eps=1e-6):
    """Finite-difference Jacobians of the one-step RK4 map."""
    f0 = rk4_step(plant, x, u, p, plant.dt)
    A = np.zeros((plant.nx, plant.nx))
    B = np.zeros((plant.nx, plant.nu))
    for i in range(plant.nx):
        dx = np.zeros(plant.nx)
        dx[i] = eps
        A[:, i] = (rk4_step(plant, x + dx, u, p, plant.dt) - f0) / eps
    for j in range(plant.nu):
        du = np.zeros(plant.nu)
        du[j] = eps
        B[:, j] = (rk4_step(plant, x, u + du, p, plant.dt) - f0) / eps
    return A, B


def lqr_gain(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


@dataclass
class Dataset:
    plant_name: str
    episodes: list = field(default_factory=list)  # dicts with Xm, Xt, U

    @property
    def n_samples(self):
        return sum(len(e["U"]) for e in self.episodes)

    def transitions(self):
        """Measured (x_k, u_k, x_{k+1}) stacked over all episodes."""
        Xk = np.vstack([e["Xm"][:-1] for e in self.episodes])
        Uk = np.vstack([e["U"] for e in self.episodes])
        Xk1 = np.vstack([e["Xm"][1:] for e in self.episodes])
        return Xk, Uk, Xk1


def collect(task, n_episodes=15, ep_len=100, switch_every=25, seed=0):
    """Collect ``n_episodes`` closed-loop episodes of length ``ep_len``."""
    rng = np.random.default_rng(seed)
    p = task.plant
    prior = p.prior_params
    Qd = np.eye(p.nx)
    Rd = np.eye(p.nu)
    ds = Dataset(plant_name=p.name)
    while len(ds.episodes) < n_episodes:
        x = rng.uniform(task.setpoint_low, task.setpoint_high)
        Xt = [x.copy()]
        Xm = [x + task.noise_std * rng.standard_normal(p.nx)]
        U = []
        ok = True
        for k in range(ep_len):
            if k % switch_every == 0:
                xr = rng.uniform(task.setpoint_low, task.setpoint_high)
                ur = p.equilibrium_input(xr, prior)
                A, B = linearise(p, xr, ur, prior)
                K = lqr_gain(A, B, Qd, Rd)
            u = ur - K @ (Xm[-1] - xr) + task.dither_std * rng.standard_normal(p.nu)
            u = np.clip(u, p.u_min, p.u_max)
            x = true_step(p, x, u)
            if np.any(np.abs(x) > task.abort_box):
                ok = False
                break
            U.append(u)
            Xt.append(x.copy())
            Xm.append(x + task.noise_std * rng.standard_normal(p.nx))
        if ok:
            ds.episodes.append(dict(Xm=np.array(Xm), Xt=np.array(Xt), U=np.array(U)))
    return ds
